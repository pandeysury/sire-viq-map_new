"""
RISQ endpoints — extracted from app/main.py verbatim (structural move
only, no behavior change). Fully isolated from viq.py: no route here
reads or writes any VIQ collection or VIQ SQL table.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app import state
from app.config import settings
from app.database import get_risq_db, RisqIngestionReport, RisqQueryHistory, RisqFeedback
from app.schemas import RisqAskRequest, RisqAskResponse
from app.services import risq_engine
from app.services.risq_engine import resolve_module

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/v1/risq/ask", response_model=RisqAskResponse)
async def risq_ask(request: RisqAskRequest, db: Session = Depends(get_risq_db)):
    """RISQ equivalent of VIQ's /api/v1/search: same matching/ranking
    logic and response shape (a list of reference + question matches),
    just against RISQ-only sources. Never reads or falls back to VIQ
    collections."""
    start_time = datetime.now()
    try:
        top_k = request.top_k or settings.RISQ_DEFAULT_TOP_K
        matches = await risq_engine.risq_search_matches(query=request.query, code=request.code, top_k=top_k)
        time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        try:
            history = RisqQueryHistory(
                query=request.query,
                query_type="exact_code" if (matches and matches[0]["similarity_score"] == 1.0) else "semantic",
                result_code=matches[0]["risq_code"] if matches else None,
                confidence=matches[0]["similarity_score"] if matches else None,
                time_ms=time_ms,
            )
            db.add(history)
            db.commit()
        except Exception as e:
            logger.error(f"RISQ: failed to log query history: {str(e)}")

        return {"matches": matches, "query": request.query}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RISQ ask failed: {str(e)}")
        raise HTTPException(status_code=500, detail="RISQ query failed")


@router.post("/api/v1/risq/feedback")
async def risq_submit_feedback(request: Request, db: Session = Depends(get_risq_db)):
    """RISQ equivalent of VIQ's /api/v1/feedback: same thumbs-up / thumbs-down
    / correction self-learning loop, but fully isolated — logs to RisqFeedback
    (never UserFeedback) and, on a positive signal, embeds the finding into
    risq_training (never viq_training) tagged risq_code + mapping_status
    'matched' + source 'user_feedback'. Never touches any VIQ table or
    collection."""
    try:
        data = await request.json()

        feedback_log = RisqFeedback(
            query=data.get('finding', ''),
            suggested_risq_code=data.get('suggested_risq_code', ''),
            correct_risq_code=data.get('correct_risq_code'),
            feedback_type=data.get('feedback_type', 'unknown'),
        )
        db.add(feedback_log)
        db.commit()

        logger.info(f"RISQ feedback logged to database: {data}")

        if data.get('correct_risq_code') and data.get('finding'):
            correct_code = data['correct_risq_code']
            finding = data['finding']
        elif data.get('feedback_type') == 'thumbs_up' and data.get('suggested_risq_code') and data.get('finding'):
            correct_code = data['suggested_risq_code']
            finding = data['finding']
        else:
            correct_code = None
            finding = None

        if correct_code and finding and state.risq_collection_training is not None and state.client is not None:
            try:
                response = state.client.embeddings.create(model="text-embedding-3-small", input=finding)
                embedding = response.data[0].embedding

                import time
                feedback_id = f"risq_feedback_{int(time.time())}_{hash(finding) % 10000}"

                state.risq_collection_training.add(
                    ids=[feedback_id],
                    documents=[finding],
                    embeddings=[embedding],
                    metadatas=[{
                        'module': 'risq',
                        'document_type': 'historical_finding',
                        'risq_code': correct_code,
                        'matched_risq_code': correct_code,
                        'mapping_status': 'matched',
                        'finding': finding,
                        'source_file': 'user_feedback',
                        'source_sheet': '',
                        'row_index': -1,
                    }],
                )

                logger.info(f"RISQ: added feedback to risq_training: {finding[:50]}... -> RISQ {correct_code}")

                feedback_type = data.get('feedback_type', 'correction')
                message = "Feedback submitted and system learned from correction" if feedback_type != 'thumbs_up' else "System learned - it will give better results next time!"

                return {
                    "status": "success",
                    "message": message,
                    "learned": True,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.error(f"RISQ: failed to add feedback to risq_training: {str(e)}")

        return {
            "status": "success",
            "message": "Feedback submitted successfully",
            "learned": False,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error submitting RISQ feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit RISQ feedback")


@router.get("/api/v1/risq/stats")
async def risq_stats():
    """RISQ-equivalent of /api/v1/stats — counts from RISQ collections only."""
    try:
        return {
            "module": "risq",
            "questions_count": state.risq_collection_questions.count() if state.risq_collection_questions else 0,
            "training_count": state.risq_collection_training.count() if state.risq_collection_training else 0,
            "documents_count": state.risq_collection_documents.count() if state.risq_collection_documents else 0,
            "status": "active" if state.risq_collection_questions else "not_initialized",
        }
    except Exception as e:
        logger.error(f"RISQ stats failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get RISQ stats")


@router.get("/api/v1/risq/feedback/stats")
async def risq_get_feedback_stats(db: Session = Depends(get_risq_db)):
    """RISQ-equivalent of /api/v1/feedback/stats — same calculation, reading
    only from RisqFeedback (never UserFeedback). Also reports how many
    risq_training records were learned from feedback (source_file ==
    'user_feedback'), the RISQ-side of VIQ's 'Training Examples' count."""
    try:
        all_feedback = db.query(RisqFeedback).all()

        feedback_counts = {}
        for feedback in all_feedback:
            feedback_counts[feedback.feedback_type] = feedback_counts.get(feedback.feedback_type, 0) + 1

        total_feedback = len(all_feedback)
        positive_feedback = feedback_counts.get('thumbs_up', 0)
        negative_feedback = feedback_counts.get('thumbs_down', 0)
        corrections = feedback_counts.get('correction', 0) + feedback_counts.get('clarification', 0)

        if total_feedback > 0:
            total_score = (positive_feedback * 5) + (negative_feedback * 1) + (corrections * 3)
            average_rating = total_score / total_feedback
        else:
            average_rating = 0.0

        learned_count = 0
        if state.risq_collection_training is not None and state.risq_collection_training.count() > 0:
            try:
                learned = state.risq_collection_training.get(where={"source_file": "user_feedback"})
                learned_count = len(learned.get("ids", []))
            except Exception as e:
                logger.error(f"RISQ: failed to count learned-from-feedback rows: {str(e)}")

        return {
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "corrections_provided": corrections,
            "average_rating": round(average_rating, 2),
            "feedback_breakdown": feedback_counts,
            "user_satisfaction": round((positive_feedback / total_feedback * 100), 1) if total_feedback > 0 else 0.0,
            "learned_from_feedback": learned_count,
        }
    except Exception as e:
        logger.error(f"RISQ feedback stats failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get RISQ feedback stats")


RISQ_ALLOWED_UPLOAD_EXTENSIONS = {".xlsx", ".xls", ".pdf"}


@router.post("/api/v1/risq/upload")
async def risq_upload(file: UploadFile = File(...)):
    """Upload a RISQ source file (Excel question-master/training or PDF).
    Saves the file only; ingestion is a separate explicit /risq/train call."""
    resolve_module("risq")

    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in RISQ_ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {sorted(RISQ_ALLOWED_UPLOAD_EXTENSIONS)}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.RISQ_MAX_UPLOAD_MB:
        raise HTTPException(status_code=400, detail=f"File too large ({size_mb:.1f}MB). Max allowed: {settings.RISQ_MAX_UPLOAD_MB}MB")

    if ext == ".pdf":
        target_dir = Path(settings.RISQ_PDF_DIR)
    else:
        target_dir = Path(settings.RISQ_DATA_DIR)

    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w\.\- ]", "_", filename)
    target_path = target_dir / safe_name

    with open(target_path, "wb") as f:
        f.write(contents)

    # Identify the file by its ACTUAL columns, never by filename — a
    # question-master file has {code, viqQuestionsFullText, tmsaDescription},
    # a historical-findings file has {Date, RISQ Ref, Finding}. This is the
    # same classification "Run Training" uses, so what's reported here is
    # exactly what will be ingested, regardless of what the file is named.
    from scripts.train_risq_from_excel import (
        RISQTrainingSystem, QUESTION_MASTER_REQUIRED_COLUMNS, TRAINING_REQUIRED_COLUMNS,
    )

    if ext == ".pdf":
        detected_sheets = {}
        document_type = "pdf"
    else:
        try:
            classifier = RISQTrainingSystem()
            detected_sheets = classifier._classify_excel_sheets(target_path)
        except Exception as e:
            logger.error(f"RISQ: failed to classify uploaded file columns: {str(e)}")
            detected_sheets = {}

        detected_types = set(detected_sheets.values())
        if detected_types == {"question_master"}:
            document_type = "question_master"
        elif detected_types == {"historical_finding"}:
            document_type = "historical_finding"
        elif detected_types:
            document_type = "mixed"
        else:
            document_type = "unrecognized"

    logger.info(f"RISQ: saved uploaded file {safe_name} ({size_mb:.2f}MB), detected_type={document_type}")

    response = {
        "saved_as": safe_name,
        "size_bytes": len(contents),
        "document_type": document_type,
        "detected_sheets": detected_sheets,
    }
    if document_type == "unrecognized":
        response["message"] = (
            f"No sheet in '{safe_name}' matched the required RISQ question-master columns "
            f"({sorted(QUESTION_MASTER_REQUIRED_COLUMNS)}) or historical-finding columns "
            f"({sorted(TRAINING_REQUIRED_COLUMNS)}). It was saved but will be skipped by Run Training."
        )
    return response


@router.post("/api/v1/risq/train")
async def risq_train(db: Session = Depends(get_risq_db)):
    """Trigger RISQ ingestion (question-master + historical findings) using
    the same idempotent, upsert-based logic as scripts/train_risq_from_excel.py.
    Never touches any VIQ collection or VIQ table."""
    resolve_module("risq")
    try:
        from scripts.train_risq_from_excel import RISQTrainingSystem

        # Reuse the server's own long-lived chroma_client instead of letting
        # RISQTrainingSystem open a second, independent PersistentClient —
        # otherwise training writes through one connection while /risq/ask
        # keeps reading through another, and the server would go on serving
        # pre-training results until it's restarted.
        trainer = RISQTrainingSystem(chroma_client=state.risq_chroma_client)
        reports = trainer.run_full_ingestion()

        saved_reports = []
        for r in reports:
            row = RisqIngestionReport(
                source_file=r["source_file"],
                document_type=r["document_type"],
                file_checksum=r.get("file_checksum"),
                total_rows=r.get("total_rows", 0),
                embedded=r.get("embedded", 0),
                upserted=r.get("upserted", 0),
                matched=r.get("matched", 0),
                unmatched=r.get("unmatched", 0),
                invalid=r.get("invalid", 0),
                status=r.get("status", "failed"),
                error_summary=r.get("error_summary") or ("; ".join(r.get("errors", [])[:20]) or None),
                triggered_by="api",
                completed_at=datetime.utcnow(),
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            saved_reports.append({
                "id": row.id,
                "source_file": row.source_file,
                "document_type": row.document_type,
                "total_rows": row.total_rows,
                "embedded": row.embedded,
                "upserted": row.upserted,
                "matched": row.matched,
                "unmatched": row.unmatched,
                "invalid": row.invalid,
                "status": row.status,
                "error_summary": row.error_summary,
            })

        # Refresh in-memory collection handles/counts after training
        await risq_engine.load_risq_collections()

        return {"reports": saved_reports}
    except Exception as e:
        logger.error(f"RISQ training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RISQ training failed: {str(e)}")


@router.get("/api/v1/risq/training-status")
async def risq_training_status(db: Session = Depends(get_risq_db)):
    """Latest ingestion report per source file, plus live collection counts."""
    try:
        rows = db.query(RisqIngestionReport).order_by(RisqIngestionReport.started_at.desc()).limit(50).all()
        latest_by_file = {}
        for row in rows:
            if row.source_file not in latest_by_file:
                latest_by_file[row.source_file] = {
                    "id": row.id,
                    "source_file": row.source_file,
                    "document_type": row.document_type,
                    "total_rows": row.total_rows,
                    "embedded": row.embedded,
                    "upserted": row.upserted,
                    "matched": row.matched,
                    "unmatched": row.unmatched,
                    "invalid": row.invalid,
                    "status": row.status,
                    "error_summary": row.error_summary,
                    "started_at": row.started_at.isoformat() if row.started_at else None,
                }
        return {
            "reports": list(latest_by_file.values()),
            "collection_counts": {
                "questions": state.risq_collection_questions.count() if state.risq_collection_questions else 0,
                "training": state.risq_collection_training.count() if state.risq_collection_training else 0,
                "documents": state.risq_collection_documents.count() if state.risq_collection_documents else 0,
            },
        }
    except Exception as e:
        logger.error(f"RISQ training status failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get RISQ training status")


@router.get("/api/v1/risq/documents")
async def risq_list_documents(db: Session = Depends(get_risq_db)):
    """List distinct ingested RISQ source files with their latest report summary."""
    try:
        rows = db.query(RisqIngestionReport).order_by(RisqIngestionReport.started_at.desc()).all()
        latest_by_file = {}
        for row in rows:
            if row.source_file not in latest_by_file:
                latest_by_file[row.source_file] = {
                    "source_file": row.source_file,
                    "document_type": row.document_type,
                    "upserted": row.upserted,
                    "status": row.status,
                    "started_at": row.started_at.isoformat() if row.started_at else None,
                }
        return {"documents": list(latest_by_file.values())}
    except Exception as e:
        logger.error(f"RISQ document listing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list RISQ documents")


@router.delete("/api/v1/risq/documents/{source_file}")
async def risq_delete_document(source_file: str, db: Session = Depends(get_risq_db)):
    """Module-scoped safe deletion: removes rows only from risq_* Chroma
    collections and RisqIngestionReport rows for the given source file.
    Cannot touch VIQ collections — different collection objects entirely,
    and no VIQ deletion endpoint exists to accidentally share a code path with."""
    resolve_module("risq")
    try:
        deleted_counts = {}
        for name, coll in (
            ("questions", state.risq_collection_questions),
            ("training", state.risq_collection_training),
            ("documents", state.risq_collection_documents),
        ):
            if coll is None:
                continue
            try:
                existing = coll.get(where={"source_file": source_file})
                if existing["ids"]:
                    coll.delete(where={"source_file": source_file})
                deleted_counts[name] = len(existing["ids"])
            except Exception as e:
                logger.error(f"RISQ: delete failed for collection {name}: {str(e)}")
                deleted_counts[name] = 0

        report_rows = db.query(RisqIngestionReport).filter(RisqIngestionReport.source_file == source_file).all()
        for row in report_rows:
            db.delete(row)
        db.commit()

        return {
            "source_file": source_file,
            "deleted_from_collections": deleted_counts,
            "deleted_report_rows": len(report_rows),
        }
    except Exception as e:
        logger.error(f"RISQ document deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete RISQ document")
