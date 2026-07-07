import json
import logging
import os
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core import rag_engine
from app.database import SearchHistory, get_db
from app.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", response_model=QueryResponse)
async def search_viq(request: QueryRequest, db: Session = Depends(get_db)):
    """Search VIQ questions based on audit finding"""
    start_time = datetime.now()

    try:
        matches = rag_engine.perform_search(
            request.query, request.vessel_type, request.top_k, request.confidence_threshold
        )

        # Log search to database
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        if matches:
            search_log = SearchHistory(
                query=request.query,
                result_viq=matches[0].viq_number,
                confidence=matches[0].similarity_score,
                time_ms=processing_time,
                vessel_type=request.vessel_type or "All",
                search_method=request.search_method or "hybrid"
            )
            db.add(search_log)
            db.commit()

        return QueryResponse(
            matches=matches,
            query=request.query,
            vessel_type=request.vessel_type or "All"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-finding", response_model=QueryResponse)
async def analyze_finding(request: QueryRequest):
    """Analyze finding with AI enhancement"""
    try:
        matches = rag_engine.perform_search(
            request.query, request.vessel_type, request.top_k, request.confidence_threshold
        )
        search_response = QueryResponse(
            matches=matches,
            query=request.query,
            vessel_type=request.vessel_type or "All"
        )

        if not rag_engine.client or not search_response.matches:
            return search_response

        # Create focused analysis prompt for ONLY the best match
        best_match = search_response.matches[0]

        prompt = f"""
Analyze this ship audit finding and explain why the best matched VIQ question is relevant:

Finding: "{request.query}"
Vessel Type: {request.vessel_type or 'All'}

Best Match:
VIQ {best_match.viq_number}: {best_match.question}
Vessel Types: {best_match.vessel_types}
Confidence Score: {best_match.similarity_score}

In a SHORT single paragraph of no more than 4 sentences, cover: what this finding
indicates about the ship's operations, why this specific VIQ question is the best
match, and the key compliance or safety concern it highlights. Be concise - combine
these points into flowing sentences, not separate sections.

Write as plain prose only. Do not use markdown: no headings (#, ##, ###), no bullet
points or numbered lists, no bold/italic (** or *), no blockquotes (>), no
horizontal rules (---).
"""

        try:
            response = rag_engine.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
                temperature=0.2
            )

            analysis = response.choices[0].message.content

            if search_response.matches:
                search_response.matches[0].context = f"AI Analysis: {analysis}\\n\\n{search_response.matches[0].context}"

        except Exception as e:
            logger.warning(f"OpenAI analysis failed: {str(e)}")

        return search_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_finding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-finding/stream")
async def analyze_finding_stream(request: QueryRequest):
    """Same analysis as /analyze-finding, but streams the AI text token-by-token
    as newline-delimited JSON (NDJSON) events: {"type": "match"|"token"|"error"|"done", ...}
    """
    matches = rag_engine.perform_search(
        request.query, request.vessel_type, request.top_k, request.confidence_threshold
    )

    def generate():
        yield json.dumps({
            "type": "match",
            "matches": [m.model_dump() for m in matches]
        }) + "\n"

        if not rag_engine.client or not matches:
            yield json.dumps({"type": "done"}) + "\n"
            return

        best_match = matches[0]

        prompt = f"""
Analyze this ship audit finding and explain why the best matched VIQ question is relevant:

Finding: "{request.query}"
Vessel Type: {request.vessel_type or 'All'}

Best Match:
VIQ {best_match.viq_number}: {best_match.question}
Vessel Types: {best_match.vessel_types}
Confidence Score: {best_match.similarity_score}

In a SHORT single paragraph of no more than 4 sentences, cover: what this finding
indicates about the ship's operations, why this specific VIQ question is the best
match, and the key compliance or safety concern it highlights. Be concise - combine
these points into flowing sentences, not separate sections.

Write as plain prose only. Do not use markdown: no headings (#, ##, ###), no bullet
points or numbered lists, no bold/italic (** or *), no blockquotes (>), no
horizontal rules (---).
"""

        try:
            stream = rag_engine.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
                temperature=0.2,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield json.dumps({"type": "token", "text": delta}) + "\n"
        except Exception as e:
            logger.warning(f"OpenAI streaming analysis failed: {str(e)}")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/vessel-types")
async def get_vessel_types():
    """Get available vessel types"""
    return {"vessel_types": ["All", "Oil", "Chemical", "LPG", "LNG"]}


@router.get("/search-methods")
async def get_search_methods():
    """Get available search methods"""
    return {
        "search_methods": ["semantic", "keyword", "hybrid"],
        "default": "hybrid"
    }
