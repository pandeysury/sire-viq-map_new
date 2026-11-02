#!/usr/bin/env python3
"""
RISQ Training System — ingest the RISQ question-master and historical-findings
Excel workbooks into their own, VIQ-isolated ChromaDB collections.

Mirrors the structure of scripts/train_from_excel.py (the VIQ training script)
but is fully independent:
  - Writes only to risq_questions / risq_training (never viq_questions / viq_training)
  - Is idempotent (deterministic IDs + upsert, not add-with-timestamp-IDs)
  - Never deletes/recreates a collection
  - Never fails the whole run because of one bad row

This module is designed to be BOTH:
  1. Run standalone:  python scripts/train_risq_from_excel.py
  2. Imported by the backend (app/main.py) so the `/api/v1/risq/train` endpoint
     can trigger the same logic in-process. All side effects are behind the
     RISQTrainingSystem class methods; module import alone does nothing.
"""

import hashlib
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

import chromadb
import pandas as pd
from dotenv import load_dotenv

# Load environment variables (root .env if present; harmless no-op otherwise)
load_dotenv()

# Add parent directory to path so `app.config` is importable when run standalone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.config import settings
except ImportError:
    class Settings:
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        RISQ_DATA_DIR = Path(__file__).parent.parent / "data" / "risqData"
        RISQ_QUESTIONS_FILE = "Risq 3.2 R2 - 25.02.26 updated 2.xlsx"
        RISQ_QUESTIONS_SHEET = "Risq 3.2 R1"
        RISQ_TRAINING_FILE = "RISQ training data.xlsx"
        RISQ_TRAINING_SHEET = "ag-grid"
        RISQ_CHROMA_QUESTIONS_COLLECTION = "risq_questions"
        RISQ_CHROMA_TRAINING_COLLECTION = "risq_training"
    settings = Settings()

QUESTION_MASTER_REQUIRED_COLUMNS = {"code", "viqQuestionsFullText", "tmsaDescription"}
# Date is intentionally NOT required/used — only the reference and the
# finding text are ingested; any "Date" column in the source file is ignored.
TRAINING_REQUIRED_COLUMNS = {"RISQ Ref", "Finding"}

# Marker inserted by the source editorial tooling; must be stripped from all
# ingested/displayed text without disturbing surrounding whitespace/paragraphs.
FORMATTING_MARKER = "#z#"


def _clean_risq_text(value) -> str:
    """Remove the #z# formatting marker, normalize NaN to '', and preserve
    paragraphs/line breaks and meaningful whitespace otherwise."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    if text.strip().lower() == "nan":
        return ""
    text = text.replace(FORMATTING_MARKER, "")
    # Trim only leading/trailing whitespace of the whole block; keep internal
    # newlines/paragraphs exactly as authored.
    return text.strip()


def _normalize_risq_code(code) -> str:
    """Normalize a RISQ code for COMPARISON only (never for storage/display).

    - Strips ALL whitespace (not just collapsing repeats) — the source data
      is inconsistent about whether a letter suffix has a space before it
      (e.g. question master "7.2C" vs training data "7.2 C" — same code,
      just typed differently). Verified safe: no two distinct codes in the
      question master collide once whitespace is removed.
    - Case-folds (for letter suffixes like 'A'/'a')
    Never strips dots, digits, or letters — "12.8a" and "12.8" are distinct.
    """
    s = str(code).strip()
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _stable_code_string(code) -> str:
    """Return the code as a clean display/storage string (no normalization
    beyond whitespace trimming), always a string, never a float."""
    return re.sub(r"\s+", " ", str(code).strip())


def _file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _deterministic_id(prefix: str, *parts: str) -> str:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


class RISQTrainingSystem:
    """Ingest RISQ question-master and historical-findings workbooks into
    isolated, idempotent ChromaDB collections. Never touches VIQ data."""

    def __init__(self, data_dir: Optional[Path] = None, chroma_client=None):
        """
        chroma_client: an already-open chromadb client to reuse instead of
        opening a new one. THIS MATTERS: when this class is invoked from
        inside the long-running FastAPI process (the /api/v1/risq/train
        endpoint), opening a second, independent PersistentClient pointed
        at the same on-disk path can leave the server's own client with a
        stale view of the data it just wrote — searches keep returning
        pre-training results until the process is restarted. Passing the
        server's existing `chroma_client` here makes training and serving
        share one connection, so newly ingested data is visible immediately.
        When run as a standalone script (a fresh process each time), no
        client is injected and a new one is opened as before — there's no
        staleness risk there since nothing else holds an older handle.
        """
        self.data_dir = Path(data_dir) if data_dir else Path(settings.RISQ_DATA_DIR)
        self.questions_collection_name = settings.RISQ_CHROMA_QUESTIONS_COLLECTION
        self.training_collection_name = settings.RISQ_CHROMA_TRAINING_COLLECTION
        self.reports = []  # one dict per processed file, for the ingestion report

        script_dir = Path(__file__).parent.parent
        # RISQ has its own, physically separate vector store — never the
        # shared/VIQ one — so a standalone run of this script (no injected
        # client) can never touch VIQ's data.
        self.persist_dir = script_dir / "data" / "risq_vectordb"

        self._injected_chroma_client = chroma_client
        self._openai_client = None
        self._code_index = {}  # normalized code -> stable display code (from question master)

    # ---------- infra helpers ----------

    def _get_chroma_client(self):
        if self._injected_chroma_client is not None:
            return self._injected_chroma_client
        return chromadb.PersistentClient(path=str(self.persist_dir))

    def _get_or_create_collection(self, client, name):
        collections = [c.name for c in client.list_collections()]
        if name in collections:
            return client.get_collection(name)
        return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai
            api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not found in environment")
            self._openai_client = openai.OpenAI(api_key=api_key)
        return self._openai_client

    def _embed_texts(self, texts, batch_size=100):
        client = self._get_openai_client()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(model="text-embedding-3-small", input=batch)
            all_embeddings.extend([e.embedding for e in response.data])
        return all_embeddings

    @staticmethod
    def _validate_columns(df, required_columns, file_label):
        present = set(str(c).strip() for c in df.columns)
        missing = required_columns - present
        if missing:
            raise ValueError(
                f"{file_label}: missing required column(s) {sorted(missing)}. "
                f"Found columns: {sorted(present)}"
            )

    # ---------- question master ----------

    def ingest_question_master(self, file_path: Optional[Path] = None, sheet_name: Optional[str] = None):
        file_path = Path(file_path) if file_path else self.data_dir / settings.RISQ_QUESTIONS_FILE
        sheet_name = sheet_name or settings.RISQ_QUESTIONS_SHEET

        report = {
            "source_file": file_path.name,
            "document_type": "question_master",
            "file_checksum": None,
            "total_rows": 0,
            "embedded": 0,
            "upserted": 0,
            "matched": 0,
            "unmatched": 0,
            "invalid": 0,
            "status": "success",
            "errors": [],
        }

        if not file_path.exists():
            report["status"] = "failed"
            report["errors"].append(f"File not found: {file_path}")
            self.reports.append(report)
            return report

        report["file_checksum"] = _file_checksum(file_path)

        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, dtype={"code": str})
        except Exception as e:
            report["status"] = "failed"
            report["errors"].append(f"Could not read sheet '{sheet_name}': {e}")
            self.reports.append(report)
            return report

        try:
            self._validate_columns(df, QUESTION_MASTER_REQUIRED_COLUMNS, f"{file_path.name}[{sheet_name}]")
        except ValueError as e:
            report["status"] = "failed"
            report["errors"].append(str(e))
            self.reports.append(report)
            return report

        report["total_rows"] = len(df)

        records = []
        for idx, row in df.iterrows():
            try:
                raw_code = row.get("code")
                if raw_code is None or (isinstance(raw_code, float) and pd.isna(raw_code)) or str(raw_code).strip() == "":
                    report["invalid"] += 1
                    report["errors"].append(f"row {idx}: missing code")
                    continue

                code = _stable_code_string(raw_code)
                question = _clean_risq_text(row.get("viqQuestionsFullText"))
                guidance = _clean_risq_text(row.get("tmsaDescription"))

                if not question:
                    report["invalid"] += 1
                    report["errors"].append(f"row {idx} (code {code}): missing question text")
                    continue

                records.append({
                    "code": code,
                    "question": question,
                    "guidance": guidance,  # may legitimately be "" — never invented
                    "row_index": int(idx),
                })
            except Exception as e:
                report["invalid"] += 1
                report["errors"].append(f"row {idx}: {e}")

        if not records:
            report["status"] = "failed" if report["invalid"] else "success"
            self.reports.append(report)
            return report

        # Build the code index for later cross-referencing by ingest_training_data()
        self._code_index = {_normalize_risq_code(r["code"]): r["code"] for r in records}

        try:
            embeddings = self._embed_texts([r["question"] for r in records])
            report["embedded"] = len(embeddings)
        except Exception as e:
            report["status"] = "failed"
            report["errors"].append(f"Embedding failed: {e}")
            self.reports.append(report)
            return report

        client = self._get_chroma_client()
        collection = self._get_or_create_collection(client, self.questions_collection_name)

        ids = [_deterministic_id("risq_q", file_path.name, sheet_name, r["code"]) for r in records]
        documents = [r["question"] for r in records]
        metadatas = [
            {
                "module": "risq",
                "document_type": "question_master",
                "risq_code": r["code"],
                "question": r["question"],
                "description": r["guidance"],
                "source_file": file_path.name,
                "source_sheet": sheet_name,
                "row_index": r["row_index"],
            }
            for r in records
        ]

        batch_size = 1000
        for i in range(0, len(records), batch_size):
            collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )
            report["upserted"] += len(ids[i:i + batch_size])

        report["error_summary"] = "; ".join(report["errors"][:20]) if report["errors"] else None
        if report["invalid"] > 0 and report["upserted"] > 0:
            report["status"] = "partial"
        self.reports.append(report)
        return report

    # ---------- historical findings ----------

    def ingest_training_data(self, file_path: Optional[Path] = None, sheet_name: Optional[str] = None):
        file_path = Path(file_path) if file_path else self.data_dir / settings.RISQ_TRAINING_FILE
        sheet_name = sheet_name or settings.RISQ_TRAINING_SHEET

        report = {
            "source_file": file_path.name,
            "document_type": "historical_finding",
            "file_checksum": None,
            "total_rows": 0,
            "embedded": 0,
            "upserted": 0,
            "matched": 0,
            "unmatched": 0,
            "invalid": 0,
            "status": "success",
            "errors": [],
        }

        if not file_path.exists():
            report["status"] = "failed"
            report["errors"].append(f"File not found: {file_path}")
            self.reports.append(report)
            return report

        report["file_checksum"] = _file_checksum(file_path)

        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, dtype={"RISQ Ref": str})
        except Exception as e:
            report["status"] = "failed"
            report["errors"].append(f"Could not read sheet '{sheet_name}': {e}")
            self.reports.append(report)
            return report

        try:
            self._validate_columns(df, TRAINING_REQUIRED_COLUMNS, f"{file_path.name}[{sheet_name}]")
        except ValueError as e:
            report["status"] = "failed"
            report["errors"].append(str(e))
            self.reports.append(report)
            return report

        report["total_rows"] = len(df)

        # If the question master hasn't been ingested in this run, load the
        # existing code index straight from the risq_questions collection so
        # matching still works when only the training file is (re-)run.
        if not self._code_index:
            self._load_code_index_from_collection()

        records = []
        for idx, row in df.iterrows():
            try:
                raw_ref = row.get("RISQ Ref")
                finding = _clean_risq_text(row.get("Finding"))
                # Note: the "Date" column, if present in the source file, is
                # intentionally not read or stored — date is out of scope.

                if raw_ref is None or (isinstance(raw_ref, float) and pd.isna(raw_ref)) or str(raw_ref).strip() == "":
                    report["invalid"] += 1
                    report["errors"].append(f"row {idx}: missing RISQ Ref")
                    continue
                if not finding:
                    report["invalid"] += 1
                    report["errors"].append(f"row {idx}: missing Finding text")
                    continue

                ref = _stable_code_string(raw_ref)
                normalized = _normalize_risq_code(ref)
                matched_code = self._code_index.get(normalized)
                mapping_status = "matched" if matched_code else "unmatched"
                if not matched_code:
                    report["errors"].append(f"row {idx}: unmatched RISQ Ref '{ref}'")

                records.append({
                    "risq_code": ref,
                    "matched_risq_code": matched_code,
                    "mapping_status": mapping_status,
                    "finding": finding,
                    "row_index": int(idx),
                })
                if mapping_status == "matched":
                    report["matched"] += 1
                else:
                    report["unmatched"] += 1
            except Exception as e:
                report["invalid"] += 1
                report["errors"].append(f"row {idx}: {e}")

        if not records:
            report["status"] = "failed" if report["invalid"] else "success"
            self.reports.append(report)
            return report

        try:
            embeddings = self._embed_texts([r["finding"] for r in records])
            report["embedded"] = len(embeddings)
        except Exception as e:
            report["status"] = "failed"
            report["errors"].append(f"Embedding failed: {e}")
            self.reports.append(report)
            return report

        client = self._get_chroma_client()
        collection = self._get_or_create_collection(client, self.training_collection_name)

        ids = [
            _deterministic_id("risq_t", file_path.name, sheet_name, r["row_index"], r["finding"])
            for r in records
        ]
        documents = [r["finding"] for r in records]
        metadatas = [
            {
                "module": "risq",
                "document_type": "historical_finding",
                "risq_code": r["risq_code"],
                "matched_risq_code": r["matched_risq_code"] or "",
                "mapping_status": r["mapping_status"],
                "finding": r["finding"],
                "source_file": file_path.name,
                "source_sheet": sheet_name,
                "row_index": r["row_index"],
            }
            for r in records
        ]

        batch_size = 1000
        for i in range(0, len(records), batch_size):
            collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )
            report["upserted"] += len(ids[i:i + batch_size])

        report["error_summary"] = "; ".join(report["errors"][:20]) if report["errors"] else None
        if report["invalid"] > 0 and report["upserted"] > 0:
            report["status"] = "partial"
        self.reports.append(report)
        return report

    def _load_code_index_from_collection(self):
        try:
            client = self._get_chroma_client()
            collections = [c.name for c in client.list_collections()]
            if self.questions_collection_name not in collections:
                return
            collection = client.get_collection(self.questions_collection_name)
            if collection.count() == 0:
                return
            data = collection.get(include=["metadatas"])
            for meta in data["metadatas"]:
                code = meta.get("risq_code")
                if code:
                    self._code_index[_normalize_risq_code(code)] = code
        except Exception:
            # Non-fatal: matching will simply mark rows as unmatched if the
            # question master isn't available yet.
            pass

    # ---------- orchestration ----------

    def _classify_excel_sheets(self, file_path: Path) -> dict:
        """Identify what each sheet in an Excel file is by its ACTUAL
        columns — never by filename. Returns {sheet_name: 'question_master'
        | 'historical_finding'} for every sheet whose header row contains
        all the required columns for one of the two known schemas:
          question_master:   code, viqQuestionsFullText, tmsaDescription
          historical_finding: Date, RISQ Ref, Finding
        Sheets matching neither schema are omitted (and reported as
        unrecognized by the caller)."""
        try:
            xl = pd.ExcelFile(file_path)
        except Exception:
            return {}

        classified = {}
        for sheet in xl.sheet_names:
            try:
                header_only = pd.read_excel(file_path, sheet_name=sheet, nrows=0)
            except Exception:
                continue
            columns = set(str(c).strip() for c in header_only.columns)
            if QUESTION_MASTER_REQUIRED_COLUMNS.issubset(columns):
                classified[sheet] = "question_master"
            elif TRAINING_REQUIRED_COLUMNS.issubset(columns):
                classified[sheet] = "historical_finding"
        return classified

    def run_full_ingestion(self):
        """Auto-discover every Excel file in the RISQ data directory and
        classify each sheet by its columns (see _classify_excel_sheets) —
        NOT by filename. This means a re-uploaded file is correctly picked
        up regardless of what it's named, as long as its columns match one
        of the two known RISQ schemas. Question-master sheets are ingested
        first (to build the code index for matching), then historical-
        finding sheets. A file/sheet that matches neither schema is
        reported as unrecognized rather than silently skipped or guessed
        at. Returns the full list of per-file(/sheet) reports."""
        self.reports = []

        excel_files = sorted(self.data_dir.glob("*.xlsx")) + sorted(self.data_dir.glob("*.xls"))
        if not excel_files:
            self.reports.append({
                "source_file": None,
                "document_type": "unknown",
                "file_checksum": None,
                "total_rows": 0, "embedded": 0, "upserted": 0,
                "matched": 0, "unmatched": 0, "invalid": 0,
                "status": "failed",
                "errors": [f"No .xlsx/.xls files found in {self.data_dir}"],
            })
            return self.reports

        question_master_jobs = []
        training_jobs = []

        for file_path in excel_files:
            sheets = self._classify_excel_sheets(file_path)
            if not sheets:
                self.reports.append({
                    "source_file": file_path.name,
                    "document_type": "unknown",
                    "file_checksum": _file_checksum(file_path) if file_path.exists() else None,
                    "total_rows": 0, "embedded": 0, "upserted": 0,
                    "matched": 0, "unmatched": 0, "invalid": 0,
                    "status": "failed",
                    "errors": [
                        f"No sheet in '{file_path.name}' matched the required columns for either "
                        f"a RISQ question master ({sorted(QUESTION_MASTER_REQUIRED_COLUMNS)}) or "
                        f"RISQ historical findings ({sorted(TRAINING_REQUIRED_COLUMNS)})."
                    ],
                })
                continue
            for sheet_name, doc_type in sheets.items():
                if doc_type == "question_master":
                    question_master_jobs.append((file_path, sheet_name))
                else:
                    training_jobs.append((file_path, sheet_name))

        # Question masters first — builds the code index used to match
        # historical findings against official reference codes.
        for file_path, sheet_name in question_master_jobs:
            self.ingest_question_master(file_path=file_path, sheet_name=sheet_name)
        for file_path, sheet_name in training_jobs:
            self.ingest_training_data(file_path=file_path, sheet_name=sheet_name)

        return self.reports

    def print_summary(self):
        print("\n" + "=" * 60)
        print("RISQ INGESTION REPORT")
        print("=" * 60)
        for r in self.reports:
            print(f"\nFile: {r['source_file']} ({r['document_type']})")
            print(f"  Status: {r['status']}")
            print(f"  Total rows: {r['total_rows']}")
            print(f"  Embedded: {r['embedded']}  Upserted: {r['upserted']}")
            if r["document_type"] == "historical_finding":
                print(f"  Matched: {r['matched']}  Unmatched: {r['unmatched']}")
            print(f"  Invalid/skipped: {r['invalid']}")
            if r.get("errors"):
                print(f"  First errors ({min(5, len(r['errors']))} of {len(r['errors'])}):")
                for e in r["errors"][:5]:
                    print(f"    - {e}")


def main():
    print("RISQ Training System")
    print("=" * 60)
    trainer = RISQTrainingSystem()
    trainer.run_full_ingestion()
    trainer.print_summary()


if __name__ == "__main__":
    main()
