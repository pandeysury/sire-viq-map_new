"""
RISQ business logic — extracted from app/main.py verbatim (structural move
only, no behavior change). Fully isolated from viq_engine.py: never reads
or writes any VIQ collection or VIQ SQL table.
"""

import logging
import re
from typing import List, Optional

from fastapi import HTTPException

from app import state
from app.config import settings

logger = logging.getLogger(__name__)

ALLOWED_MODULES = {"viq", "risq"}


def resolve_module(module: Optional[str]) -> str:
    """Validate a client-supplied module string against the allowlist.
    Never trust a frontend-supplied collection name — only this resolved
    module value is used to pick backend-controlled collection constants."""
    m = (module or "viq").strip().lower()
    if m not in ALLOWED_MODULES:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_module",
                "message": f"Unsupported module '{module}'. Allowed values: {sorted(ALLOWED_MODULES)}",
                "allowed": sorted(ALLOWED_MODULES),
                "received": module,
            },
        )
    return m


# Two-part RISQ code pattern (e.g. "2.1", "12.8", "7.1 A"), distinct from
# VIQ's three-part "X.Y.Z" numbering — must not be confused with it.
RISQ_CODE_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}(?:\.\d{1,3})?[A-Za-z]?\b")


def _risq_normalize_code(code: str) -> str:
    # Strip ALL whitespace (not just collapse) — source files are
    # inconsistent about spacing before letter suffixes (e.g. "7.2C" vs
    # "7.2 C" are the same code). Mirrors scripts/train_risq_from_excel.py's
    # _normalize_risq_code.
    return re.sub(r"\s+", "", str(code).strip()).lower()


def _risq_detect_code(query: str) -> Optional[str]:
    """Best-effort detection of a RISQ code embedded in free text."""
    if not query:
        return None
    match = RISQ_CODE_PATTERN.search(query)
    return match.group().strip() if match else None


async def load_risq_collections():
    """Load/attach the three RISQ Chroma collections at startup. Failures
    here are logged and swallowed — RISQ being unavailable must never
    prevent VIQ startup or break VIQ requests."""
    try:
        if state.risq_chroma_client is None:
            logger.warning("RISQ: risq_chroma_client not initialized, skipping RISQ collection load")
            return

        def _get_or_create(name):
            names = [c.name for c in state.risq_chroma_client.list_collections()]
            if name in names:
                return state.risq_chroma_client.get_collection(name)
            return state.risq_chroma_client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

        state.risq_collection_questions = _get_or_create(settings.RISQ_CHROMA_QUESTIONS_COLLECTION)
        state.risq_collection_training = _get_or_create(settings.RISQ_CHROMA_TRAINING_COLLECTION)
        state.risq_collection_documents = _get_or_create(settings.RISQ_CHROMA_DOCUMENTS_COLLECTION)
        logger.info(
            "RISQ collections ready: questions=%d training=%d documents=%d",
            state.risq_collection_questions.count(),
            state.risq_collection_training.count(),
            state.risq_collection_documents.count(),
        )
    except Exception as e:
        logger.error(f"RISQ: failed to load collections (non-fatal, VIQ unaffected): {str(e)}")


def _risq_question_lookup(risq_code: str) -> Optional[dict]:
    """Fetch a risq_questions record by exact code, for enriching a
    training-data match with its official question text."""
    result = state.risq_collection_questions.get(where={"risq_code": risq_code})
    if result["ids"]:
        return result["metadatas"][0]
    normalized = _risq_normalize_code(risq_code)
    all_docs = state.risq_collection_questions.get(include=["metadatas"])
    for meta in all_docs["metadatas"]:
        if _risq_normalize_code(meta.get("risq_code", "")) == normalized:
            return meta
    return None


async def risq_search_matches(query: str, code: Optional[str], top_k: int) -> List[dict]:
    """RISQ retrieval — mirrors VIQ's plain /api/v1/search logic exactly:
    a ranked list of {risq_code, question, similarity_score} matches, no
    LLM composition, no guidance/PDF/historical sections. Priority order,
    same as VIQ's search_viq():
      1. Exact reference code (given or detected in the query) — score 1.0
      2. Historical findings (risq_training) semantic match — catches the
         case where the user pastes a finding that's already on file,
         even though its wording differs from the official question text
      3. Fallback: semantic match directly against risq_questions
    Never reads or falls back to VIQ collections."""
    if state.risq_collection_questions is None:
        raise HTTPException(status_code=500, detail="RISQ system not initialized")
    if state.client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    matches: List[dict] = []
    seen_codes = set()
    detected_code = code or _risq_detect_code(query)
    query_embedding = None

    if detected_code:
        meta = _risq_question_lookup(detected_code.strip())
        if meta:
            ref = meta.get("risq_code")
            matches.append({
                "risq_code": ref,
                "question": meta.get("question"),
                "similarity_score": 1.0,
            })
            seen_codes.add(ref)

    if len(matches) < top_k and state.risq_collection_training is not None and state.risq_collection_training.count() > 0:
        if query_embedding is None:
            embed_response = state.client.embeddings.create(model="text-embedding-3-small", input=query)
            query_embedding = embed_response.data[0].embedding

        n_results = min(top_k * 3, state.risq_collection_training.count())
        t_results = state.risq_collection_training.query(query_embeddings=[query_embedding], n_results=n_results)

        if t_results["ids"] and t_results["ids"][0]:
            for i, meta in enumerate(t_results["metadatas"][0]):
                # Only follow a training match to a real question-master code —
                # never surface an unmatched historical reference as an answer.
                ref = meta.get("matched_risq_code") or None
                if not ref or ref in seen_codes:
                    continue
                distance = t_results["distances"][0][i]
                similarity_score = 1 - distance
                if similarity_score < 0.6:
                    continue
                q_meta = _risq_question_lookup(ref)
                if not q_meta:
                    continue
                matches.append({
                    "risq_code": ref,
                    "question": q_meta.get("question"),
                    "similarity_score": round(similarity_score, 4),
                })
                seen_codes.add(ref)
                if len(matches) >= top_k:
                    break

    if len(matches) < top_k and state.risq_collection_questions.count() > 0:
        if query_embedding is None:
            embed_response = state.client.embeddings.create(model="text-embedding-3-small", input=query)
            query_embedding = embed_response.data[0].embedding

        n_results = min(max(top_k * 2, top_k), state.risq_collection_questions.count())
        q_results = state.risq_collection_questions.query(query_embeddings=[query_embedding], n_results=n_results)

        if q_results["ids"] and q_results["ids"][0]:
            for i, meta in enumerate(q_results["metadatas"][0]):
                ref = meta.get("risq_code")
                if ref in seen_codes:
                    continue
                distance = q_results["distances"][0][i]
                similarity_score = 1 - distance
                if similarity_score < settings.RISQ_SIMILARITY_THRESHOLD:
                    continue
                matches.append({
                    "risq_code": ref,
                    "question": meta.get("question"),
                    "similarity_score": round(similarity_score, 4),
                })
                seen_codes.add(ref)
                if len(matches) >= top_k:
                    break

    return matches
