"""
Pydantic request/response models, split out of app/main.py.

VIQ models and RISQ models are grouped separately and never share a base
class or fields — this mirrors the data-isolation rule that VIQ and RISQ
must never be mixed.
"""

from typing import List, Optional
from pydantic import BaseModel


# ===== VIQ models (unchanged from the original app/main.py) =====

class QueryRequest(BaseModel):
    query: str
    vessel_type: Optional[str] = "All"
    top_k: Optional[int] = 5
    search_method: Optional[str] = "hybrid"
    confidence_threshold: Optional[float] = 0.3


class VIQMatch(BaseModel):
    viq_number: str
    question: str
    vessel_types: str
    similarity_score: float
    context: Optional[str] = None


class QueryResponse(BaseModel):
    matches: List[VIQMatch]
    query: str
    vessel_type: str


# ===== RISQ models (unchanged from the original app/main.py) =====

class RisqAskRequest(BaseModel):
    query: str
    code: Optional[str] = None
    top_k: Optional[int] = None


class RisqMatch(BaseModel):
    risq_code: str
    question: str
    similarity_score: float


class RisqAskResponse(BaseModel):
    matches: List[RisqMatch]
    query: str
