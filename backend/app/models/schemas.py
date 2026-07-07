from pydantic import BaseModel
from typing import List, Optional


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
