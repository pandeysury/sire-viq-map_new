from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum

class VesselType(str, Enum):
    ALL = "All"
    OIL = "Oil"
    CHEMICAL = "Chemical"
    LPG = "LPG"
    LNG = "LNG"

class SearchMethod(str, Enum):
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    KEYWORD = "keyword"

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Search query")
    vessel_type: VesselType = Field(default=VesselType.ALL, description="Vessel type filter")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    search_method: SearchMethod = Field(default=SearchMethod.HYBRID, description="Search method")
    include_context: bool = Field(default=True, description="Include context in results")
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence score")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class VIQMatch(BaseModel):
    viq_number: str = Field(..., description="VIQ question number")
    question: str = Field(..., description="VIQ question text")
    vessel_type: str = Field(..., description="Applicable vessel type")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    context: Optional[str] = Field(None, description="Additional context or guidance")
    source_file: Optional[str] = Field(None, description="Source document")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
class QueryResponse(BaseModel):
    matches: List[VIQMatch] = Field(..., description="List of matching VIQ questions")
    query: str = Field(..., description="Original query")
    vessel_type: str = Field(..., description="Vessel type filter used")
    total_results: int = Field(..., description="Total number of results")
    search_method: str = Field(..., description="Search method used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    enhanced_query: Optional[str] = Field(None, description="LLM-enhanced query")

class EnhancedQueryResponse(QueryResponse):
    ai_analysis: Optional[str] = Field(None, description="AI analysis of the finding")
    recommendations: Optional[List[str]] = Field(None, description="AI recommendations")
    risk_assessment: Optional[str] = Field(None, description="Risk assessment")

class SystemStats(BaseModel):
    total_questions: int = Field(..., description="Total VIQ questions in database")
    vessel_types: List[str] = Field(..., description="Available vessel types")
    collections: List[str] = Field(..., description="Available collections")
    status: str = Field(..., description="System status")
    search_methods: List[str] = Field(..., description="Available search methods")
    openai_available: bool = Field(..., description="OpenAI API availability")
    vector_store_status: str = Field(..., description="Vector store status")
    last_updated: Optional[str] = Field(None, description="Last data update timestamp")

class ProcessingStatus(BaseModel):
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    progress: Optional[float] = Field(None, ge=0.0, le=100.0, description="Progress percentage")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")

class HealthCheck(BaseModel):
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Component health status")

class VIQDocument(BaseModel):
    viq_number: str = Field(..., description="VIQ question number")
    question: str = Field(..., description="VIQ question text")
    vessel_type: str = Field(..., description="Applicable vessel type")
    guidance: Optional[str] = Field(None, description="Guidance text")
    source_file: str = Field(..., description="Source document")
    chapter: Optional[str] = Field(None, description="Document chapter")
    section: Optional[str] = Field(None, description="Document section")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")