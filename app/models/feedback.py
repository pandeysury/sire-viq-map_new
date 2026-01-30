from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class FeedbackType(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"

class UserFeedback(BaseModel):
    finding: str = Field(..., description="Original finding/query")
    suggested_viq: str = Field(..., description="VIQ number that was suggested")
    feedback_type: FeedbackType = Field(..., description="thumbs_up or thumbs_down")
    correct_viq: Optional[str] = Field(None, description="Correct VIQ if thumbs_down")
    vessel_type: str = Field(default="All")
    confidence_score: Optional[float] = None
    user_comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    success: bool
    message: str
    learned: bool = False
    feedback_id: Optional[str] = None
