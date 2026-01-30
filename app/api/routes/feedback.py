from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging
from app.models.feedback import UserFeedback, FeedbackResponse, FeedbackType
from app.core.feedback_manager import FeedbackManager
from openai import OpenAI
from app.config import settings
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize feedback manager
feedback_manager = FeedbackManager()
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: UserFeedback):
    """
    Submit user feedback on VIQ match
    
    - **Thumbs Up**: Confirms the VIQ match is correct → System learns
    - **Thumbs Down**: Reports incorrect match → Provide correct VIQ to help system learn
    """
    try:
        # Get embedding for the finding
        if not openai_client:
            raise HTTPException(status_code=500, detail="OpenAI client not available")
        
        response = await asyncio.to_thread(
            openai_client.embeddings.create,
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=feedback.finding
        )
        embedding = response.data[0].embedding
        
        # Handle thumbs up
        if feedback.feedback_type == FeedbackType.THUMBS_UP:
            result = await feedback_manager.save_thumbs_up(
                finding=feedback.finding,
                viq_number=feedback.suggested_viq,
                confidence=feedback.confidence_score or 1.0,
                embedding=embedding
            )
            return FeedbackResponse(**result)
        
        # Handle thumbs down
        elif feedback.feedback_type == FeedbackType.THUMBS_DOWN:
            result = await feedback_manager.save_thumbs_down(
                finding=feedback.finding,
                wrong_viq=feedback.suggested_viq,
                correct_viq=feedback.correct_viq,
                embedding=embedding,
                user_comment=feedback.user_comment
            )
            return FeedbackResponse(**result)
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feedback/stats")
async def get_feedback_stats() -> Dict[str, Any]:
    """Get feedback statistics"""
    try:
        stats = feedback_manager.get_stats()
        return {
            "feedback_stats": stats,
            "message": "Feedback statistics retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
