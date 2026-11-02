"""
VIQ endpoints — extracted from app/main.py verbatim (structural move only,
route paths, request/response shapes and status codes all unchanged).
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app import state
from app.database import get_db
from app.schemas import QueryRequest, QueryResponse
from app.services import viq_engine

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": "VIQ RAG System",
        "version": "1.0.0",
        "description": "AI-powered system for matching ship audit findings to VIQ questions",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs_url": "/docs"
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        components_status = {
            "vector_store": "healthy" if state.collection else "unhealthy",
            "openai_client": "healthy" if state.client else "unhealthy"
        }

        overall_status = "healthy" if state.collection and state.client else "degraded"

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": components_status
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@router.post("/api/v1/search", response_model=QueryResponse)
async def search_viq(request: QueryRequest, db: Session = Depends(get_db)):
    """Search VIQ questions based on audit finding"""
    return await viq_engine.search_viq(request, db)


@router.post("/api/v1/analyze-finding", response_model=QueryResponse)
async def analyze_finding(request: QueryRequest, db: Session = Depends(get_db)):
    """Analyze finding with AI enhancement"""
    return await viq_engine.analyze_finding(request, db)


@router.get("/api/v1/vessel-types")
async def get_vessel_types():
    """Get available vessel types"""
    return {"vessel_types": ["All", "Oil", "Chemical", "LPG", "LNG"]}


@router.get("/api/v1/search-methods")
async def get_search_methods():
    """Get available search methods"""
    return {
        "search_methods": ["semantic", "keyword", "hybrid"],
        "default": "hybrid"
    }


@router.get("/api/v1/stats")
async def get_stats():
    """Get system statistics"""
    try:
        if not state.collection:
            return {"total_questions": 0, "status": "not_initialized"}

        count = state.collection.count()
        return {
            "total_questions": count,
            "status": "ready",
            "embedding_model": "text-embedding-3-small",
            "database": "ChromaDB",
            "openai_enabled": state.client is not None
        }

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {"total_questions": 0, "status": "error", "error": str(e)}


@router.get("/api/v1/feedback/stats")
async def get_feedback_stats(db: Session = Depends(get_db)):
    """Get real feedback statistics from database"""
    from app.database import UserFeedback
    try:
        all_feedback = db.query(UserFeedback).all()

        feedback_counts = {}
        for feedback in all_feedback:
            feedback_counts[feedback.feedback_type] = feedback_counts.get(feedback.feedback_type, 0) + 1

        total_feedback = len(all_feedback)
        positive_feedback = feedback_counts.get('thumbs_up', 0)
        negative_feedback = feedback_counts.get('thumbs_down', 0)
        corrections = feedback_counts.get('correction', 0)

        # Calculate average rating (thumbs up = 5, thumbs down = 1, correction = 3)
        if total_feedback > 0:
            total_score = (positive_feedback * 5) + (negative_feedback * 1) + (corrections * 3)
            average_rating = total_score / total_feedback
        else:
            average_rating = 0.0

        return {
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "corrections_provided": corrections,
            "average_rating": round(average_rating, 2),
            "feedback_breakdown": feedback_counts,
            "user_satisfaction": round((positive_feedback / total_feedback * 100), 1) if total_feedback > 0 else 0.0
        }

    except Exception as e:
        logger.error(f"Error getting feedback stats: {str(e)}")
        return {
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "corrections_provided": 0,
            "average_rating": 0.0,
            "error": str(e)
        }


@router.post("/api/v1/feedback")
async def submit_feedback(request: Request, db: Session = Depends(get_db)):
    """Submit user feedback and learn from corrections"""
    from app.database import UserFeedback
    try:
        data = await request.json()

        # Log feedback to database
        feedback_log = UserFeedback(
            query=data.get('finding', ''),
            suggested_viq=data.get('suggested_viq', ''),
            correct_viq=data.get('correct_viq'),
            feedback_type=data.get('feedback_type', 'unknown')
        )
        db.add(feedback_log)
        db.commit()

        logger.info(f"Feedback logged to database: {data}")

        # If user provided correct VIQ OR thumbs up, add it to training data
        if data.get('correct_viq') and data.get('finding'):
            correct_viq = data['correct_viq']
            finding = data['finding']
        elif data.get('feedback_type') == 'thumbs_up' and data.get('suggested_viq') and data.get('finding'):
            correct_viq = data['suggested_viq']
            finding = data['finding']
        else:
            correct_viq = None
            finding = None

        if correct_viq and finding:
            try:
                training_collection = state.viq_chroma_client.get_collection("viq_training")

                response = state.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=finding
                )
                embedding = response.data[0].embedding

                import time
                feedback_id = f"feedback_{int(time.time())}_{hash(finding) % 10000}"

                training_collection.add(
                    ids=[feedback_id],
                    documents=[finding],
                    embeddings=[embedding],
                    metadatas=[{
                        'finding': finding,
                        'viq_number': correct_viq,
                        'source': 'user_feedback',
                        'timestamp': datetime.now().isoformat()
                    }]
                )

                logger.info(f"Added feedback to training: {finding[:50]}... -> VIQ {correct_viq}")

                feedback_type = data.get('feedback_type', 'correction')
                message = "Feedback submitted and system learned from correction" if feedback_type != 'thumbs_up' else "System learned - it will give better results next time!"

                return {
                    "status": "success",
                    "message": message,
                    "learned": True,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Failed to add feedback to training: {str(e)}")

        return {
            "status": "success",
            "message": "Feedback submitted successfully",
            "learned": False,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")
