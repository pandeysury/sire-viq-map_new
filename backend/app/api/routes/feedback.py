import io
import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core import rag_engine
from app.database import SearchHistory, UserFeedback, get_db

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/search-history")
async def get_search_history(db: Session = Depends(get_db)):
    """Get real search history from database"""
    try:
        # Get recent searches from database
        searches = db.query(SearchHistory).order_by(SearchHistory.timestamp.desc()).limit(1000).all()

        search_history = []
        for search in searches:
            # Get feedback for this search if exists
            feedback = db.query(UserFeedback).filter(
                UserFeedback.query == search.query,
                UserFeedback.suggested_viq == search.result_viq
            ).first()

            search_data = {
                "id": search.id,
                "query": search.query,
                "result_viq": search.result_viq,
                "confidence": search.confidence,
                "time_ms": search.time_ms,
                "vessel_type": search.vessel_type,
                "search_method": search.search_method,
                "timestamp": search.timestamp.isoformat()
            }

            if feedback:
                search_data["feedback"] = feedback.feedback_type
                if feedback.correct_viq:
                    search_data["correct_viq"] = feedback.correct_viq
            else:
                search_data["feedback"] = None

            search_history.append(search_data)

        return {
            "search_history": search_history,
            "total_count": len(search_history),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting search history: {str(e)}")
        return {
            "search_history": [],
            "total_count": 0,
            "error": str(e)
        }


@router.get("/search-history/export")
async def export_search_history(db: Session = Depends(get_db)):
    """Export search history as CSV"""
    try:
        searches = db.query(SearchHistory).order_by(SearchHistory.timestamp.desc()).limit(1000).all()
        csv_data = io.StringIO()
        csv_data.write("ID,Query,Result VIQ,Confidence,Time (ms),Vessel Type,Search Method,Feedback,Correct VIQ,Timestamp\n")
        for search in searches:
            feedback = db.query(UserFeedback).filter(
                UserFeedback.query == search.query,
                UserFeedback.suggested_viq == search.result_viq
            ).first()
            query_escaped = search.query.replace('"', '""')
            result_viq = search.result_viq if search.result_viq else 'N/A'
            feedback_type = feedback.feedback_type if feedback else 'None'
            correct_viq = feedback.correct_viq if feedback and feedback.correct_viq else 'N/A'
            csv_data.write(f'{search.id},"{query_escaped}","{result_viq}",{search.confidence},{search.time_ms},{search.vessel_type},{search.search_method},{feedback_type},"{correct_viq}",{search.timestamp.isoformat()}\n')
        csv_data.seek(0)
        return StreamingResponse(iter([csv_data.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=search_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"})
    except Exception as e:
        logger.error(f"Error exporting search history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export search history")


@router.get("/feedback/stats")
async def get_feedback_stats(db: Session = Depends(get_db)):
    """Get real feedback statistics from database"""
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


@router.post("/feedback")
async def submit_feedback(request: Request, db: Session = Depends(get_db)):
    """Submit user feedback and learn from corrections"""
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
                training_collection = rag_engine.chroma_client.get_collection("viq_training")

                response = rag_engine.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=finding
                )
                embedding = response.data[0].embedding

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
