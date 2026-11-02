"""
VIQ admin/dashboard endpoints — extracted from app/main.py verbatim
(structural move only, no behavior change).
"""

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app import state
from app.database import get_db, SearchHistory, UserFeedback, SystemAnalytics

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/v1/dashboard")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get comprehensive dashboard statistics from database"""
    try:
        dashboard_data = {
            "system_overview": {},
            "search_analytics": {},
            "feedback_analytics": {},
            "training_analytics": {},
            "performance_metrics": {},
            "recent_activity": []
        }

        # System Overview
        if state.collection:
            viq_count = state.collection.count()
            dashboard_data["system_overview"] = {
                "total_viq_questions": viq_count,
                "system_status": "healthy" if state.collection and state.client else "degraded",
                "openai_enabled": state.client is not None,
                "embedding_model": "text-embedding-3-small",
                "database": "ChromaDB",
                "uptime": "Running"
            }

        # Real Feedback Analytics from database
        all_feedback = db.query(UserFeedback).all()
        feedback_counts = {}
        for feedback in all_feedback:
            feedback_counts[feedback.feedback_type] = feedback_counts.get(feedback.feedback_type, 0) + 1

        total_feedback = len(all_feedback)
        positive_feedback = feedback_counts.get('thumbs_up', 0)
        negative_feedback = feedback_counts.get('thumbs_down', 0)
        corrections = feedback_counts.get('correction', 0)

        satisfaction = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0

        # Real Search Analytics from database
        today = datetime.now().date()
        searches_today = db.query(SearchHistory).filter(
            SearchHistory.timestamp >= today
        ).all()

        vessel_type_counts = {}
        total_response_time = 0
        query_counts = {}

        for search in searches_today:
            vessel_type_counts[search.vessel_type] = vessel_type_counts.get(search.vessel_type, 0) + 1
            total_response_time += search.time_ms
            query_counts[search.query] = query_counts.get(search.query, 0) + 1

        avg_response_time = int(total_response_time / len(searches_today)) if searches_today else 0
        most_searched = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        dashboard_data["search_analytics"] = {
            "total_searches_today": len(searches_today),
            "average_response_time": f"{avg_response_time}ms",
            "top_vessel_types": vessel_type_counts,
            "search_success_rate": round(satisfaction, 1) if total_feedback > 0 else 0.0,
            "most_searched_findings": [query for query, _ in most_searched]
        }

        dashboard_data["feedback_analytics"] = {
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "corrections_provided": corrections,
            "feedback_trend": "Improving" if positive_feedback > negative_feedback else "Needs Attention",
            "user_satisfaction": round(satisfaction, 1),
            "feedback_breakdown": feedback_counts
        }

        # Training Analytics
        try:
            training_collection = state.viq_chroma_client.get_collection("viq_training")
            training_count = training_collection.count()

            training_results = training_collection.get(include=['metadatas'])
            source_breakdown = {}
            recent_feedback = 0

            for metadata in training_results.get('metadatas', []):
                source = metadata.get('source', 'unknown')
                source_breakdown[source] = source_breakdown.get(source, 0) + 1

                timestamp = metadata.get('timestamp', '')
                if timestamp:
                    try:
                        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        if ts > datetime.now() - timedelta(days=1):
                            recent_feedback += 1
                    except Exception:
                        pass

            dashboard_data["training_analytics"] = {
                "total_training_examples": training_count,
                "source_breakdown": source_breakdown,
                "recent_feedback_24h": recent_feedback,
                "learning_rate": "Active" if recent_feedback > 0 else "Stable"
            }
        except Exception:
            dashboard_data["training_analytics"] = {
                "total_training_examples": 0,
                "source_breakdown": {},
                "recent_feedback_24h": 0,
                "learning_rate": "No Training Data"
            }

        # Performance metrics from real data
        recent_searches = db.query(SearchHistory).order_by(SearchHistory.timestamp.desc()).limit(100).all()
        avg_confidence = sum(s.confidence for s in recent_searches) / len(recent_searches) if recent_searches else 0
        high_confidence = len([s for s in recent_searches if s.confidence > 0.8])

        dashboard_data["performance_metrics"] = {
            "avg_similarity_score": round(avg_confidence, 2),
            "high_confidence_matches": high_confidence,
            "training_accuracy_improvement": f"+{len(all_feedback)}%" if all_feedback else "0%",
            "system_load": "Normal",
            "memory_usage": "2.1GB",
            "api_uptime": "99.8%"
        }

        # Recent activity from database
        recent_activity = []

        # Add recent searches
        for search in recent_searches[:3]:
            recent_activity.append({
                "timestamp": search.timestamp.isoformat(),
                "type": "search",
                "description": f"Finding matched to VIQ {search.result_viq}",
                "confidence": search.confidence
            })

        # Add recent feedback
        recent_feedback_items = db.query(UserFeedback).order_by(UserFeedback.timestamp.desc()).limit(3).all()
        for feedback in recent_feedback_items:
            recent_activity.append({
                "timestamp": feedback.timestamp.isoformat(),
                "type": "feedback" if feedback.feedback_type != "correction" else "correction",
                "description": f"{feedback.feedback_type.replace('_', ' ').title()} feedback received",
                "viq_number": feedback.suggested_viq,
                "learned": feedback.feedback_type == "correction"
            })

        # Sort by timestamp
        recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
        dashboard_data["recent_activity"] = recent_activity[:10]

        return dashboard_data

    except Exception as e:
        logger.error(f"Error getting dashboard stats: {str(e)}")
        return {
            "error": "Failed to load dashboard data",
            "message": str(e)
        }


@router.get("/api/v1/analytics")
async def get_analytics(db: Session = Depends(get_db)):
    """Get detailed analytics from database"""
    try:
        # Search analytics
        total_searches = db.query(SearchHistory).count()
        recent_searches = db.query(SearchHistory).filter(
            SearchHistory.timestamp >= datetime.now() - timedelta(days=7)
        ).all()

        # Vessel type distribution
        vessel_stats = {}
        confidence_stats = []
        response_times = []

        for search in recent_searches:
            vessel_stats[search.vessel_type] = vessel_stats.get(search.vessel_type, 0) + 1
            confidence_stats.append(search.confidence)
            response_times.append(search.time_ms)

        # Feedback analytics
        total_feedback = db.query(UserFeedback).count()
        recent_feedback = db.query(UserFeedback).filter(
            UserFeedback.timestamp >= datetime.now() - timedelta(days=7)
        ).all()

        feedback_by_type = {}
        for feedback in recent_feedback:
            feedback_by_type[feedback.feedback_type] = feedback_by_type.get(feedback.feedback_type, 0) + 1

        # Performance metrics
        avg_confidence = sum(confidence_stats) / len(confidence_stats) if confidence_stats else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        high_confidence_count = len([c for c in confidence_stats if c > 0.8])

        return {
            "search_analytics": {
                "total_searches": total_searches,
                "searches_last_7_days": len(recent_searches),
                "vessel_type_distribution": vessel_stats,
                "average_confidence": round(avg_confidence, 3),
                "high_confidence_percentage": round((high_confidence_count / len(confidence_stats) * 100), 1) if confidence_stats else 0
            },
            "feedback_analytics": {
                "total_feedback": total_feedback,
                "feedback_last_7_days": len(recent_feedback),
                "feedback_distribution": feedback_by_type,
                "positive_feedback_rate": round((feedback_by_type.get('thumbs_up', 0) / len(recent_feedback) * 100), 1) if recent_feedback else 0
            },
            "performance_metrics": {
                "average_response_time_ms": round(avg_response_time, 2),
                "average_confidence_score": round(avg_confidence, 3),
                "high_confidence_matches": high_confidence_count,
                "total_queries_processed": total_searches
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        return {
            "error": "Failed to load analytics data",
            "message": str(e)
        }


@router.get("/api/v1/system-analytics")
async def get_system_analytics(db: Session = Depends(get_db)):
    """Get system analytics from database"""
    try:
        # Get recent system analytics
        analytics = db.query(SystemAnalytics).order_by(SystemAnalytics.timestamp.desc()).limit(100).all()

        analytics_data = {}
        for analytic in analytics:
            metric_name = analytic.metric_name
            if metric_name not in analytics_data:
                analytics_data[metric_name] = []

            analytics_data[metric_name].append({
                "value": analytic.metric_value,
                "timestamp": analytic.timestamp.isoformat()
            })

        return {
            "system_analytics": analytics_data,
            "total_metrics": len(analytics),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting system analytics: {str(e)}")
        return {
            "system_analytics": {},
            "total_metrics": 0,
            "error": str(e)
        }


@router.post("/api/v1/system-analytics")
async def log_system_metric(request: Request, db: Session = Depends(get_db)):
    """Log a system metric to database"""
    try:
        data = await request.json()

        metric_log = SystemAnalytics(
            metric_name=data.get('metric_name'),
            metric_value=str(data.get('metric_value', ''))
        )
        db.add(metric_log)
        db.commit()

        return {
            "status": "success",
            "message": "Metric logged successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error logging system metric: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to log metric")


@router.get("/api/v1/search-history")
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


@router.get("/api/v1/search-history/export")
async def export_search_history(db: Session = Depends(get_db)):
    """Export search history as CSV"""
    import io
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
