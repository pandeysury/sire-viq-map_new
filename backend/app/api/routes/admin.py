from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from app.core.rag_engine import RAGEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Will be set by main.py
rag_engine: RAGEngine = None

def set_dependencies(engine: RAGEngine):
    """Set RAG engine dependency"""
    global rag_engine
    rag_engine = engine

@router.get("/admin/search-history")
async def get_search_history(limit: int = 100) -> Dict[str, Any]:
    """Get recent search history"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=500, detail="RAG engine not initialized")
        
        history = rag_engine.search_history.get_recent_searches(limit)
        
        return {
            "searches": history,
            "total": len(history),
            "message": "Search history retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Failed to get search history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/admin/search-stats")
async def get_search_stats() -> Dict[str, Any]:
    """Get search statistics"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=500, detail="RAG engine not initialized")
        
        stats = rag_engine.search_history.get_stats()
        
        return {
            "stats": stats,
            "message": "Search statistics retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Failed to get search stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/admin/dashboard")
async def get_dashboard_data() -> Dict[str, Any]:
    """Get complete dashboard data"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=500, detail="RAG engine not initialized")
        
        # Get all stats
        system_stats = await rag_engine.get_stats()
        search_stats = rag_engine.search_history.get_stats()
        search_history = rag_engine.search_history.get_recent_searches(100)
        
        # Get feedback stats from feedback manager
        from app.api.routes.feedback import feedback_manager
        feedback_stats = feedback_manager.get_stats()
        
        # Get training data count
        training_count = 0
        if rag_engine.training_collection:
            training_count = rag_engine.training_collection.count()
        
        return {
            "system": system_stats,
            "search_stats": search_stats,
            "recent_searches": search_history,
            "feedback_stats": feedback_stats,
            "training_count": training_count,
            "timestamp": system_stats.get('last_updated'),
            "message": "Dashboard data retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
