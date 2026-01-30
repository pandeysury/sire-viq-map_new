import chromadb
import asyncio
from datetime import datetime
from typing import List, Dict, Any
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class SearchHistoryTracker:
    """Track all search queries and results for analytics"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        self.history_collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize search history collection"""
        try:
            collections = [c.name for c in self.client.list_collections()]
            if 'search_history' in collections:
                self.history_collection = self.client.get_collection('search_history')
            else:
                self.history_collection = self.client.create_collection(
                    name='search_history',
                    metadata={"description": "Search query history and results"}
                )
            logger.info(f"Search history ready: {self.history_collection.count()} searches")
        except Exception as e:
            logger.error(f"Failed to initialize search history: {str(e)}")
    
    async def log_search(
        self,
        query: str,
        results: List[Dict[str, Any]],
        vessel_type: str,
        search_method: str,
        processing_time_ms: float,
        enhanced_query: str = None
    ):
        """Log a search query and its results"""
        try:
            if not self.history_collection:
                return
            
            search_id = f"search_{datetime.now().timestamp()}"
            
            # Prepare metadata
            metadata = {
                "query": query,
                "vessel_type": vessel_type,
                "search_method": search_method,
                "processing_time_ms": processing_time_ms,
                "num_results": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
            if enhanced_query:
                metadata["enhanced_query"] = enhanced_query
            
            if results:
                metadata["top_viq"] = results[0]['viq_number']
                metadata["top_confidence"] = results[0]['confidence_score']
            
            # Store in collection
            await asyncio.to_thread(
                self.history_collection.add,
                documents=[query],
                metadatas=[metadata],
                ids=[search_id]
            )
            
            logger.info(f"📝 Logged search: '{query[:50]}...' → {len(results)} results")
            
        except Exception as e:
            logger.error(f"Failed to log search: {str(e)}")
    
    def get_recent_searches(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent search history"""
        try:
            if not self.history_collection:
                return []
            
            all_data = self.history_collection.get()
            
            # Convert to list of dicts
            searches = []
            for i in range(len(all_data['ids'])):
                search = {
                    'id': all_data['ids'][i],
                    'query': all_data['documents'][i],
                    **all_data['metadatas'][i]
                }
                searches.append(search)
            
            # Sort by timestamp (newest first)
            searches.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return searches[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get search history: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        try:
            if not self.history_collection:
                return {"total_searches": 0}
            
            all_data = self.history_collection.get()
            total = len(all_data['ids'])
            
            if total == 0:
                return {"total_searches": 0}
            
            # Calculate stats
            processing_times = [m.get('processing_time_ms', 0) for m in all_data['metadatas']]
            confidences = [m.get('top_confidence', 0) for m in all_data['metadatas'] if 'top_confidence' in m]
            
            return {
                "total_searches": total,
                "avg_processing_time_ms": sum(processing_times) / len(processing_times) if processing_times else 0,
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
                "search_methods": self._count_by_field(all_data['metadatas'], 'search_method'),
                "vessel_types": self._count_by_field(all_data['metadatas'], 'vessel_type')
            }
            
        except Exception as e:
            logger.error(f"Failed to get search stats: {str(e)}")
            return {"total_searches": 0, "error": str(e)}
    
    def _count_by_field(self, metadatas: List[Dict], field: str) -> Dict[str, int]:
        """Count occurrences of field values"""
        counts = {}
        for m in metadatas:
            value = m.get(field, 'Unknown')
            counts[value] = counts.get(value, 0) + 1
        return counts
