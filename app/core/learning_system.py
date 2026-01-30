import chromadb
import asyncio
from datetime import datetime
from typing import Optional
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class LearningSystem:
    """Auto-learning system that saves user queries for future training"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        self.learning_collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get learning collection"""
        try:
            collections = [c.name for c in self.client.list_collections()]
            if 'viq_learning' in collections:
                self.learning_collection = self.client.get_collection('viq_learning')
            else:
                self.learning_collection = self.client.create_collection(
                    name='viq_learning',
                    metadata={"description": "Auto-learned query-VIQ pairs"}
                )
            logger.info(f"Learning collection ready: {self.learning_collection.count()} examples")
        except Exception as e:
            logger.error(f"Failed to initialize learning collection: {str(e)}")
    
    async def save_query_result(
        self, 
        query: str, 
        viq_number: str, 
        confidence: float,
        user_confirmed: bool = False
    ):
        """Save query-VIQ pair for future learning"""
        try:
            if not self.learning_collection:
                return
            
            # Only save high confidence or user-confirmed results
            if confidence < 0.85 and not user_confirmed:
                return
            
            # Create unique ID
            query_id = f"learn_{datetime.now().timestamp()}"
            
            # Save to learning collection
            await asyncio.to_thread(
                self.learning_collection.add,
                documents=[query],
                metadatas=[{
                    "viq_number": viq_number,
                    "confidence": confidence,
                    "user_confirmed": user_confirmed,
                    "timestamp": datetime.now().isoformat()
                }],
                ids=[query_id]
            )
            
            logger.info(f"✅ Learned: '{query[:50]}...' → VIQ {viq_number}")
            
        except Exception as e:
            logger.error(f"Failed to save learning: {str(e)}")
    
    async def check_learned_query(self, query: str, embedding: list) -> Optional[str]:
        """Check if we've learned this query before"""
        try:
            if not self.learning_collection:
                return None
            
            results = await asyncio.to_thread(
                self.learning_collection.query,
                query_embeddings=[embedding],
                n_results=1
            )
            
            if results['distances'][0] and results['distances'][0][0] < 0.25:
                viq_number = results['metadatas'][0][0]['viq_number']
                logger.info(f"📚 Found learned query: VIQ {viq_number}")
                return viq_number
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check learned query: {str(e)}")
            return None
    
    def get_stats(self) -> dict:
        """Get learning statistics"""
        try:
            if not self.learning_collection:
                return {"total_learned": 0}
            
            count = self.learning_collection.count()
            return {
                "total_learned": count,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Failed to get learning stats: {str(e)}")
            return {"total_learned": 0, "status": "error"}
