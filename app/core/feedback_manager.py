import chromadb
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class FeedbackManager:
    """Manages user feedback and learns from corrections"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        self.feedback_collection = None
        self.training_collection = None
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Initialize feedback and training collections"""
        try:
            collections = [c.name for c in self.client.list_collections()]
            
            # Feedback collection
            if 'viq_feedback' in collections:
                self.feedback_collection = self.client.get_collection('viq_feedback')
            else:
                self.feedback_collection = self.client.create_collection(
                    name='viq_feedback',
                    metadata={"description": "User feedback on VIQ matches"}
                )
            
            # Training collection
            if 'viq_training' in collections:
                self.training_collection = self.client.get_collection('viq_training')
            
            logger.info(f"Feedback system ready: {self.feedback_collection.count()} feedbacks")
        except Exception as e:
            logger.error(f"Failed to initialize feedback collections: {str(e)}")
    
    async def save_thumbs_up(
        self,
        finding: str,
        viq_number: str,
        confidence: float,
        embedding: list
    ) -> Dict[str, Any]:
        """Save positive feedback (thumbs up)"""
        try:
            feedback_id = f"feedback_{datetime.now().timestamp()}"
            
            # Save feedback
            await asyncio.to_thread(
                self.feedback_collection.add,
                documents=[finding],
                embeddings=[embedding],
                metadatas=[{
                    "viq_number": viq_number,
                    "feedback_type": "thumbs_up",
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }],
                ids=[feedback_id]
            )
            
            # Add to training collection with high confidence
            if self.training_collection:
                training_id = f"train_{datetime.now().timestamp()}"
                await asyncio.to_thread(
                    self.training_collection.add,
                    documents=[finding],
                    embeddings=[embedding],
                    metadatas=[{
                        "viq_number": viq_number,
                        "source": "user_feedback",
                        "timestamp": datetime.now().isoformat()
                    }],
                    ids=[training_id]
                )
                logger.info(f"✅ Thumbs UP: '{finding[:50]}...' → VIQ {viq_number} (Added to training)")
                return {
                    "success": True,
                    "message": "Feedback saved! System learned from your confirmation.",
                    "learned": True,
                    "feedback_id": feedback_id
                }
            
            logger.info(f"✅ Thumbs UP: '{finding[:50]}...' → VIQ {viq_number}")
            return {
                "success": True,
                "message": "Feedback saved!",
                "learned": False,
                "feedback_id": feedback_id
            }
            
        except Exception as e:
            logger.error(f"Failed to save thumbs up: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "learned": False
            }
    
    async def save_thumbs_down(
        self,
        finding: str,
        wrong_viq: str,
        correct_viq: Optional[str],
        embedding: list,
        user_comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Save negative feedback (thumbs down) and learn correct answer"""
        try:
            feedback_id = f"feedback_{datetime.now().timestamp()}"
            
            # Save feedback
            metadata = {
                "wrong_viq": wrong_viq,
                "feedback_type": "thumbs_down",
                "timestamp": datetime.now().isoformat()
            }
            
            if correct_viq:
                metadata["correct_viq"] = correct_viq
            if user_comment:
                metadata["comment"] = user_comment
            
            await asyncio.to_thread(
                self.feedback_collection.add,
                documents=[finding],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[feedback_id]
            )
            
            # If user provided correct VIQ, add to training
            if correct_viq and self.training_collection:
                training_id = f"train_{datetime.now().timestamp()}"
                await asyncio.to_thread(
                    self.training_collection.add,
                    documents=[finding],
                    embeddings=[embedding],
                    metadatas=[{
                        "viq_number": correct_viq,
                        "source": "user_correction",
                        "timestamp": datetime.now().isoformat()
                    }],
                    ids=[training_id]
                )
                logger.info(f"👎 Thumbs DOWN: '{finding[:50]}...' | Wrong: {wrong_viq} → Correct: {correct_viq} (Learned!)")
                return {
                    "success": True,
                    "message": f"Thank you! System learned: VIQ {correct_viq} is correct for this finding.",
                    "learned": True,
                    "feedback_id": feedback_id
                }
            
            logger.info(f"👎 Thumbs DOWN: '{finding[:50]}...' | Wrong: {wrong_viq}")
            return {
                "success": True,
                "message": "Feedback saved. Please provide the correct VIQ number to help system learn.",
                "learned": False,
                "feedback_id": feedback_id,
                "needs_correction": True
            }
            
        except Exception as e:
            logger.error(f"Failed to save thumbs down: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "learned": False
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        try:
            if not self.feedback_collection:
                return {"total_feedback": 0}
            
            all_feedback = self.feedback_collection.get()
            total = len(all_feedback['ids'])
            
            thumbs_up = sum(1 for m in all_feedback['metadatas'] if m.get('feedback_type') == 'thumbs_up')
            thumbs_down = sum(1 for m in all_feedback['metadatas'] if m.get('feedback_type') == 'thumbs_down')
            
            # Count training data from Excel files
            training_data_count = self._count_training_data()
            
            # Calculate success rate based on actual feedback (thumbs_up vs thumbs_down)
            feedback_total = thumbs_up + thumbs_down
            success_rate = (thumbs_up / feedback_total * 100) if feedback_total > 0 else 0
            
            return {
                "total_feedback": total,
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "accuracy_rate": f"{success_rate:.1f}%" if feedback_total > 0 else "N/A",
                "training_data_rows": training_data_count
            }
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {str(e)}")
            return {"total_feedback": 0, "error": str(e)}
    
    def _count_training_data(self) -> int:
        """Count total training data rows from Excel files"""
        try:
            import pandas as pd
            import os
            
            training_dir = os.path.join(settings.DATA_DIR, 'training')
            if not os.path.exists(training_dir):
                return 0
            
            total_count = 0
            for file in os.listdir(training_dir):
                if file.endswith('.xlsx'):
                    try:
                        file_path = os.path.join(training_dir, file)
                        df = pd.read_excel(file_path)
                        total_count += len(df)
                    except Exception as e:
                        logger.error(f"Error reading {file}: {str(e)}")
            
            return total_count
            
        except Exception as e:
            logger.error(f"Failed to count training data: {str(e)}")
            return 0
