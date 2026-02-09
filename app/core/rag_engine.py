import openai
import json
import asyncio
from typing import List, Dict, Optional, Any, Tuple
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings
from app.core.vector_store import VectorStore
from app.core.learning_system import LearningSystem
from app.core.search_history import SearchHistoryTracker
from app.models.schemas import SearchMethod, VIQMatch

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.openai_client = None
        self.training_collection = None
        self.learning_system = LearningSystem()
        self.search_history = SearchHistoryTracker()
        self._initialize_openai()
        self._initialize_training_collection()
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("OpenAI client initialized for RAG engine")
            else:
                logger.warning("OpenAI API key not found - AI features disabled")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _initialize_training_collection(self):
        """Initialize training data collection"""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
            collections = [c.name for c in client.list_collections()]
            if 'viq_training' in collections:
                self.training_collection = client.get_collection('viq_training')
                count = self.training_collection.count()
                logger.info(f"Training collection loaded: {count} examples")
            else:
                logger.info("No training data found")
        except Exception as e:
            logger.error(f"Failed to load training collection: {str(e)}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def enhance_query(self, query: str, vessel_type: str = "All") -> str:
        """Enhance user query using LLM for better search results"""
        try:
            if not self.openai_client:
                return query
            
            enhancement_prompt = f"""
You are an expert in maritime vessel inspections and VIQ (Vessel Inspection Questionnaire) standards.
The user has described an audit finding or deficiency. Convert this into search terms that will find the exact VIQ question.

Audit Finding: "{query}"
Vessel Type: {vessel_type}

Instructions:
1. Identify the core inspection topic (safety equipment, procedures, maintenance, etc.)
2. Extract key technical terms and equipment names
3. Focus on what should be inspected or verified
4. Convert findings into question format keywords
5. Include compliance and procedural aspects

Example:
Finding: "Emergency fire pump not tested regularly"
Search Terms: "emergency fire pump testing procedure maintenance records demonstration"

Search Terms:"""

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": enhancement_prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            enhanced_query = response.choices[0].message.content.strip()
            # Remove quotes if present
            enhanced_query = enhanced_query.strip('"')
            logger.info(f"Query enhanced: '{query}' -> '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {str(e)}")
            return query
    
    async def search(
        self, 
        query: str, 
        vessel_type: str = "All", 
        top_k: int = 3,
        search_method: SearchMethod = SearchMethod.HYBRID,
        confidence_threshold: float = 0.3,
        enhance_query: bool = True
    ) -> Tuple[List[Dict[str, Any]], str, float]:
        """Perform search with optional query enhancement"""
        start_time = datetime.now()
        
        try:
            # Check training data first for exact matches
            training_match = await self._check_training_data(query)
            if training_match:
                logger.info(f"Found training match: {training_match['viq_number']}")
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                return [training_match], query, processing_time
            
            # Enhance query if requested and OpenAI is available
            enhanced_query = query
            if enhance_query and self.openai_client:
                enhanced_query = await self.enhance_query(query, vessel_type)
            
            # Perform search based on method
            if search_method == SearchMethod.SEMANTIC:
                results = await self.vector_store.semantic_search(
                    enhanced_query, vessel_type, top_k, confidence_threshold
                )
            elif search_method == SearchMethod.KEYWORD:
                results = await self.vector_store.keyword_search(
                    enhanced_query, vessel_type, top_k, confidence_threshold
                )
            else:  # HYBRID
                results = await self.vector_store.hybrid_search(
                    enhanced_query, vessel_type, top_k, confidence_threshold
                )
            
            # Rerank results if we have enough
            if len(results) > 1 and self.openai_client:
                results = await self.rerank_results(query, results, top_k)
            
            # Auto-learn from high confidence results
            if results and results[0]['confidence_score'] >= 0.85:
                await self.learning_system.save_query_result(
                    query=query,
                    viq_number=results[0]['viq_number'],
                    confidence=results[0]['confidence_score'],
                    user_confirmed=False
                )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log search to history
            await self.search_history.log_search(
                query=query,
                results=results,
                vessel_type=vessel_type,
                search_method=search_method.value,
                processing_time_ms=processing_time,
                enhanced_query=enhanced_query
            )
            
            return results, enhanced_query, processing_time
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return [], query, processing_time
    
    async def _check_training_data(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if query matches training data"""
        try:
            if not self.training_collection:
                logger.info("Training collection not available")
                return None
            
            # Get embedding for query
            if not self.openai_client:
                logger.info("OpenAI client not available for training check")
                return None
            
            logger.info(f"Checking training data for query: {query[:50]}...")
            
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=query
            )
            query_embedding = response.data[0].embedding
            
            # Search training data
            results = await asyncio.to_thread(
                self.training_collection.query,
                query_embeddings=[query_embedding],
                n_results=1
            )
            
            if results['distances'][0] and results['distances'][0][0] < 0.35:  # High similarity threshold
                distance = results['distances'][0][0]
                viq_number = results['metadatas'][0][0]['viq_number']
                logger.info(f"Training match found! VIQ: {viq_number}, Distance: {distance:.4f}")
                
                # Get full VIQ question from main collection using direct lookup
                try:
                    import chromadb
                    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
                    viq_collection = chroma_client.get_collection(settings.CHROMA_COLLECTION_NAME)
                    
                    # Try with proper operator first, fallback to simple get if needed
                    try:
                        viq_data = await asyncio.to_thread(
                            viq_collection.get,
                            where={"viq_number": {"$eq": viq_number}}
                        )
                    except Exception:
                        # Fallback: get all and filter manually
                        all_data = await asyncio.to_thread(
                            viq_collection.get,
                            include=['metadatas', 'documents']
                        )
                        viq_data = {'ids': [], 'metadatas': [], 'documents': []}
                        if all_data['metadatas']:
                            for i, metadata in enumerate(all_data['metadatas']):
                                if metadata.get('viq_number') == viq_number:
                                    viq_data['ids'].append(all_data['ids'][i])
                                    viq_data['metadatas'].append(metadata)
                                    viq_data['documents'].append(all_data['documents'][i])
                                    break
                    
                    if viq_data['ids']:
                        result = {
                            'viq_number': viq_data['metadatas'][0]['viq_number'],
                            'question': viq_data['documents'][0],
                            'vessel_type': viq_data['metadatas'][0].get('vessel_type', 'All'),
                            'confidence_score': 0.98,
                            'similarity_score': 0.98,
                            'context': '',
                            'source_file': viq_data['metadatas'][0].get('source_file', ''),
                            'metadata': viq_data['metadatas'][0]
                        }
                        logger.info(f"✅ Returning training match: VIQ {viq_number} (98% confidence)")
                        return result
                    else:
                        logger.warning(f"VIQ {viq_number} not found in collection")
                except Exception as e:
                    logger.error(f"Error fetching VIQ {viq_number}: {str(e)}")
            else:
                if results['distances'][0]:
                    logger.info(f"No training match (distance: {results['distances'][0][0]:.4f} > 0.35)")
            
            return None
            
        except Exception as e:
            logger.error(f"Training data check failed: {str(e)}")
            return None
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def rerank_results(
        self, 
        original_query: str, 
        results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank search results using LLM for better accuracy"""
        try:
            if not self.openai_client or len(results) <= 1:
                return results
            
            # Prepare results for reranking
            candidates = []
            for i, result in enumerate(results[:settings.RERANK_TOP_K]):
                candidates.append({
                    "index": i,
                    "viq_number": result['viq_number'],
                    "question": result['question'][:500],  # Truncate for token limits
                    "vessel_type": result['vessel_type']
                })
            
            rerank_prompt = f"""
You are an expert maritime inspector. The user reported this audit finding. Rank the VIQ questions by how well they match what should have been inspected.

Audit Finding: "{original_query}"

VIQ Questions:
{json.dumps(candidates, indent=2)}

Instructions:
1. Find questions that directly address the finding topic
2. Prioritize questions about procedures, testing, and maintenance
3. Look for exact equipment or system matches
4. Consider compliance verification aspects
5. Return indices in order: [most_relevant, second_most, ...]

Ranked indices:"""

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": rerank_prompt}],
                max_tokens=100,
                temperature=0.0
            )
            
            # Parse reranking response
            try:
                ranked_indices = json.loads(response.choices[0].message.content.strip())
                if isinstance(ranked_indices, list):
                    # Reorder results based on ranking
                    reranked_results = []
                    for idx in ranked_indices[:top_k]:
                        if 0 <= idx < len(results):
                            # Boost confidence score for reranked results
                            result = results[idx].copy()
                            result['confidence_score'] = min(1.0, result['confidence_score'] + 0.1)
                            reranked_results.append(result)
                    
                    # Add any remaining results that weren't ranked
                    used_indices = set(ranked_indices[:len(results)])
                    for i, result in enumerate(results):
                        if i not in used_indices:
                            reranked_results.append(result)
                    
                    logger.info(f"Results reranked successfully: {len(reranked_results)} results")
                    return reranked_results
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse reranking response: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return results
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_finding(
        self, 
        query: str, 
        matches: List[Dict[str, Any]], 
        vessel_type: str = "All"
    ) -> Dict[str, Any]:
        """Provide AI analysis of the audit finding and VIQ matches"""
        try:
            if not self.openai_client or not matches:
                return {}
            
            # Prepare context from top matches
            context_matches = []
            for match in matches[:3]:
                context_matches.append({
                    'viq_number': match['viq_number'],
                    'question': match['question'][:300],  # Truncate for token limits
                    'vessel_type': match['vessel_type'],
                    'confidence': round(match['confidence_score'], 2)
                })
            
            analysis_prompt = f"""
You are a senior maritime inspector and VIQ expert. Analyze this audit finding and provide comprehensive insights.

Audit Finding: "{query}"
Vessel Type: {vessel_type}

Relevant VIQ Questions:
{json.dumps(context_matches, indent=2)}

Provide analysis in the following JSON format:
{{
    "analysis": "Detailed analysis of the finding and its relationship to VIQ requirements",
    "recommendations": ["Specific actionable recommendations"],
    "risk_assessment": "Risk level and potential consequences",
    "compliance_notes": "Key compliance considerations",
    "related_areas": ["Other inspection areas that might be affected"]
}}

Analysis:"""

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=settings.OPENAI_TEMPERATURE
            )
            
            # Parse analysis response
            try:
                content = response.choices[0].message.content.strip()
                # Remove markdown code blocks if present
                if content.startswith('```json'):
                    content = content.replace('```json', '').replace('```', '').strip()
                elif content.startswith('```'):
                    content = content.replace('```', '').strip()
                
                analysis = json.loads(content)
                logger.info("AI analysis generated successfully")
                return analysis
            except json.JSONDecodeError:
                # Fallback to plain text analysis
                return {
                    "analysis": response.choices[0].message.content.strip(),
                    "recommendations": [],
                    "risk_assessment": "Analysis available in text format",
                    "compliance_notes": "",
                    "related_areas": []
                }
            
        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            return {}
    
    async def get_vessel_types(self) -> List[str]:
        """Get available vessel types"""
        try:
            stats = await self.vector_store.get_collection_stats()
            vessel_types = list(stats.get('vessel_type_distribution', {}).keys())
            
            # Ensure 'All' is first and standard types are included
            result = ["All"]
            for vtype in settings.VESSEL_TYPES[1:]:  # Skip 'All' as it's already added
                if vtype in vessel_types:
                    result.append(vtype)
            
            # Add any additional vessel types found in data
            for vtype in vessel_types:
                if vtype not in result and vtype != "Unknown":
                    result.append(vtype)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get vessel types: {str(e)}")
            return settings.VESSEL_TYPES
    
    def _get_enhanced_learning_stats(self) -> Dict[str, Any]:
        """Get enhanced learning statistics including training data count"""
        try:
            learning_stats = self.learning_system.get_stats()
            
            # Get training collection count
            training_count = 0
            if self.training_collection:
                try:
                    training_count = self.training_collection.count()
                    logger.info(f"Training collection count: {training_count}")
                except Exception as e:
                    logger.error(f"Failed to get training count: {str(e)}")
            
            # Count actual training data from Excel files
            training_data_count = self._count_training_data()
            
            # Also check main vector store document count
            vector_count = 0
            try:
                if hasattr(self.vector_store, 'collection') and self.vector_store.collection:
                    vector_count = self.vector_store.collection.count()
                    logger.info(f"Vector store document count: {vector_count}")
            except Exception as e:
                logger.error(f"Failed to get vector count: {str(e)}")
            
            learning_stats["training_examples"] = training_count
            learning_stats["training_data_rows"] = training_data_count  # This is the 20k+ data
            learning_stats["processed_documents"] = vector_count
            return learning_stats
            
        except Exception as e:
            logger.error(f"Failed to get enhanced learning stats: {str(e)}")
            return {"total_learned": 0, "training_examples": 0, "training_data_rows": 0, "processed_documents": 0, "status": "error"}
    
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
            
            logger.info(f"Total training data rows: {total_count}")
            return total_count
            
        except Exception as e:
            logger.error(f"Failed to count training data: {str(e)}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            vector_stats = await self.vector_store.get_collection_stats()
            
            return {
                'total_questions': vector_stats.get('total_documents', 0),
                'vessel_types': await self.get_vessel_types(),
                'collections': [settings.CHROMA_COLLECTION_NAME],
                'status': 'ready' if self.vector_store.is_initialized else 'not_ready',
                'search_methods': [method.value for method in SearchMethod],
                'openai_available': self.openai_client is not None,
                'vector_store_status': 'initialized' if self.vector_store.is_initialized else 'not_initialized',
                'last_updated': datetime.now().isoformat(),
                'vessel_type_distribution': vector_stats.get('vessel_type_distribution', {}),
                'embedding_model': settings.EMBEDDING_MODEL,
                'openai_model': settings.OPENAI_MODEL if self.openai_client else None,
                'learning_stats': self._get_enhanced_learning_stats()
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {
                'total_questions': 0,
                'vessel_types': ["All"],
                'collections': [],
                'status': 'error',
                'search_methods': [],
                'openai_available': False,
                'vector_store_status': 'error',
                'last_updated': datetime.now().isoformat()
            }