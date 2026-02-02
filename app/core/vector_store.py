import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.models.schemas import VIQDocument

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.openai_client = None
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize vector store with ChromaDB and OpenAI embeddings"""
        try:
            # Initialize ChromaDB
            persist_dir = settings.BASE_DIR / settings.CHROMA_PERSIST_DIRECTORY
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=settings.CHROMA_COLLECTION_NAME
                )
                logger.info(f"Loaded existing collection: {settings.CHROMA_COLLECTION_NAME}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=settings.CHROMA_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {settings.CHROMA_COLLECTION_NAME}")
            
            # Initialize OpenAI client if API key is available
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("OpenAI client initialized for embeddings")
            else:
                logger.warning("No OpenAI API key found, using simple text matching")
            
            self.is_initialized = True
            logger.info("Vector store initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI API with retry logic"""
        try:
            if not self.openai_client:
                raise ValueError("OpenAI client not initialized")
            
            response = self.openai_client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=texts
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {str(e)}")
            # Return dummy embeddings for fallback
            return [[0.0] * 1536 for _ in texts]
    
    async def add_documents(self, documents: List[VIQDocument]) -> bool:
        """Add documents to ChromaDB vector store"""
        try:
            if not self.is_initialized or not self.collection:
                logger.error("Vector store not initialized")
                return False
            
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            embeddings = []
            
            for i, doc in enumerate(documents):
                # Create unique ID
                doc_id = f"{doc.viq_number}_{i}"
                ids.append(doc_id)
                
                # Combine question and guidance for embedding
                full_text = doc.question
                if doc.guidance:
                    full_text += f" {doc.guidance}"
                texts.append(full_text)
                
                # Prepare metadata
                metadata = {
                    "viq_number": doc.viq_number,
                    "question": doc.question,
                    "vessel_type": doc.vessel_type,
                    "guidance": doc.guidance or "",
                    "source_file": doc.source_file,
                    "chapter": doc.chapter or "",
                    "section": doc.section or "",
                    "category": doc.metadata.get("category", "") if doc.metadata else ""
                }
                metadatas.append(metadata)
            
            # Get embeddings if OpenAI is available
            if self.openai_client:
                # Get embeddings in batches
                batch_size = 100
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = await self.get_openai_embeddings(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                embeddings = all_embeddings
            
            # Add to ChromaDB
            if embeddings:
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return False
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    async def semantic_search(
        self, 
        query: str, 
        vessel_type: str = "All", 
        top_k: int = 5,
        confidence_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using ChromaDB"""
        try:
            if not self.is_initialized or not self.collection:
                return []
            
            # Try with ChromaDB filtering first
            try:
                # Prepare where clause for vessel type filtering
                where_clause = None
                if vessel_type != "All":
                    where_clause = {"vessel_type": {"$eq": vessel_type}}
                
                # Query ChromaDB
                if self.openai_client:
                    # Get query embedding
                    query_embedding = await self.get_openai_embeddings([query])
                    if query_embedding:
                        results = self.collection.query(
                            query_embeddings=[query_embedding[0]],
                            n_results=top_k * 2,  # Get more results to filter
                            where=where_clause
                        )
                    else:
                        # Fallback to text search
                        results = self.collection.query(
                            query_texts=[query],
                            n_results=top_k * 2,
                            where=where_clause
                        )
                else:
                    # Use ChromaDB's built-in text search
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=top_k * 2,
                        where=where_clause
                    )
            except Exception as filter_error:
                logger.warning(f"ChromaDB filtering failed, falling back to post-filtering: {str(filter_error)}")
                # Fallback: Query without filtering and filter results manually
                if self.openai_client:
                    query_embedding = await self.get_openai_embeddings([query])
                    if query_embedding:
                        results = self.collection.query(
                            query_embeddings=[query_embedding[0]],
                            n_results=top_k * 3  # Get more results for manual filtering
                        )
                    else:
                        results = self.collection.query(
                            query_texts=[query],
                            n_results=top_k * 3
                        )
                else:
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=top_k * 3
                    )
            
            # Format results with additional vessel type filtering
            matches = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if results['distances'] else 0.5
                    similarity = 1.0 - distance  # Convert distance to similarity
                    
                    # Additional vessel type check (in case ChromaDB filtering didn't work)
                    if vessel_type != "All":
                        doc_vessel_type = metadata.get('vessel_type', 'All')
                        if doc_vessel_type != vessel_type and doc_vessel_type != 'All':
                            continue
                    
                    if similarity >= confidence_threshold:
                        matches.append({
                            'viq_number': metadata['viq_number'],
                            'question': metadata['question'],
                            'vessel_type': metadata['vessel_type'],
                            'similarity_score': similarity,
                            'confidence_score': similarity,
                            'source_file': metadata['source_file'],
                            'guidance': metadata['guidance'],
                            'metadata': {'category': metadata.get('category', '')}
                        })
            
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    async def keyword_search(
        self, 
        query: str, 
        vessel_type: str = "All", 
        top_k: int = 5,
        confidence_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Perform keyword-based search"""
        try:
            if not self.documents:
                return []
            
            matches = []
            query_words = set(query.lower().split())
            
            for doc in self.documents:
                # Filter by vessel type
                if vessel_type != "All" and doc.vessel_type != vessel_type:
                    continue
                
                # Combine question and guidance for search
                full_text = doc.question.lower()
                if doc.guidance:
                    full_text += f" {doc.guidance.lower()}"
                
                doc_words = set(full_text.split())
                
                # Calculate keyword overlap score
                if query_words and doc_words:
                    intersection = query_words.intersection(doc_words)
                    union = query_words.union(doc_words)
                    keyword_score = len(intersection) / len(union) if union else 0.0
                    
                    # Boost score for exact phrase matches
                    if query.lower() in full_text:
                        keyword_score += 0.2
                    
                    if keyword_score >= confidence_threshold:
                        matches.append({
                            'viq_number': doc.viq_number,
                            'question': doc.question,
                            'vessel_type': doc.vessel_type,
                            'similarity_score': keyword_score,
                            'confidence_score': keyword_score,
                            'source_file': doc.source_file,
                            'guidance': doc.guidance,
                            'metadata': doc.metadata
                        })
            
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    async def hybrid_search(
        self, 
        query: str, 
        vessel_type: str = "All", 
        top_k: int = 5,
        confidence_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search"""
        try:
            # Get semantic results
            semantic_results = await self.semantic_search(
                query, vessel_type, top_k * 2, confidence_threshold * 0.8
            )
            
            # Get keyword results
            keyword_results = await self.keyword_search(
                query, vessel_type, top_k * 2, confidence_threshold * 0.6
            )
            
            # Combine results
            combined = {}
            
            # Add semantic results with higher weight
            for result in semantic_results:
                viq_num = result['viq_number']
                result['final_score'] = result['similarity_score'] * 0.7
                combined[viq_num] = result
            
            # Add keyword results with lower weight, boost if already exists
            for result in keyword_results:
                viq_num = result['viq_number']
                if viq_num in combined:
                    combined[viq_num]['final_score'] += result['similarity_score'] * 0.3
                    combined[viq_num]['confidence_score'] = min(1.0, combined[viq_num]['confidence_score'] + 0.1)
                else:
                    result['final_score'] = result['similarity_score'] * 0.4
                    combined[viq_num] = result
            
            # Sort by final score
            final_results = list(combined.values())
            final_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Update similarity scores
            for result in final_results:
                result['similarity_score'] = result['final_score']
                del result['final_score']
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return await self.semantic_search(query, vessel_type, top_k, confidence_threshold)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics from ChromaDB"""
        try:
            if not self.collection:
                return {'total_documents': 0}
            
            # Get collection count
            count = self.collection.count()
            
            if count == 0:
                return {'total_documents': 0}
            
            # Get sample of documents to analyze vessel types
            sample_results = self.collection.get(
                limit=min(1000, count),
                include=['metadatas']
            )
            
            # Count vessel types
            vessel_types = {}
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    vtype = metadata.get('vessel_type', 'Unknown')
                    vessel_types[vtype] = vessel_types.get(vtype, 0) + 1
            
            return {
                'total_documents': count,
                'vessel_type_distribution': vessel_types,
                'has_embeddings': True,
                'embedding_model': settings.OPENAI_EMBEDDING_MODEL if self.openai_client else 'chromadb_default'
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {'total_documents': 0}
    
    async def reset_collection(self) -> bool:
        """Reset the ChromaDB collection (delete all data)"""
        try:
            if self.collection:
                # Delete the collection
                self.chroma_client.delete_collection(settings.CHROMA_COLLECTION_NAME)
                
                # Recreate the collection
                self.collection = self.chroma_client.create_collection(
                    name=settings.CHROMA_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
                
            logger.info("Collection reset successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            return False