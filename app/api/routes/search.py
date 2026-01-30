from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import asyncio
import logging
import time
from datetime import datetime

from app.models.schemas import (
    QueryRequest, QueryResponse, EnhancedQueryResponse, VIQMatch,
    SystemStats, ProcessingStatus, ErrorResponse, HealthCheck,
    SearchMethod, VesselType
)
from app.core.rag_engine import RAGEngine
from app.core.pdf_processor import PDFProcessor
from app.utils.helpers import validate_query, create_error_response
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances (will be injected via dependency)
rag_engine: Optional[RAGEngine] = None
pdf_processor: Optional[PDFProcessor] = None

def get_rag_engine() -> RAGEngine:
    """Dependency to get RAG engine instance"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    return rag_engine

def get_pdf_processor() -> PDFProcessor:
    """Dependency to get PDF processor instance"""
    if not pdf_processor:
        raise HTTPException(status_code=503, detail="PDF processor not initialized")
    return pdf_processor

def set_dependencies(engine: RAGEngine, processor: PDFProcessor):
    """Set global dependencies"""
    global rag_engine, pdf_processor
    rag_engine = engine
    pdf_processor = processor

@router.post("/search", response_model=QueryResponse)
async def search_viq(
    request: QueryRequest,
    engine: RAGEngine = Depends(get_rag_engine)
):
    """Search for matching VIQ questions using various methods"""
    try:
        # Validate request
        if not validate_query(request.query):
            raise HTTPException(
                status_code=400, 
                detail="Query must be between 3 and 5000 characters"
            )
        
        # Perform search
        results, enhanced_query, processing_time = await engine.search(
            query=request.query,
            vessel_type=request.vessel_type.value,
            top_k=request.top_k,
            search_method=request.search_method,
            confidence_threshold=request.confidence_threshold,
            enhance_query=True
        )
        
        # Convert results to VIQMatch objects
        matches = []
        for result in results:
            match = VIQMatch(
                viq_number=result['viq_number'],
                question=result['question'],
                vessel_type=result['vessel_type'],
                similarity_score=round(result['similarity_score'], 4),
                confidence_score=round(result['confidence_score'], 4),
                context=result.get('guidance', '') if request.include_context else None,
                source_file=result.get('source_file', ''),
                metadata=result.get('metadata', {})
            )
            matches.append(match)
        
        return QueryResponse(
            matches=matches,
            query=request.query,
            vessel_type=request.vessel_type.value,
            total_results=len(matches),
            search_method=request.search_method.value,
            processing_time_ms=round(processing_time, 2),
            enhanced_query=enhanced_query if enhanced_query != request.query else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/analyze-finding", response_model=EnhancedQueryResponse)
async def analyze_finding(
    request: QueryRequest,
    engine: RAGEngine = Depends(get_rag_engine)
):
    """Analyze audit finding with AI enhancement and detailed insights"""
    try:
        # Validate request
        if not validate_query(request.query):
            raise HTTPException(
                status_code=400, 
                detail="Query must be between 3 and 5000 characters"
            )
        
        # For AI Analysis, check training data first for best match
        start_time = time.time()
        
        # Check training data for exact match (98% confidence)
        training_match = await engine._check_training_data(request.query)
        
        # Get diverse results from hybrid search
        enhanced_query = request.query
        if engine.openai_client:
            enhanced_query = await engine.enhance_query(request.query, request.vessel_type.value)
        
        # Get more candidates for diversity (2x requested)
        search_top_k = request.top_k * 2
        results = await engine.vector_store.hybrid_search(
            enhanced_query,
            request.vessel_type.value,
            search_top_k,
            confidence_threshold=0.1
        )
        
        # If training match exists and passes vessel filter, add it at the top
        if training_match:
            # Check vessel type filter
            match_vessel = training_match.get('vessel_type', 'All')
            if request.vessel_type.value == 'All' or match_vessel == 'All' or match_vessel == request.vessel_type.value:
                # Remove duplicate if exists in results
                results = [r for r in results if r['viq_number'] != training_match['viq_number']]
                # Add training match at top
                results.insert(0, training_match)
        
        # Keep only requested top_k
        results = results[:request.top_k]
        
        # Get AI analysis
        ai_analysis = await engine.analyze_finding(
            query=request.query,
            matches=results,
            vessel_type=request.vessel_type.value
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Parse AI analysis if it's a JSON string
        analysis_text = ai_analysis.get('analysis', '')
        if analysis_text and analysis_text.strip().startswith('{'):
            try:
                import json
                parsed = json.loads(analysis_text)
                analysis_text = parsed.get('analysis', analysis_text)
            except:
                pass
        
        # Don't rerank for AI analysis - keep all diverse results
        # Convert results to VIQMatch objects
        matches = []
        for i, result in enumerate(results):
            context = result.get('guidance', '') if request.include_context else None
            
            # Add AI analysis to the first result
            if i == 0 and analysis_text:
                if context:
                    context = f"AI Analysis: {analysis_text}\n\nGuidance: {context}"
                else:
                    context = f"AI Analysis: {analysis_text}"
            
            match = VIQMatch(
                viq_number=result['viq_number'],
                question=result['question'],
                vessel_type=result['vessel_type'],
                similarity_score=round(result['similarity_score'], 4),
                confidence_score=round(result['confidence_score'], 4),
                context=context,
                source_file=result.get('source_file', ''),
                metadata=result.get('metadata', {})
            )
            matches.append(match)
        
        return EnhancedQueryResponse(
            matches=matches,
            query=request.query,
            vessel_type=request.vessel_type.value,
            total_results=len(matches),
            search_method="hybrid_with_ai",
            processing_time_ms=round(processing_time, 2),
            enhanced_query=enhanced_query if enhanced_query != request.query else None,
            ai_analysis=analysis_text,
            recommendations=ai_analysis.get('recommendations', []),
            risk_assessment=ai_analysis.get('risk_assessment')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/vessel-types")
async def get_vessel_types(engine: RAGEngine = Depends(get_rag_engine)):
    """Get available vessel types"""
    try:
        vessel_types = await engine.get_vessel_types()
        return {"vessel_types": vessel_types}
    except Exception as e:
        logger.error(f"Error getting vessel types: {str(e)}")
        return {"vessel_types": [vt.value for vt in VesselType]}

@router.get("/search-methods")
async def get_search_methods():
    """Get available search methods"""
    return {
        "search_methods": [
            {
                "value": method.value,
                "name": method.value.title(),
                "description": {
                    "semantic": "Uses AI embeddings for meaning-based search",
                    "keyword": "Traditional keyword-based search",
                    "hybrid": "Combines semantic and keyword search for best results"
                }[method.value]
            }
            for method in SearchMethod
        ]
    }

@router.get("/stats", response_model=SystemStats)
async def get_system_stats(engine: RAGEngine = Depends(get_rag_engine)):
    """Get comprehensive system statistics"""
    try:
        stats = await engine.get_stats()
        
        return SystemStats(
            total_questions=stats['total_questions'],
            vessel_types=stats['vessel_types'],
            collections=stats['collections'],
            status=stats['status'],
            search_methods=stats['search_methods'],
            openai_available=stats['openai_available'],
            vector_store_status=stats['vector_store_status'],
            last_updated=stats['last_updated']
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.post("/reprocess-data")
async def reprocess_data(
    background_tasks: BackgroundTasks,
    processor: PDFProcessor = Depends(get_pdf_processor),
    engine: RAGEngine = Depends(get_rag_engine)
):
    """Reprocess all VIQ data and update vector store"""
    try:
        # Start background processing
        background_tasks.add_task(_reprocess_data_task, processor, engine)
        
        return {
            "message": "Data reprocessing started",
            "status": "processing",
            "estimated_time": "2-5 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error starting reprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start reprocessing: {str(e)}")

async def _reprocess_data_task(processor: PDFProcessor, engine: RAGEngine):
    """Background task for reprocessing data"""
    try:
        logger.info("Starting data reprocessing task")
        
        # Process documents
        documents = await processor.process_all_documents()
        
        if documents:
            # Reset vector store
            await engine.vector_store.reset_collection()
            
            # Re-initialize vector store
            await engine.vector_store.initialize()
            
            # Add documents to vector store
            await engine.vector_store.add_documents(documents)
            
            logger.info(f"Data reprocessing completed: {len(documents)} documents")
        else:
            logger.warning("No documents found during reprocessing")
            
    except Exception as e:
        logger.error(f"Data reprocessing task failed: {str(e)}")

@router.get("/processing-status", response_model=ProcessingStatus)
async def get_processing_status(processor: PDFProcessor = Depends(get_pdf_processor)):
    """Get current data processing status"""
    try:
        status = processor.get_processing_status()
        
        return ProcessingStatus(
            status=status["status"],
            message=status["message"],
            progress=status.get("progress"),
            details=status.get("details", {})
        )
        
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get processing status: {str(e)}")

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        components = {
            "api": "healthy",
            "vector_store": "healthy" if rag_engine and rag_engine.vector_store.is_initialized else "unhealthy",
            "openai": "available" if rag_engine and rag_engine.openai_client else "unavailable",
            "pdf_processor": "ready" if pdf_processor else "not_ready"
        }
        
        overall_status = "healthy" if all(
            status in ["healthy", "available", "ready"] 
            for status in components.values()
        ) else "degraded"
        
        return HealthCheck(
            status=overall_status,
            version=settings.VERSION,
            timestamp=datetime.now().isoformat(),
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheck(
            status="unhealthy",
            version=settings.VERSION,
            timestamp=datetime.now().isoformat(),
            components={"error": str(e)}
        )

@router.get("/version")
async def get_version():
    """Get API version information"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "api_version": settings.API_V1_STR
    }

