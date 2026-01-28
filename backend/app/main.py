from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import logging
from datetime import datetime

from app.config import settings
from app.core.pdf_processor import PDFProcessor
from app.core.vector_store import VectorStore
from app.core.rag_engine import RAGEngine
from app.api.routes import search
from app.api.routes import feedback
from app.api.routes import admin
from app.utils.helpers import setup_logging, ensure_directories, create_error_response
from app.models.schemas import ErrorResponse

# Setup logging first
setup_logging()
logger = logging.getLogger(__name__)

# Global components
vector_store: VectorStore = None
rag_engine: RAGEngine = None
pdf_processor: PDFProcessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    
    try:
        await initialize_system()
        logger.info("System initialization completed successfully")
        yield
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down VIQ RAG System")
        await cleanup_system()

async def initialize_system():
    """Initialize all system components"""
    global vector_store, rag_engine, pdf_processor
    
    try:
        # Ensure directories exist
        ensure_directories([
            settings.DATA_DIR,
            settings.PDFS_DIR,
            settings.VECTORDB_DIR,
            settings.BASE_DIR / "logs"
        ])
        
        # Initialize PDF processor
        logger.info("Initializing PDF processor...")
        pdf_processor = PDFProcessor()
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore()
        
        if not await vector_store.initialize():
            raise RuntimeError("Failed to initialize vector store")
        
        # Check if we need to process documents
        processed_docs = await pdf_processor.load_processed_documents()
        
        if not processed_docs:
            logger.info("No processed documents found, processing data files...")
            processed_docs = await pdf_processor.process_all_documents()
        
        # Add documents to vector store if collection is empty
        collection_stats = await vector_store.get_collection_stats()
        if collection_stats.get('total_documents', 0) == 0 and processed_docs:
            logger.info("Adding documents to vector store...")
            await vector_store.add_documents(processed_docs)
        
        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        rag_engine = RAGEngine(vector_store)
        
        # Set dependencies in routes
        search.set_dependencies(rag_engine, pdf_processor)
        admin.set_dependencies(rag_engine)
        
        # Log system stats
        stats = await rag_engine.get_stats()
        logger.info(f"System ready - {stats['total_questions']} VIQ questions loaded")
        
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise

async def cleanup_system():
    """Cleanup system resources"""
    try:
        # Add any cleanup logic here
        logger.info("System cleanup completed")
    except Exception as e:
        logger.error(f"System cleanup failed: {str(e)}")

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Custom middleware for request logging and error handling
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Custom middleware for request processing"""
    start_time = datetime.now()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        
        # Log response
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Response: {response.status_code} ({processing_time:.2f}ms)")
        
        return response
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Request failed after {processing_time:.2f}ms: {str(e)}")
        
        # Return error response
        error_response = create_error_response(
            error="REQUEST_FAILED",
            message="Request processing failed",
            details={
                "path": str(request.url.path),
                "method": request.method,
                "processing_time_ms": round(processing_time, 2)
            }
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )

# Include API routes
app.include_router(
    search.router,
    prefix=settings.API_V1_STR,
    tags=["search"]
)

app.include_router(
    feedback.router,
    prefix=settings.API_V1_STR,
    tags=["feedback"]
)

app.include_router(
    admin.router,
    prefix=settings.API_V1_STR,
    tags=["admin"]
)

# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs_url": "/docs",
        "api_url": settings.API_V1_STR
    }

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    try:
        # Check system components
        components_status = {
            "vector_store": "healthy" if vector_store and vector_store.is_initialized else "unhealthy",
            "rag_engine": "healthy" if rag_engine else "unhealthy",
            "pdf_processor": "healthy" if pdf_processor else "unhealthy"
        }
        
        overall_status = "healthy" if all(
            status == "healthy" for status in components_status.values()
        ) else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "version": settings.VERSION,
            "components": components_status
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

# Global exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content=create_error_response(
            error="NOT_FOUND",
            message=f"Endpoint not found: {request.url.path}",
            details={"path": str(request.url.path), "method": request.method}
        ).dict()
    )

@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content=create_error_response(
            error="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": exc.errors() if hasattr(exc, 'errors') else str(exc)}
        ).dict()
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            error="INTERNAL_ERROR",
            message="An internal server error occurred",
            details={"path": str(request.url.path)}
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )