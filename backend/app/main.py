from contextlib import asynccontextmanager
import logging
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import admin, feedback, search
from app.core import rag_engine
from app.database import create_tables

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting VIQ RAG System")

    try:
        create_tables()
        logger.info("Database initialized")

        await rag_engine.initialize_system()
        logger.info("System initialization completed successfully")
        yield
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down VIQ RAG System")


# Create FastAPI app with lifespan
app = FastAPI(
    title="VIQ RAG System",
    description="AI-powered system for matching ship audit findings to VIQ questions",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)


# Custom middleware for request logging
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Custom middleware for request processing"""
    start_time = datetime.now()

    logger.info(f"Request: {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Response: {response.status_code} ({processing_time:.2f}ms)")
        return response

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Request failed after {processing_time:.2f}ms: {str(e)}")

        return JSONResponse(
            status_code=500,
            content={"error": "REQUEST_FAILED", "message": "An internal server error occurred"}
        )


# Routers
app.include_router(search.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": "VIQ RAG System",
        "version": "1.0.0",
        "description": "AI-powered system for matching ship audit findings to VIQ questions",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        components_status = {
            "vector_store": "healthy" if rag_engine.collection else "unhealthy",
            "openai_client": "healthy" if rag_engine.client else "unhealthy"
        }

        overall_status = "healthy" if rag_engine.collection and rag_engine.client else "degraded"

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": components_status
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "NOT_FOUND", "message": f"Endpoint not found: {request.url.path}"}
    )


@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "VALIDATION_ERROR", "message": "Request validation failed"}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": "An internal server error occurred"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
