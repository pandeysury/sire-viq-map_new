"""
VIQ + RISQ RAG System — application entry point.

This file only wires things together: creates the FastAPI app, sets up
middleware, runs startup/shutdown, and mounts the routers. The actual
VIQ and RISQ business logic lives in app/services/, and the shared
OpenAI/Chroma client handles live in app/state.py (see that module's
docstring for why they're centralized there).
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app import state
from app.database import create_tables
from app.services import viq_engine, risq_engine
from app.api.routes import viq as viq_routes
from app.api.routes import risq as risq_routes
from app.api.routes import admin as admin_routes

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting VIQ RAG System")
    try:
        await initialize_system()
        logger.info("System initialization completed successfully")
        yield
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise
    finally:
        logger.info("Shutting down VIQ RAG System")


async def initialize_system():
    """Initialize all system components"""
    try:
        create_tables()
        logger.info("Database initialized")

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        from openai import OpenAI
        state.client = OpenAI(api_key=openai_api_key)

        logger.info("Initializing ChromaDB (separate VIQ and RISQ stores)...")
        # VIQ and RISQ each get their own physically separate ChromaDB
        # directory — not just separate collections inside one shared
        # store. Two independent PersistentClient instances, two
        # independent on-disk trees; nothing to accidentally cross over.
        state.viq_chroma_client = chromadb.PersistentClient(path="./data/viq_vectordb")
        state.risq_chroma_client = chromadb.PersistentClient(path="./data/risq_vectordb")

        # Load VIQ data and create embeddings
        await viq_engine.load_viq_data()

        # Load RISQ collections (additive; failures here must never break
        # VIQ startup — see risq_engine.load_risq_collections()).
        await risq_engine.load_risq_collections()

        logger.info("System initialized successfully!")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


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


# Routers — VIQ, RISQ and admin/dashboard endpoints, fully isolated from
# each other at the module level (see app/api/routes/*.py).
app.include_router(viq_routes.router)
app.include_router(risq_routes.router)
app.include_router(admin_routes.router)


# Exception handlers (must be registered on the app instance, not a router)
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
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
