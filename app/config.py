from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List, Optional
import os

class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "VIQ RAG System"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "AI-powered VIQ question matching system using RAG architecture"
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_MAX_TOKENS: int = 1000
    OPENAI_TEMPERATURE: float = 0.1
    
    # ChromaDB Settings (VIQ and RISQ each have their own physically
    # separate vector store directory — see app/state.py / app/main.py)
    CHROMA_PERSIST_DIRECTORY: str = "data/viq_vectordb"
    CHROMA_COLLECTION_NAME: str = "viq_questions"
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Text Processing Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CONTENT_LENGTH: int = 8000
    
    # Search Settings
    DEFAULT_TOP_K: int = 3
    MAX_TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.3
    RERANK_TOP_K: int = 3
    
    # Vessel Types
    VESSEL_TYPES: List[str] = ["All", "Oil", "Chemical", "LPG", "LNG"]
    
    # Data Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    VIQ_VECTORDB_DIR: Path = DATA_DIR / "viq_vectordb"
    RISQ_VECTORDB_DIR: Path = DATA_DIR / "risq_vectordb"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30

    # ===== RISQ MODULE SETTINGS (additive, isolated from VIQ) =====
    RISQ_DATA_DIR: Path = Path(__file__).parent.parent / "data" / "risqData"
    RISQ_PDF_DIR: Path = RISQ_DATA_DIR / "pdfs"
    RISQ_QUESTIONS_FILE: str = "Risq 3.2 R2 - 25.02.26 updated 2.xlsx"
    RISQ_QUESTIONS_SHEET: str = "Risq 3.2 R1"
    RISQ_TRAINING_FILE: str = "RISQ training data.xlsx"
    RISQ_TRAINING_SHEET: str = "ag-grid"
    RISQ_CHROMA_QUESTIONS_COLLECTION: str = "risq_questions"
    RISQ_CHROMA_TRAINING_COLLECTION: str = "risq_training"
    RISQ_CHROMA_DOCUMENTS_COLLECTION: str = "risq_documents"
    RISQ_SIMILARITY_THRESHOLD: float = 0.3
    RISQ_MAX_UPLOAD_MB: int = 25
    RISQ_DEFAULT_TOP_K: int = 3

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()