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
    
    # ChromaDB Settings
    CHROMA_PERSIST_DIRECTORY: str = "data/vectordb"
    CHROMA_COLLECTION_NAME: str = "viq_questions"
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Text Processing Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CONTENT_LENGTH: int = 8000
    
    # Search Settings
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20
    SIMILARITY_THRESHOLD: float = 0.3
    RERANK_TOP_K: int = 10
    
    # Vessel Types
    VESSEL_TYPES: List[str] = ["All", "Oil", "Chemical", "LPG", "LNG"]
    
    # Data Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PDFS_DIR: Path = DATA_DIR / "pdfs"
    VECTORDB_DIR: Path = DATA_DIR / "vectordb"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()