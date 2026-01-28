import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from app.models.schemas import ErrorResponse
from app.config import settings

def setup_logging():
    """Setup comprehensive logging configuration"""
    try:
        from loguru import logger as loguru_logger
        
        # Remove default handler
        loguru_logger.remove()
        
        # Add console handler with rich formatting
        loguru_logger.add(
            sink=lambda msg: print(msg, end=""),
            format=settings.LOG_FORMAT,
            level=settings.LOG_LEVEL,
            colorize=True
        )
        
        # Add file handler
        log_file = settings.BASE_DIR / "logs" / "viq_rag.log"
        log_file.parent.mkdir(exist_ok=True)
        
        loguru_logger.add(
            sink=str(log_file),
            format=settings.LOG_FORMAT,
            level=settings.LOG_LEVEL,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
        
        # Intercept standard logging
        class InterceptHandler(logging.Handler):
            def emit(self, record):
                try:
                    level = loguru_logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno
                
                frame, depth = logging.currentframe(), 2
                while frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1
                
                loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                    level, record.getMessage()
                )
        
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        
    except ImportError:
        # Fallback to standard logging if loguru is not available
        logging.basicConfig(
            level=getattr(logging, settings.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(settings.BASE_DIR / "logs" / "viq_rag.log")
            ]
        )

def ensure_directories(paths: list):
    """Ensure directories exist"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def validate_query(query: str) -> bool:
    """Validate search query with comprehensive checks"""
    if not query or not isinstance(query, str):
        return False
    
    query = query.strip()
    
    # Check length - increased to 5000 for long findings
    if len(query) < 3 or len(query) > 5000:
        return False
    
    # Only check for meaningful content
    if not any(c.isalnum() for c in query):
        return False
    
    return True

def create_error_response(
    error: str, 
    message: str, 
    details: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
    """Create standardized error response"""
    return ErrorResponse(
        error=error,
        message=message,
        details=details or {},
        timestamp=datetime.now().isoformat()
    )

def sanitize_text(text: str, max_length: int = 1000) -> str:
    """Sanitize and truncate text content"""
    if not text:
        return ""
    
    # Remove control characters and normalize whitespace
    import re
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length-3] + "..."
    
    return text.strip()

def extract_keywords(text: str, max_keywords: int = 10) -> list:
    """Extract keywords from text for search enhancement"""
    import re
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
        'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'shall'
    }
    
    keywords = [word for word in words if word not in stop_words]
    
    # Count frequency and return most common
    from collections import Counter
    word_counts = Counter(keywords)
    
    return [word for word, count in word_counts.most_common(max_keywords)]

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using Jaccard index"""
    if not text1 or not text2:
        return 0.0
    
    # Convert to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def format_processing_time(milliseconds: float) -> str:
    """Format processing time in human-readable format"""
    if milliseconds < 1000:
        return f"{milliseconds:.0f}ms"
    elif milliseconds < 60000:
        return f"{milliseconds/1000:.1f}s"
    else:
        minutes = int(milliseconds // 60000)
        seconds = (milliseconds % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks for processing"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

async def rate_limit(calls_per_second: float = 10.0):
    """Simple rate limiting for API calls"""
    await asyncio.sleep(1.0 / calls_per_second)

def validate_vessel_type(vessel_type: str) -> bool:
    """Validate vessel type against allowed values"""
    return vessel_type in [vt.value for vt in settings.VESSEL_TYPES] + ["All"]

def normalize_viq_number(viq_number: str) -> str:
    """Normalize VIQ number format"""
    import re
    
    # Extract numbers and dots
    normalized = re.sub(r'[^\d\.]', '', viq_number)
    
    # Ensure proper format (e.g., "1.2.3")
    parts = normalized.split('.')
    if len(parts) >= 3:
        return '.'.join(parts[:3])
    
    return normalized

def get_file_hash(file_path: Path) -> str:
    """Get MD5 hash of file for change detection"""
    import hashlib
    
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""

def measure_performance(func):
    """Decorator to measure function performance"""
    import functools
    import time
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            logging.getLogger(func.__module__).debug(
                f"{func.__name__} executed in {execution_time:.2f}ms"
            )
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logging.getLogger(func.__module__).error(
                f"{func.__name__} failed after {execution_time:.2f}ms: {str(e)}"
            )
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            logging.getLogger(func.__module__).debug(
                f"{func.__name__} executed in {execution_time:.2f}ms"
            )
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logging.getLogger(func.__module__).error(
                f"{func.__name__} failed after {execution_time:.2f}ms: {str(e)}"
            )
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper