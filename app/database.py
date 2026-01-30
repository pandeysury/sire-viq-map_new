from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime

Base = declarative_base()

class SearchHistory(Base):
    __tablename__ = "search_history"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    result_viq = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    time_ms = Column(Integer, nullable=False)
    vessel_type = Column(String(50), default="All")
    search_method = Column(String(50), default="hybrid")
    timestamp = Column(DateTime, default=datetime.utcnow)

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    suggested_viq = Column(String(50), nullable=False)
    correct_viq = Column(String(50), nullable=True)
    feedback_type = Column(String(20), nullable=False)  # thumbs_up, thumbs_down, correction
    timestamp = Column(DateTime, default=datetime.utcnow)

class SystemAnalytics(Base):
    __tablename__ = "system_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(String(500), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Database setup
DATABASE_URL = "sqlite:///./data/viq_system.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()