"""
Two fully separate SQLite databases — one for VIQ, one for RISQ. Each has
its own engine, its own declarative Base, and its own session factory, so
there is no shared table, no shared metadata, and no way for a query
against one to accidentally touch the other's data.

  VIQ  -> data/viq_system.db  (search_history, user_feedback, system_analytics)
  RISQ -> data/risq_system.db (risq_ingestion_report, risq_query_history, risq_feedback)
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime

# ===== VIQ database =====

VIQBase = declarative_base()


class SearchHistory(VIQBase):
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    result_viq = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    time_ms = Column(Integer, nullable=False)
    vessel_type = Column(String(50), default="All")
    search_method = Column(String(50), default="hybrid")
    timestamp = Column(DateTime, default=datetime.utcnow)


class UserFeedback(VIQBase):
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    suggested_viq = Column(String(50), nullable=False)
    correct_viq = Column(String(50), nullable=True)
    feedback_type = Column(String(20), nullable=False)  # thumbs_up, thumbs_down, correction
    timestamp = Column(DateTime, default=datetime.utcnow)


class SystemAnalytics(VIQBase):
    __tablename__ = "system_analytics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(String(500), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


VIQ_DATABASE_URL = "sqlite:///./data/viq_system.db"
viq_engine = create_engine(VIQ_DATABASE_URL, connect_args={"check_same_thread": False})
VIQSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=viq_engine)


def get_db():
    """VIQ database session — used by every VIQ route."""
    db = VIQSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ===== RISQ database (fully separate file, separate Base, separate engine) =====

RisqBase = declarative_base()


class RisqIngestionReport(RisqBase):
    __tablename__ = "risq_ingestion_report"

    id = Column(Integer, primary_key=True, index=True)
    source_file = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False)  # question_master | historical_finding | pdf
    file_checksum = Column(String(64), nullable=True)
    total_rows = Column(Integer, default=0)
    embedded = Column(Integer, default=0)
    upserted = Column(Integer, default=0)
    matched = Column(Integer, default=0)
    unmatched = Column(Integer, default=0)
    invalid = Column(Integer, default=0)
    status = Column(String(20), default="success")  # success | partial | failed
    error_summary = Column(Text, nullable=True)
    triggered_by = Column(String(100), nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class RisqQueryHistory(RisqBase):
    __tablename__ = "risq_query_history"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    query_type = Column(String(20), nullable=False)  # exact_code | semantic
    result_code = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    time_ms = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class RisqFeedback(RisqBase):
    __tablename__ = "risq_feedback"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    suggested_risq_code = Column(String(50), nullable=False)
    correct_risq_code = Column(String(50), nullable=True)
    feedback_type = Column(String(20), nullable=False)  # thumbs_up, thumbs_down, correction
    timestamp = Column(DateTime, default=datetime.utcnow)


RISQ_DATABASE_URL = "sqlite:///./data/risq_system.db"
risq_engine = create_engine(RISQ_DATABASE_URL, connect_args={"check_same_thread": False})
RisqSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=risq_engine)


def get_risq_db():
    """RISQ database session — used by every RISQ route. Never the same
    engine/file/session as VIQ's get_db()."""
    db = RisqSessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables in BOTH databases (each Base only creates its own
    tables in its own engine — VIQBase never touches risq_system.db and
    vice versa)."""
    VIQBase.metadata.create_all(bind=viq_engine)
    RisqBase.metadata.create_all(bind=risq_engine)
