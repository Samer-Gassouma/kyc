"""SQLAlchemy models and session factory."""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import Column, DateTime, Float, String, Text, Boolean, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from core.config import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Capture(Base):
    __tablename__ = "captures"

    id = Column(String, primary_key=True, default=lambda: f"cap_{uuid.uuid4().hex[:12]}")
    side = Column(String, nullable=False)  # "front" | "back"
    s3_key = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending | processing | completed | failed
    validation_passed = Column(Boolean, default=False)
    confidence = Column(Float, default=0.0)
    card_type_detected = Column(String, nullable=True)
    rejection_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class KYCResult(Base):
    __tablename__ = "kyc_results"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    capture_id = Column(String, nullable=False)
    side = Column(String, nullable=False)
    mrz_raw = Column(Text, nullable=True)
    mrz_parsed = Column(Text, nullable=True)  # JSON string
    ocr_fields = Column(Text, nullable=True)  # JSON string
    mrz_check_digits_valid = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ExtractionSession(Base):
    __tablename__ = "extraction_sessions"

    id = Column(String, primary_key=True, default=lambda: f"ext_{uuid.uuid4().hex[:12]}")
    front_capture_id = Column(String, nullable=True)
    back_capture_id = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending | processing | completed | failed
    merged_fields = Column(Text, nullable=True)  # JSON string
    error_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
