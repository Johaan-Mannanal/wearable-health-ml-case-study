"""
Database configuration and models for TelemetryHealthCare
"""

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean, JSON, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid

from app.config import settings

# Create database engine
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Database Models

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String(255), unique=True, nullable=False, index=True)
    age = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)
    
    # Relationships
    health_metrics = relationship("HealthMetric", back_populates="patient", cascade="all, delete-orphan")
    ml_assessments = relationship("MLAssessment", back_populates="patient", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="patient", cascade="all, delete-orphan")

class HealthMetric(Base):
    __tablename__ = "health_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    heart_rate = Column(Float)
    heart_rate_std = Column(Float)
    hrv_mean = Column(Float)
    pnn50 = Column(Float)
    respiratory_rate = Column(Float)
    activity_level = Column(Float)
    sleep_quality = Column(Float)
    raw_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="health_metrics")
    ml_assessment = relationship("MLAssessment", back_populates="health_metric", uselist=False)

class MLAssessment(Base):
    __tablename__ = "ml_assessments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    metric_id = Column(UUID(as_uuid=True), ForeignKey("health_metrics.id"))
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # ML Results
    rhythm_status = Column(String(50))
    rhythm_confidence = Column(Float)
    risk_level = Column(String(50))
    risk_score = Column(Float)
    risk_confidence = Column(Float)
    hrv_pattern = Column(String(50))
    pattern_confidence = Column(Float)
    vo2max = Column(Float)
    cardiovascular_age = Column(Integer)
    fitness_category = Column(String(50))
    
    # Metadata
    processing_time_ms = Column(Integer)
    model_versions = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="ml_assessments")
    health_metric = relationship("HealthMetric", back_populates="ml_assessment")

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("ml_assessments.id"))
    alert_type = Column(String(50), nullable=False)  # critical, warning, info
    title = Column(String(255), nullable=False)
    message = Column(Text)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(255))
    acknowledged_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    patient = relationship("Patient", back_populates="alerts")

class DashboardUser(Base):
    __tablename__ = "dashboard_users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="viewer")  # admin, clinician, viewer
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

class AuditLog(Base):
    __tablename__ = "audit_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("dashboard_users.id"))
    action = Column(String(255), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(UUID(as_uuid=True))
    details = Column(JSON)
    ip_address = Column(String(45))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

# Database dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()