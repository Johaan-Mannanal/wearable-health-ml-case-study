"""
Pydantic models for health data validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID

class HealthMetrics(BaseModel):
    heart_rate: float = Field(..., ge=30, le=250, description="Heart rate in BPM")
    heart_rate_std: float = Field(..., ge=0, le=100, description="Heart rate standard deviation")
    hrv_mean: float = Field(..., ge=0, le=200, description="Mean HRV in milliseconds")
    pnn50: float = Field(..., ge=0, le=1, description="pNN50 percentage")
    respiratory_rate: float = Field(..., ge=8, le=30, description="Respiratory rate")
    activity_level: float = Field(..., ge=0, le=1000, description="Activity level in kcal")
    sleep_quality: float = Field(..., ge=0, le=1, description="Sleep quality score")
    
    @validator('heart_rate')
    def validate_heart_rate(cls, v):
        if v < 30 or v > 250:
            raise ValueError('Heart rate must be between 30 and 250 BPM')
        return v

class DeviceInfo(BaseModel):
    model: str = Field(..., description="Device model (e.g., Apple Watch Series 10)")
    os_version: str = Field(..., description="Operating system version")

class HealthDataSubmission(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier from iOS app")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: HealthMetrics
    device_info: Optional[DeviceInfo] = None
    age: Optional[int] = Field(None, ge=18, le=120)

class RhythmPrediction(BaseModel):
    prediction: str
    confidence: float = Field(..., ge=0, le=1)

class RiskPrediction(BaseModel):
    level: str
    score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)

class HRVPattern(BaseModel):
    pattern: str
    confidence: float = Field(..., ge=0, le=1)

class FitnessLevel(BaseModel):
    vo2max: float = Field(..., ge=10, le=80)
    cardiovascular_age: int = Field(..., ge=20, le=100)
    fitness_category: str

class MLPredictions(BaseModel):
    rhythm_status: RhythmPrediction
    health_risk: RiskPrediction
    hrv_pattern: HRVPattern
    fitness_level: FitnessLevel

class Alert(BaseModel):
    id: str
    type: str  # critical, warning, info
    message: str

class HealthAssessmentResponse(BaseModel):
    assessment_id: str
    predictions: MLPredictions
    alerts: List[Alert] = []
    recommendations: List[str] = []

class PatientData(BaseModel):
    id: str
    name: str
    age: int
    last_update: datetime
    status: str  # stable, warning, critical, offline
    current_metrics: Optional[HealthMetrics] = None
    alerts: List[Alert] = []

class HistoricalDataPoint(BaseModel):
    timestamp: datetime
    heart_rate: float
    risk_score: float
    fitness_level: str

class PatientHistory(BaseModel):
    patient_id: str
    timeframe: str
    data_points: List[HistoricalDataPoint]
    statistics: Dict[str, Any]