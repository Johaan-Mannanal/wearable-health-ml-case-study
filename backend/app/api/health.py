"""
Health data API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from app.database import get_db, Patient, HealthMetric, MLAssessment, Alert
from app.models.health_data import HealthDataSubmission, HealthAssessmentResponse
from app.ml.inference import InferenceEngine
from app.services.alert_service import AlertService

router = APIRouter()

@router.post("/data", response_model=HealthAssessmentResponse)
async def submit_health_data(
    data: HealthDataSubmission,
    request: Request,
    db: Session = Depends(get_db)
):
    """Submit health data from iOS app and get ML assessment"""
    
    # Get model manager from app state
    model_manager = request.app.state.model_manager
    if not model_manager:
        raise HTTPException(status_code=503, detail="ML models not loaded")
    
    # Get or create patient
    patient = db.query(Patient).filter(Patient.external_id == data.patient_id).first()
    if not patient:
        patient = Patient(
            external_id=data.patient_id,
            age=data.age if hasattr(data, 'age') else None,
            patient_metadata=data.device_info.dict() if data.device_info else {}
        )
        db.add(patient)
        db.commit()
    
    # Store health metrics
    health_metric = HealthMetric(
        patient_id=patient.id,
        timestamp=data.timestamp,
        heart_rate=data.metrics.heart_rate,
        heart_rate_std=data.metrics.heart_rate_std,
        hrv_mean=data.metrics.hrv_mean,
        pnn50=data.metrics.pnn50,
        respiratory_rate=data.metrics.respiratory_rate,
        activity_level=data.metrics.activity_level,
        sleep_quality=data.metrics.sleep_quality,
        raw_data=data.metrics.dict()
    )
    db.add(health_metric)
    db.commit()
    
    # Run ML inference
    inference_engine = InferenceEngine(model_manager)
    ml_results = inference_engine.run_inference(data.metrics.dict())
    
    # Store ML assessment
    ml_assessment = MLAssessment(
        patient_id=patient.id,
        metric_id=health_metric.id,
        timestamp=data.timestamp,
        rhythm_status=ml_results.get('rhythm', {}).get('prediction'),
        rhythm_confidence=ml_results.get('rhythm', {}).get('confidence'),
        risk_level=ml_results.get('risk', {}).get('level'),
        risk_score=ml_results.get('risk', {}).get('score'),
        risk_confidence=ml_results.get('risk', {}).get('confidence'),
        hrv_pattern=ml_results.get('hrv', {}).get('pattern'),
        pattern_confidence=ml_results.get('hrv', {}).get('confidence'),
        vo2max=ml_results.get('fitness', {}).get('vo2max'),
        cardiovascular_age=ml_results.get('fitness', {}).get('cardiovascular_age'),
        fitness_category=ml_results.get('fitness', {}).get('fitness_category'),
        processing_time_ms=ml_results.get('processing_time_ms'),
        model_versions=ml_results.get('model_versions')
    )
    db.add(ml_assessment)
    
    # Check for alerts
    alert_service = AlertService(db)
    alerts = alert_service.check_for_alerts(patient.id, health_metric, ml_assessment)
    
    # Commit all changes
    db.commit()
    
    # Prepare response
    response = HealthAssessmentResponse(
        assessment_id=str(ml_assessment.id),
        predictions={
            "rhythm_status": {
                "prediction": ml_assessment.rhythm_status or "Unknown",
                "confidence": ml_assessment.rhythm_confidence or 0.0
            },
            "health_risk": {
                "level": ml_assessment.risk_level or "Unknown",
                "score": ml_assessment.risk_score or 0.0,
                "confidence": ml_assessment.risk_confidence or 0.0
            },
            "hrv_pattern": {
                "pattern": ml_assessment.hrv_pattern or "Unknown",
                "confidence": ml_assessment.pattern_confidence or 0.0
            },
            "fitness_level": {
                "vo2max": ml_assessment.vo2max or 0.0,
                "cardiovascular_age": ml_assessment.cardiovascular_age or 0,
                "fitness_category": ml_assessment.fitness_category or "Unknown"
            }
        },
        alerts=[{"id": str(alert.id), "type": alert.alert_type, "message": alert.message} for alert in alerts],
        recommendations=generate_recommendations(ml_assessment)
    )
    
    return response

def generate_recommendations(assessment: MLAssessment) -> list:
    """Build generic, non-diagnostic wellness tips from the model's synthetic labels.

    NOTE: These are illustrative wellness suggestions on SYNTHETIC-model output, not medical advice.
    """
    recommendations = []

    if assessment.risk_level == "High" or assessment.risk_level == "Critical":
        recommendations.append("This is a synthetic 'risk' label, not a health assessment")
        recommendations.append("For any real health concern, talk to a qualified clinician")

    if assessment.fitness_category == "Poor":
        recommendations.append("Gradually increase physical activity")
        recommendations.append("Consider starting a walking routine")
    elif assessment.fitness_category == "Fair":
        recommendations.append("Maintain current activity level")
        recommendations.append("Consider adding variety to your exercise routine")
    else:
        recommendations.append("Continue your current fitness routine")

    if assessment.hrv_pattern == "Atrial Fibrillation":
        recommendations.append("The model assigned a synthetic 'afib' class label (not a diagnosis)")

    return recommendations

@router.get("/patient/{patient_id}/latest")
async def get_latest_health_data(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Get the latest health data for a patient"""
    
    patient = db.query(Patient).filter(Patient.external_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    latest_metric = db.query(HealthMetric).filter(
        HealthMetric.patient_id == patient.id
    ).order_by(HealthMetric.timestamp.desc()).first()
    
    if not latest_metric:
        raise HTTPException(status_code=404, detail="No health data found")
    
    latest_assessment = db.query(MLAssessment).filter(
        MLAssessment.metric_id == latest_metric.id
    ).first()
    
    return {
        "patient_id": patient_id,
        "timestamp": latest_metric.timestamp,
        "metrics": {
            "heart_rate": latest_metric.heart_rate,
            "hrv_mean": latest_metric.hrv_mean,
            "respiratory_rate": latest_metric.respiratory_rate,
            "activity_level": latest_metric.activity_level,
            "sleep_quality": latest_metric.sleep_quality
        },
        "assessment": {
            "rhythm_status": latest_assessment.rhythm_status if latest_assessment else None,
            "risk_level": latest_assessment.risk_level if latest_assessment else None,
            "fitness_category": latest_assessment.fitness_category if latest_assessment else None
        }
    }