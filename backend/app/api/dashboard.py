"""
Dashboard API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime, timedelta

from app.database import get_db, Patient, HealthMetric, MLAssessment, Alert
from app.models.health_data import PatientData, PatientHistory

router = APIRouter()

@router.get("/patients", response_model=List[PatientData])
async def get_patients(
    db: Session = Depends(get_db),
    limit: int = Query(10, ge=1, le=100)
):
    """Get list of patients for dashboard display"""
    
    patients = db.query(Patient).limit(limit).all()
    
    patient_data = []
    for patient in patients:
        # Get latest health metric
        latest_metric = db.query(HealthMetric).filter(
            HealthMetric.patient_id == patient.id
        ).order_by(desc(HealthMetric.timestamp)).first()
        
        # Get latest assessment
        latest_assessment = None
        if latest_metric:
            latest_assessment = db.query(MLAssessment).filter(
                MLAssessment.metric_id == latest_metric.id
            ).first()
        
        # Get unacknowledged alerts
        alerts = db.query(Alert).filter(
            Alert.patient_id == patient.id,
            Alert.is_acknowledged == False
        ).order_by(desc(Alert.created_at)).limit(5).all()
        
        # Determine patient status
        status = "offline"
        if latest_metric:
            time_diff = datetime.utcnow() - latest_metric.timestamp
            if time_diff < timedelta(minutes=5):
                status = "stable"
                if latest_assessment:
                    if latest_assessment.risk_level == "Critical":
                        status = "critical"
                    elif latest_assessment.risk_level == "High":
                        status = "warning"
        
        patient_data.append(PatientData(
            id=str(patient.external_id),
            name=f"Patient {patient.external_id[:8]}",  # Anonymized name
            age=patient.age or 40,
            last_update=latest_metric.timestamp if latest_metric else patient.created_at,
            status=status,
            current_metrics={
                "heart_rate": latest_metric.heart_rate if latest_metric else 0,
                "hrv_mean": latest_metric.hrv_mean if latest_metric else 0,
                "respiratory_rate": latest_metric.respiratory_rate if latest_metric else 0,
                "activity_level": latest_metric.activity_level if latest_metric else 0,
                "sleep_quality": latest_metric.sleep_quality if latest_metric else 0,
                "risk_level": latest_assessment.risk_level if latest_assessment else "Unknown",
                "fitness_category": latest_assessment.fitness_category if latest_assessment else "Unknown"
            } if latest_metric else None,
            alerts=[{
                "id": str(alert.id),
                "type": alert.alert_type,
                "message": alert.message
            } for alert in alerts]
        ))
    
    return patient_data

@router.get("/patient/{patient_id}/history")
async def get_patient_history(
    patient_id: str,
    db: Session = Depends(get_db),
    timeframe: str = Query("24h", regex="^(1h|6h|24h|7d|30d)$")
):
    """Get historical data for a specific patient"""
    
    # Get patient
    patient = db.query(Patient).filter(Patient.external_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Calculate time range
    time_ranges = {
        "1h": timedelta(hours=1),
        "6h": timedelta(hours=6),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30)
    }
    
    start_time = datetime.utcnow() - time_ranges[timeframe]
    
    # Get health metrics and assessments
    metrics = db.query(HealthMetric, MLAssessment).join(
        MLAssessment, HealthMetric.id == MLAssessment.metric_id
    ).filter(
        HealthMetric.patient_id == patient.id,
        HealthMetric.timestamp >= start_time
    ).order_by(HealthMetric.timestamp).all()
    
    # Format data points
    data_points = []
    heart_rates = []
    risk_scores = []
    
    for metric, assessment in metrics:
        data_points.append({
            "timestamp": metric.timestamp,
            "heart_rate": metric.heart_rate,
            "risk_score": assessment.risk_score if assessment else 0,
            "fitness_level": assessment.fitness_category if assessment else "Unknown"
        })
        heart_rates.append(metric.heart_rate)
        risk_scores.append(assessment.risk_score if assessment else 0)
    
    # Calculate statistics
    statistics = {}
    if heart_rates:
        statistics["avg_heart_rate"] = sum(heart_rates) / len(heart_rates)
        statistics["min_heart_rate"] = min(heart_rates)
        statistics["max_heart_rate"] = max(heart_rates)
        statistics["avg_risk_score"] = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        
        # Simple trend calculation
        if len(heart_rates) > 1:
            recent_avg = sum(heart_rates[-5:]) / len(heart_rates[-5:])
            older_avg = sum(heart_rates[:5]) / len(heart_rates[:5])
            if recent_avg > older_avg * 1.1:
                statistics["trend"] = "increasing"
            elif recent_avg < older_avg * 0.9:
                statistics["trend"] = "decreasing"
            else:
                statistics["trend"] = "stable"
        else:
            statistics["trend"] = "insufficient_data"
    
    return PatientHistory(
        patient_id=patient_id,
        timeframe=timeframe,
        data_points=data_points,
        statistics=statistics
    )

@router.post("/alert/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    db: Session = Depends(get_db),
    acknowledged_by: str = "dashboard_user"
):
    """Acknowledge an alert"""
    
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.is_acknowledged = True
    alert.acknowledged_by = acknowledged_by
    alert.acknowledged_at = datetime.utcnow()
    
    db.commit()
    
    return {"status": "acknowledged", "alert_id": alert_id}