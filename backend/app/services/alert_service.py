"""
Alert service for health monitoring
"""

from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import uuid

from app.database import Alert, Patient, HealthMetric, MLAssessment

class AlertService:
    def __init__(self, db: Session):
        self.db = db
    
    def check_for_alerts(
        self,
        patient_id: uuid.UUID,
        health_metric: HealthMetric,
        ml_assessment: MLAssessment
    ) -> List[Alert]:
        """Check health data and ML assessment for alert conditions"""
        
        alerts = []
        
        # Critical heart rate
        if health_metric.heart_rate > 120:
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="warning",
                title="High Heart Rate",
                message=f"Heart rate is elevated at {health_metric.heart_rate} bpm"
            ))
        
        if health_metric.heart_rate < 50:
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="warning",
                title="Low Heart Rate",
                message=f"Heart rate is low at {health_metric.heart_rate} bpm"
            ))
        
        # Critical risk level
        if ml_assessment.risk_level == "Critical":
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="critical",
                title="Critical Health Risk",
                message="ML models detected critical health risk level"
            ))
        elif ml_assessment.risk_level == "High":
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="warning",
                title="High Health Risk",
                message="ML models detected high health risk level"
            ))
        
        # Irregular rhythm
        if ml_assessment.rhythm_status == "Irregular":
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="warning",
                title="Irregular Heart Rhythm",
                message="Irregular heart rhythm pattern detected"
            ))
        
        # Atrial fibrillation pattern
        if ml_assessment.hrv_pattern == "Atrial Fibrillation":
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="critical",
                title="Possible Atrial Fibrillation",
                message="HRV pattern suggests possible atrial fibrillation"
            ))
        
        # Low HRV
        if health_metric.hrv_mean < 20:
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="info",
                title="Low Heart Rate Variability",
                message=f"HRV is low at {health_metric.hrv_mean} ms"
            ))
        
        # Save all alerts to database
        for alert in alerts:
            self.db.add(alert)
        
        return alerts
    
    def create_alert(
        self,
        patient_id: uuid.UUID,
        assessment_id: uuid.UUID,
        alert_type: str,
        title: str,
        message: str
    ) -> Alert:
        """Create a new alert"""
        
        return Alert(
            patient_id=patient_id,
            assessment_id=assessment_id,
            alert_type=alert_type,
            title=title,
            message=message,
            created_at=datetime.utcnow()
        )