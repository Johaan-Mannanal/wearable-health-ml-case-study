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
        """Build illustrative, non-diagnostic wellness cues from inputs and model outputs.

        NOTE: All thresholds/labels below are illustrative and operate on SYNTHETIC data. They are
        NOT medical alerts or diagnoses. Titles/messages are worded as "outside the typical range"
        or "synthetic label" cues rather than clinical statements.
        """

        alerts = []

        # Heart-rate cue (illustrative, non-diagnostic)
        if health_metric.heart_rate > 120:
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="warning",
                title="Heart Rate Above Typical Range",
                message=f"Heart rate is above the typical range at {health_metric.heart_rate} bpm"
            ))

        if health_metric.heart_rate < 50:
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="warning",
                title="Heart Rate Below Typical Range",
                message=f"Heart rate is below the typical range at {health_metric.heart_rate} bpm"
            ))

        # Synthetic risk label (not a health assessment)
        if ml_assessment.risk_level == "Critical":
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="critical",
                title="Model Risk Label: Critical (synthetic)",
                message="Model assigned the highest synthetic 'risk' label (not a health assessment)"
            ))
        elif ml_assessment.risk_level == "High":
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="warning",
                title="Model Risk Label: High (synthetic)",
                message="Model assigned a high synthetic 'risk' label (not a health assessment)"
            ))

        # Synthetic rhythm label
        if ml_assessment.rhythm_status == "Irregular":
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="warning",
                title="Model Rhythm Label: Irregular (synthetic)",
                message="Model assigned the 'Irregular' synthetic label (not arrhythmia detection)"
            ))

        # Synthetic HRV pattern label
        if ml_assessment.hrv_pattern == "Atrial Fibrillation":
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="warning",
                title="Model HRV Label: afib-class (synthetic)",
                message="Model assigned the synthetic 'afib' class label (not a diagnosis)"
            ))

        # Low HRV cue (illustrative, non-diagnostic)
        if health_metric.hrv_mean < 20:
            alerts.append(self.create_alert(
                patient_id=patient_id,
                assessment_id=ml_assessment.id,
                alert_type="info",
                title="HRV Below Typical Range",
                message=f"HRV is below the typical range at {health_metric.hrv_mean} ms"
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