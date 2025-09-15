"""
ML Inference engine for running health predictions
"""

import numpy as np
from typing import Dict, Any
import time
import logging

from app.ml.model_loader import ModelManager

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def validate_inputs(self, features: Dict[str, float]) -> Dict[str, float]:
        """Validate and clamp inputs to physiological bounds"""
        validated = features.copy()
        
        # Physiological bounds
        bounds = {
            'heart_rate': (30, 250),
            'heart_rate_std': (0, 100),
            'hrv_mean': (0, 200),
            'pnn50': (0, 1),
            'respiratory_rate': (8, 30),
            'activity_level': (0, 1000),
            'sleep_quality': (0, 1)
        }
        
        for key, (min_val, max_val) in bounds.items():
            if key in validated:
                validated[key] = max(min_val, min(max_val, validated[key]))
        
        return validated
    
    def prepare_svm_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for SVM heart rhythm model"""
        return np.array([
            features['heart_rate'],
            features['heart_rate_std'],
            features['pnn50']
        ]).reshape(1, -1)
    
    def prepare_gbm_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for GBM risk assessment model"""
        # Calculate derived features
        stress_indicator = features['heart_rate'] / (features['hrv_mean'] + 1)
        recovery_score = features['sleep_quality'] * features['hrv_mean']
        hr_hrv_ratio = features['heart_rate'] / (features['hrv_mean'] + 1)
        
        return np.array([
            features['heart_rate'],
            features['hrv_mean'],
            features['respiratory_rate'],
            features['activity_level'],
            features['sleep_quality'],
            stress_indicator,
            hr_hrv_ratio,
            recovery_score
        ]).reshape(1, -1)
    
    def prepare_nn_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for Neural Network HRV model"""
        # Extended HRV features
        return np.array([
            features['hrv_mean'],
            features['heart_rate'],
            features['heart_rate_std'],
            features['pnn50'],
            features['respiratory_rate'],
            features['activity_level'],
            features['sleep_quality'],
            # Add derived HRV metrics
            features['hrv_mean'] / features['heart_rate'] if features['heart_rate'] > 0 else 0,
            features['pnn50'] * 100,  # Convert to percentage
            features['heart_rate_std'] / features['heart_rate'] if features['heart_rate'] > 0 else 0,
            # Placeholder for frequency domain features (would be calculated from raw RR intervals)
            0.3,  # LF power placeholder
            0.4,  # HF power placeholder
            0.75  # LF/HF ratio placeholder
        ]).reshape(1, -1)
    
    def prepare_rf_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for Random Forest fitness model"""
        # Comprehensive fitness features
        return np.array([
            features['heart_rate'],
            features['hrv_mean'],
            features['activity_level'],
            features['sleep_quality'],
            features['respiratory_rate'],
            features['heart_rate_std'],
            features['pnn50'],
            # Fitness-specific derived features
            features['heart_rate'] / 220 * 100,  # % of max HR (assuming age ~40)
            features['activity_level'] / 100,  # Normalized activity
            features['sleep_quality'] * features['hrv_mean'],  # Recovery index
            features['hrv_mean'] / features['heart_rate'] if features['heart_rate'] > 0 else 0,
            # Placeholder for historical features
            features['heart_rate'] * 0.95,  # 7-day average HR placeholder
            features['activity_level'] * 1.1,  # 7-day average activity placeholder
            features['hrv_mean'] * 0.98,  # 7-day average HRV placeholder
            0.15,  # HR trend placeholder
            0.05,  # HRV trend placeholder
            features['activity_level'] * 0.9,  # Yesterday's activity placeholder
            features['sleep_quality'] * 0.95,  # Average sleep quality placeholder
            40  # Age placeholder (would come from patient data)
        ]).reshape(1, -1)
    
    def check_critical_conditions(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Check for critical health conditions requiring immediate attention"""
        hr = features['heart_rate']
        hrv = features['hrv_mean']
        rr = features['respiratory_rate']
        activity = features['activity_level']
        
        critical_conditions = []
        
        # Critical heart rate conditions
        if hr > 150 and activity < 100:
            critical_conditions.append({
                'type': 'critical',
                'message': 'Dangerously high resting heart rate detected (>150 bpm)'
            })
        
        if hr < 40:
            critical_conditions.append({
                'type': 'critical',
                'message': 'Dangerously low heart rate detected (<40 bpm)'
            })
        
        # Critical respiratory conditions
        if rr > 25:
            critical_conditions.append({
                'type': 'warning',
                'message': 'Elevated respiratory rate detected (>25 breaths/min)'
            })
        
        if rr < 8:
            critical_conditions.append({
                'type': 'warning',
                'message': 'Low respiratory rate detected (<8 breaths/min)'
            })
        
        # Severely compromised autonomic function
        if hrv < 10 and hr > 80:
            critical_conditions.append({
                'type': 'warning',
                'message': 'Very low heart rate variability detected'
            })
        
        return {
            'has_critical': len([c for c in critical_conditions if c['type'] == 'critical']) > 0,
            'conditions': critical_conditions
        }
    
    def run_inference(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Run all models and return predictions"""
        start_time = time.time()
        
        # Validate inputs
        validated = self.validate_inputs(features)
        
        results = {
            'alerts': []
        }
        
        # Check for critical conditions first
        critical_check = self.check_critical_conditions(validated)
        if critical_check['conditions']:
            results['alerts'] = critical_check['conditions']
        
        # Run SVM heart rhythm classification
        try:
            svm_features = self.prepare_svm_features(validated)
            svm_result = self.model_manager.predict_svm(svm_features)
            results['rhythm'] = svm_result
        except Exception as e:
            logger.error(f"SVM inference failed: {e}")
            results['rhythm'] = {"prediction": "Unknown", "confidence": 0.0}
        
        # Run GBM risk assessment
        try:
            gbm_features = self.prepare_gbm_features(validated)
            gbm_result = self.model_manager.predict_gbm(gbm_features)
            results['risk'] = gbm_result
        except Exception as e:
            logger.error(f"GBM inference failed: {e}")
            results['risk'] = {"level": "Unknown", "score": 0.0, "confidence": 0.0}
        
        # Run Neural Network HRV analysis
        try:
            nn_features = self.prepare_nn_features(validated)
            nn_result = self.model_manager.predict_nn(nn_features)
            results['hrv'] = nn_result
        except Exception as e:
            logger.error(f"NN inference failed: {e}")
            results['hrv'] = {"pattern": "Unknown", "confidence": 0.0}
        
        # Run Random Forest fitness assessment
        try:
            rf_features = self.prepare_rf_features(validated)
            rf_result = self.model_manager.predict_rf(rf_features)
            results['fitness'] = rf_result
        except Exception as e:
            logger.error(f"RF inference failed: {e}")
            results['fitness'] = {
                "vo2max": 35.0,
                "cardiovascular_age": 40,
                "fitness_category": "Unknown"
            }
        
        # Add metadata
        results['processing_time_ms'] = int((time.time() - start_time) * 1000)
        results['model_versions'] = self.model_manager.model_versions
        
        return results