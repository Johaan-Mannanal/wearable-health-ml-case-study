"""
ML Model loading and management for TelemetryHealthCare
"""

import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.models = {}
        self.model_versions = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all ML models at startup"""
        try:
            # Load SVM Heart Rhythm Model
            svm_path = self.model_dir / "svm_heart_rhythm_model.pkl"
            if svm_path.exists():
                self.models['svm'] = joblib.load(svm_path)
                logger.info("Loaded SVM heart rhythm model")
            else:
                logger.warning(f"SVM model not found at {svm_path}")
            
            # Load GBM Health Risk Model
            gbm_path = self.model_dir / "gbm_health_risk_model.pkl"
            if gbm_path.exists():
                self.models['gbm'] = joblib.load(gbm_path)
                logger.info("Loaded GBM health risk model")
            else:
                logger.warning(f"GBM model not found at {gbm_path}")
            
            # Load Neural Network HRV Model
            nn_path = self.model_dir / "hrv_pattern_nn_model.pkl"
            if nn_path.exists():
                self.models['nn'] = joblib.load(nn_path)
                logger.info("Loaded Neural Network HRV model")
            else:
                logger.warning(f"NN model not found at {nn_path}")
            
            # Load Random Forest Fitness Model
            rf_path = self.model_dir / "cardiovascular_fitness_model.pkl"
            if rf_path.exists():
                self.models['rf'] = joblib.load(rf_path)
                logger.info("Loaded Random Forest fitness model")
            else:
                logger.warning(f"RF model not found at {rf_path}")
            
            # Load model metadata
            for model_name in ['svm', 'gbm', 'nn', 'rf']:
                metadata_path = self.model_dir / f"{model_name}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.model_versions[model_name] = json.load(f)
                else:
                    self.model_versions[model_name] = {"version": "1.0.0", "accuracy": "unknown"}
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is loaded"""
        return model_name in self.models
    
    def get_model(self, model_name: str):
        """Get a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        return self.models[model_name]
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get all loaded models"""
        return self.models
    
    def predict_svm(self, features: np.ndarray) -> Dict[str, Any]:
        """Run SVM heart rhythm classification"""
        if 'svm' not in self.models:
            return {"error": "SVM model not loaded"}
        
        try:
            model = self.models['svm']
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            return {
                "prediction": "Irregular" if prediction == 1 else "Normal",
                "confidence": float(max(probabilities)),
                "probabilities": {
                    "normal": float(probabilities[0]),
                    "irregular": float(probabilities[1]) if len(probabilities) > 1 else 0
                }
            }
        except Exception as e:
            logger.error(f"SVM prediction error: {e}")
            return {"error": str(e)}
    
    def predict_gbm(self, features: np.ndarray) -> Dict[str, Any]:
        """Run GBM health risk assessment"""
        if 'gbm' not in self.models:
            return {"error": "GBM model not loaded"}
        
        try:
            model = self.models['gbm']
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            risk_levels = ["Low", "Moderate", "High", "Critical"]
            risk_level = risk_levels[min(int(prediction), len(risk_levels)-1)]
            
            return {
                "risk_level": risk_level,
                "risk_score": float(probabilities[1]) if len(probabilities) > 1 else float(prediction),
                "confidence": float(max(probabilities))
            }
        except Exception as e:
            logger.error(f"GBM prediction error: {e}")
            return {"error": str(e)}
    
    def predict_nn(self, features: np.ndarray) -> Dict[str, Any]:
        """Run Neural Network HRV pattern analysis"""
        if 'nn' not in self.models:
            return {"error": "Neural Network model not loaded"}
        
        try:
            model = self.models['nn']
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            patterns = ["Normal", "Atrial Fibrillation", "Bradycardia", "Tachycardia"]
            pattern = patterns[min(int(prediction), len(patterns)-1)]
            
            return {
                "pattern": pattern,
                "confidence": float(max(probabilities)),
                "pattern_probabilities": {
                    pattern: float(prob) 
                    for pattern, prob in zip(patterns, probabilities)
                }
            }
        except Exception as e:
            logger.error(f"NN prediction error: {e}")
            return {"error": str(e)}
    
    def predict_rf(self, features: np.ndarray) -> Dict[str, Any]:
        """Run Random Forest cardiovascular fitness assessment"""
        if 'rf' not in self.models:
            return {"error": "Random Forest model not loaded"}
        
        try:
            model = self.models['rf']
            
            # The RF model might return multiple values for fitness metrics
            prediction = model.predict(features)[0]
            
            # Calculate VO2max and other fitness metrics
            vo2max = self.calculate_vo2max(features[0])
            cardiovascular_age = self.calculate_cardiovascular_age(vo2max, features[0])
            fitness_category = self.get_fitness_category(vo2max)
            
            return {
                "fitness_score": float(prediction),
                "vo2max": float(vo2max),
                "cardiovascular_age": int(cardiovascular_age),
                "fitness_category": fitness_category
            }
        except Exception as e:
            logger.error(f"RF prediction error: {e}")
            return {"error": str(e)}
    
    def calculate_vo2max(self, features: np.ndarray) -> float:
        """Calculate VO2max from features"""
        # Simplified VO2max calculation
        # In reality, this would use the model's specific formula
        base_vo2max = 45.0
        hr_factor = (220 - features[0]) * 0.15  # Assuming first feature is heart rate
        return max(20, min(80, base_vo2max + hr_factor))
    
    def calculate_cardiovascular_age(self, vo2max: float, features: np.ndarray) -> int:
        """Calculate cardiovascular age based on VO2max and features"""
        # Simplified cardiovascular age calculation
        chronological_age = 40  # Default if not provided
        vo2max_adjustment = (vo2max - 35) * -0.5
        return max(20, min(80, chronological_age + vo2max_adjustment))
    
    def get_fitness_category(self, vo2max: float) -> str:
        """Categorize fitness level based on VO2max"""
        if vo2max >= 50:
            return "Excellent"
        elif vo2max >= 42:
            return "Good"
        elif vo2max >= 35:
            return "Fair"
        else:
            return "Poor"