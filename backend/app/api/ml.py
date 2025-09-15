"""
ML inference API endpoints
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any, Optional
import numpy as np

from app.models.health_data import HealthMetrics
from app.ml.inference import InferenceEngine

router = APIRouter()

@router.post("/analyze")
async def analyze_health_data(
    metrics: HealthMetrics,
    request: Request,
    model: Optional[str] = "all"
):
    """Run ML analysis on health metrics"""
    
    # Get model manager from app state
    model_manager = request.app.state.model_manager
    if not model_manager:
        raise HTTPException(status_code=503, detail="ML models not loaded")
    
    # Create inference engine
    inference_engine = InferenceEngine(model_manager)
    
    # Run inference
    results = inference_engine.run_inference(metrics.dict())
    
    # Filter by specific model if requested
    if model != "all" and model in ["svm", "gbm", "nn", "rf"]:
        model_map = {
            "svm": "rhythm",
            "gbm": "risk",
            "nn": "hrv",
            "rf": "fitness"
        }
        if model_map[model] in results:
            return {
                "model": model,
                "results": results[model_map[model]],
                "processing_time_ms": results.get("processing_time_ms", 0)
            }
    
    return results

@router.get("/models/status")
async def get_models_status(request: Request):
    """Get status of all loaded ML models"""
    
    model_manager = request.app.state.model_manager
    if not model_manager:
        raise HTTPException(status_code=503, detail="ML models not loaded")
    
    return {
        "models": {
            "svm": {
                "loaded": model_manager.is_model_loaded("svm"),
                "name": "Heart Rhythm Classifier",
                "accuracy": "92.4%",
                "version": model_manager.model_versions.get("svm", {}).get("version", "1.0.0")
            },
            "gbm": {
                "loaded": model_manager.is_model_loaded("gbm"),
                "name": "Health Risk Assessment",
                "accuracy": "99.4%",
                "version": model_manager.model_versions.get("gbm", {}).get("version", "1.0.0")
            },
            "nn": {
                "loaded": model_manager.is_model_loaded("nn"),
                "name": "HRV Pattern Analyzer",
                "accuracy": "99.4%",
                "version": model_manager.model_versions.get("nn", {}).get("version", "1.0.0")
            },
            "rf": {
                "loaded": model_manager.is_model_loaded("rf"),
                "name": "Cardiovascular Fitness",
                "accuracy": "96.0%",
                "version": model_manager.model_versions.get("rf", {}).get("version", "1.0.0")
            }
        },
        "total_loaded": sum([
            model_manager.is_model_loaded("svm"),
            model_manager.is_model_loaded("gbm"),
            model_manager.is_model_loaded("nn"),
            model_manager.is_model_loaded("rf")
        ])
    }