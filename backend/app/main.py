"""
TelemetryHealthCare Backend API
FastAPI application for serving ML models and dashboard
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from app.api import health, ml, dashboard, websocket
from app.ml.model_loader import ModelManager
from app.database import engine, Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model manager instance
model_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global model_manager
    
    # Startup
    logger.info("Starting TelemetryHealthCare Backend...")
    
    # Initialize database
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database connected successfully")
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
        logger.warning("Running without database - some features will be unavailable")
    
    # Load ML models
    logger.info("Loading ML models...")
    from app.config import settings
    model_dir = Path(__file__).parent.parent / settings.ml_model_dir
    model_manager = ModelManager(model_dir)
    app.state.model_manager = model_manager
    
    logger.info("Backend started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down backend...")

# Create FastAPI app
app = FastAPI(
    title="TelemetryHealthCare API",
    description="Backend API for cardiovascular health monitoring with ML",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health.router, prefix="/api/health", tags=["Health Data"])
app.include_router(ml.router, prefix="/api/ml", tags=["ML Inference"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])

# Serve static files for dashboard
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return {
        "status": "healthy",
        "models_loaded": model_manager is not None,
        "version": "1.0.0"
    }

@app.get("/api/status")
async def api_status():
    """Get API status and loaded models information"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "status": "operational",
        "models": {
            "svm": model_manager.is_model_loaded("svm"),
            "gbm": model_manager.is_model_loaded("gbm"),
            "nn": model_manager.is_model_loaded("nn"),
            "rf": model_manager.is_model_loaded("rf")
        },
        "model_versions": model_manager.model_versions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)