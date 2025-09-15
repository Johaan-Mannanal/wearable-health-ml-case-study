"""
Configuration management for TelemetryHealthCare Backend
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Configuration
    api_key: str = "your-secure-api-key-here"
    secret_key: str = "your-secret-key-for-jwt"
    algorithm: str = "HS256"
    
    # Database Configuration
    database_url: str = "postgresql://user:password@localhost:5432/telemetryhealth"
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # ML Models
    ml_model_dir: str = "ml_models"
    ml_model_cache_size: int = 4
    
    # Dashboard Authentication
    dashboard_username: str = "admin"
    dashboard_password: str = "changeme123"  # Change in production!
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60
    
    # WebSocket
    ws_heartbeat_interval: int = 30
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "protected_namespaces": ("settings_",)
    }

# Create settings instance
settings = Settings()

# Override with environment variables if present
# Railway provides DATABASE_URL or DATABASE_PRIVATE_URL
if os.getenv("DATABASE_URL"):
    settings.database_url = os.getenv("DATABASE_URL")
elif os.getenv("DATABASE_PRIVATE_URL"):
    settings.database_url = os.getenv("DATABASE_PRIVATE_URL")
    
if os.getenv("API_KEY"):
    settings.api_key = os.getenv("API_KEY")
    
if os.getenv("ENVIRONMENT"):
    settings.environment = os.getenv("ENVIRONMENT")
    settings.debug = settings.environment == "development"