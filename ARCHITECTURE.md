# TelemetryHealthCare System Architecture

> ⚠️ Synthetic-data research project — not a medical device. See the root README.md and MODEL_CARD.md.

## Executive Summary

This document is a design sketch for an optional full-stack extra: a small FastAPI backend and web
dashboard that would serve the project's ML models (trained on **synthetic** data) to the WIP iOS
app. It is a learning/portfolio exercise for a handful of test users — **not** a healthcare system,
and it is not deployed. "Patient" in the schema/API below simply means "app user"; the app handles
no real patient data.

### Key Objectives
- **Speed**: stand up a working demo backend quickly
- **Cost**: keep infrastructure minimal (<$50/month)
- **Simplicity**: a single deployable unit
- **Scalability**: leave a clear path to a more modular architecture
- **Framing**: keep everything explicitly non-clinical (no compliance is claimed or implemented)

## Table of Contents
1. [System Overview](#system-overview)
2. [MVP Architecture](#mvp-architecture)
3. [API Specifications](#api-specifications)
4. [Database Schema](#database-schema)
5. [ML Model Service](#ml-model-service)
6. [Healthcare Dashboard](#healthcare-dashboard)
7. [iOS App Integration](#ios-app-integration)
8. [Security & Compliance](#security--compliance)
9. [Deployment Strategy](#deployment-strategy)
10. [Implementation Roadmap](#implementation-roadmap)
11. [Cost Analysis](#cost-analysis)

## System Overview

### Current State
- **iOS App**: SwiftUI-based with on-device ML processing
- **ML Models**: 4 trained models (SVM, XGBoost, Neural Network, Random Forest)
- **Data Source**: Apple Watch via HealthKit
- **Processing**: 100% on-device, no cloud dependencies

### Target State (MVP)
```
┌─────────────────┐     ┌──────────────────────────────┐     ┌─────────────────┐
│                 │     │                              │     │                 │
│   iOS App       │────▶│   FastAPI Backend            │────▶│   Healthcare    │
│   (Modified)    │     │   - ML Models                │     │   Dashboard     │
│                 │     │   - Health API               │     │   (Web)         │
└─────────────────┘     │   - WebSocket Server         │     └─────────────────┘
                        │   - Database (PostgreSQL)    │
                        └──────────────────────────────┘
                                    │
                                    ▼
                            ┌──────────────┐
                            │  PostgreSQL  │
                            │   Database   │
                            └──────────────┘
```

## MVP Architecture

### Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Backend API** | FastAPI (Python 3.11) | Fast development, async support, auto-documentation |
| **ML Models** | scikit-learn, XGBoost, TensorFlow | Existing trained models, easy integration |
| **Database** | PostgreSQL 15 | ACID compliance, JSON support, time-series extensions |
| **Real-time** | WebSockets (FastAPI) | Built-in support, bidirectional communication |
| **Dashboard** | HTML5 + Vanilla JS + Chart.js | No build process, fast development |
| **Deployment** | Docker + Railway.app | Simple deployment, automatic SSL, low cost |

### System Components

#### 1. FastAPI Backend (Monolith)
```python
healthcare-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── config.py                # Configuration management
│   ├── database.py              # Database connection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── health_data.py      # Pydantic models
│   │   ├── patient.py          # Patient models
│   │   └── assessment.py       # Assessment results
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py           # Health data endpoints
│   │   ├── ml.py               # ML inference endpoints
│   │   ├── dashboard.py        # Dashboard data endpoints
│   │   └── websocket.py        # Real-time connections
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── model_loader.py     # Load pickle models
│   │   ├── inference.py        # Run predictions
│   │   └── preprocessing.py    # Data preprocessing
│   ├── services/
│   │   ├── __init__.py
│   │   ├── health_monitor.py   # Health monitoring logic
│   │   ├── alert_service.py    # Alert management
│   │   └── data_service.py     # Data persistence
│   └── static/                 # Dashboard files
│       ├── index.html
│       ├── dashboard.js
│       └── styles.css
├── ml_models/                   # Trained model files
│   ├── svm_heart_rhythm_model.pkl
│   ├── gbm_health_risk_model.pkl
│   ├── hrv_pattern_nn_model.pkl
│   └── cardiovascular_fitness_model.pkl
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

#### 2. Component Interactions

```mermaid
sequenceDiagram
    participant iOS as iOS App
    participant API as FastAPI
    participant ML as ML Service
    participant DB as PostgreSQL
    participant WS as WebSocket
    participant Dash as Dashboard

    iOS->>API: POST /api/health/data
    API->>ML: Process health data
    ML->>ML: Run 4 ML models
    ML->>API: Return predictions
    API->>DB: Store results
    API->>WS: Broadcast update
    WS->>Dash: Push real-time data
    Dash->>Dash: Update UI
```

## API Specifications

### Authentication
```python
# MVP: Basic API key authentication
headers = {
    "X-API-Key": "generated-api-key",
    "Content-Type": "application/json"
}
```

### Core Endpoints

#### Health Data Submission
```python
POST /api/health/data
{
    "patient_id": "uuid",
    "timestamp": "2025-01-15T10:30:00Z",
    "metrics": {
        "heart_rate": 72,
        "heart_rate_std": 5.2,
        "hrv_mean": 45,
        "pnn50": 0.25,
        "respiratory_rate": 16,
        "activity_level": 150,
        "sleep_quality": 0.85
    },
    "device_info": {
        "model": "Apple Watch Series 10",
        "os_version": "11.0"
    }
}

Response:
{
    "assessment_id": "uuid",
    "predictions": {
        "rhythm_status": {
            "prediction": "Normal",
            "confidence": 0.94
        },
        "health_risk": {
            "level": "Low",
            "score": 0.15,
            "confidence": 0.89
        },
        "hrv_pattern": {
            "pattern": "Healthy",
            "confidence": 0.91
        },
        "fitness_level": {
            "vo2max": 42.5,
            "cardiovascular_age": 35,
            "fitness_category": "Good"
        }
    },
    "alerts": [],
    "recommendations": [
        "Maintain current activity level",
        "Continue regular sleep schedule"
    ]
}
```

#### ML Inference
```python
POST /api/ml/analyze
{
    "model": "all",  # or specific: "svm", "gbm", "nn", "rf"
    "features": {
        "heart_rate": 72,
        "heart_rate_std": 5.2,
        "hrv_mean": 45,
        "pnn50": 0.25,
        "respiratory_rate": 16,
        "activity_level": 150,
        "sleep_quality": 0.85
    }
}

Response:
{
    "results": {
        "svm": {"prediction": "Normal", "confidence": 0.94},
        "gbm": {"risk_level": "Low", "score": 0.15},
        "nn": {"pattern": "Healthy", "confidence": 0.91},
        "rf": {"fitness_score": 0.75, "vo2max": 42.5}
    },
    "processing_time_ms": 45
}
```

#### Dashboard Data
```python
GET /api/dashboard/patients
Response:
{
    "patients": [
        {
            "id": "uuid",
            "name": "Patient Name",
            "age": 45,
            "last_update": "2025-01-15T10:30:00Z",
            "status": "stable",
            "current_metrics": {...},
            "alerts": []
        }
    ]
}

GET /api/dashboard/patient/{patient_id}/history
Parameters:
  - timeframe: "1h", "24h", "7d", "30d"
  
Response:
{
    "patient_id": "uuid",
    "timeframe": "24h",
    "data_points": [
        {
            "timestamp": "2025-01-15T10:00:00Z",
            "heart_rate": 72,
            "risk_score": 0.15,
            "fitness_level": "Good"
        }
    ],
    "statistics": {
        "avg_heart_rate": 70,
        "min_heart_rate": 55,
        "max_heart_rate": 95,
        "trend": "stable"
    }
}
```

#### WebSocket Connection
```javascript
// WebSocket endpoint for real-time updates
ws://api.telemetryhealth.com/ws/dashboard

// Message format
{
    "type": "health_update",
    "patient_id": "uuid",
    "data": {
        "timestamp": "2025-01-15T10:30:00Z",
        "heart_rate": 72,
        "alerts": []
    }
}
```

### Error Responses
```python
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid heart rate value",
        "details": {
            "field": "heart_rate",
            "value": 300,
            "constraint": "Must be between 30-250"
        }
    },
    "timestamp": "2025-01-15T10:30:00Z"
}
```

## Database Schema

### PostgreSQL Tables

```sql
-- Patients table
CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) UNIQUE NOT NULL,  -- iOS app user ID
    age INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Health metrics table (time-series data)
CREATE TABLE health_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    heart_rate REAL,
    heart_rate_std REAL,
    hrv_mean REAL,
    pnn50 REAL,
    respiratory_rate REAL,
    activity_level REAL,
    sleep_quality REAL,
    raw_data JSONB,  -- Store complete data for debugging
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX idx_patient_timestamp (patient_id, timestamp DESC)
);

-- ML assessments table
CREATE TABLE ml_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    metric_id UUID REFERENCES health_metrics(id),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    rhythm_status VARCHAR(50),
    rhythm_confidence REAL,
    risk_level VARCHAR(50),
    risk_score REAL,
    risk_confidence REAL,
    hrv_pattern VARCHAR(50),
    pattern_confidence REAL,
    vo2max REAL,
    cardiovascular_age INTEGER,
    fitness_category VARCHAR(50),
    processing_time_ms INTEGER,
    model_versions JSONB,  -- Track which model versions were used
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX idx_patient_assessment (patient_id, timestamp DESC)
);

-- Alerts table
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    assessment_id UUID REFERENCES ml_assessments(id),
    alert_type VARCHAR(50) NOT NULL,  -- 'critical', 'warning', 'info'
    title VARCHAR(255) NOT NULL,
    message TEXT,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX idx_patient_alerts (patient_id, created_at DESC),
    INDEX idx_unacknowledged (is_acknowledged, created_at DESC)
);

-- Dashboard users (healthcare providers)
CREATE TABLE dashboard_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'viewer',  -- 'admin', 'clinician', 'viewer'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Audit log
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES dashboard_users(id),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX idx_audit_user (user_id, created_at DESC)
);
```

### Database Optimization

```sql
-- Enable TimescaleDB extension for time-series optimization
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Convert health_metrics to hypertable
SELECT create_hypertable('health_metrics', 'timestamp');

-- Create continuous aggregate for hourly stats
CREATE MATERIALIZED VIEW hourly_stats
WITH (timescaledb.continuous) AS
SELECT 
    patient_id,
    time_bucket('1 hour', timestamp) AS hour,
    AVG(heart_rate) as avg_heart_rate,
    MIN(heart_rate) as min_heart_rate,
    MAX(heart_rate) as max_heart_rate,
    AVG(hrv_mean) as avg_hrv,
    AVG(risk_score) as avg_risk_score
FROM health_metrics m
JOIN ml_assessments a ON m.id = a.metric_id
GROUP BY patient_id, hour;
```

## ML Model Service

### Model Loading and Management

```python
# app/ml/model_loader.py
import joblib
import logging
from pathlib import Path
from typing import Dict, Any

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
            self.models['svm'] = joblib.load(
                self.model_dir / 'svm_heart_rhythm_model.pkl'
            )
            self.models['gbm'] = joblib.load(
                self.model_dir / 'gbm_health_risk_model.pkl'
            )
            self.models['nn'] = joblib.load(
                self.model_dir / 'hrv_pattern_nn_model.pkl'
            )
            self.models['rf'] = joblib.load(
                self.model_dir / 'cardiovascular_fitness_model.pkl'
            )
            
            # Load model metadata
            for model_name in self.models.keys():
                metadata_path = self.model_dir / f'{model_name}_metadata.json'
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        self.model_versions[model_name] = json.load(f)
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
```

### Inference Pipeline

```python
# app/ml/inference.py
import numpy as np
from typing import Dict, Any, Tuple
import time

class InferenceEngine:
    def __init__(self, model_manager: ModelManager):
        self.models = model_manager.models
        self.model_versions = model_manager.model_versions
    
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
    
    def prepare_features(self, features: Dict[str, float], model_type: str) -> np.ndarray:
        """Prepare features for specific model"""
        if model_type == 'svm':
            # SVM uses 3 features
            return np.array([
                features['heart_rate'],
                features['heart_rate_std'],
                features['pnn50']
            ]).reshape(1, -1)
        
        elif model_type == 'gbm':
            # GBM uses 8 features including derived ones
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
        
        # Add other model preparations...
        
    def run_inference(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Run all models and return predictions"""
        start_time = time.time()
        validated = self.validate_inputs(features)
        results = {}
        
        # Check for critical conditions first
        critical_check = self.check_critical_conditions(validated)
        if critical_check['is_critical']:
            results['alerts'] = [critical_check['message']]
        
        # Run SVM heart rhythm classification
        svm_features = self.prepare_features(validated, 'svm')
        svm_pred = self.models['svm'].predict(svm_features)[0]
        svm_proba = self.models['svm'].predict_proba(svm_features)[0]
        results['rhythm'] = {
            'prediction': 'Irregular' if svm_pred == 1 else 'Normal',
            'confidence': float(max(svm_proba))
        }
        
        # Run GBM risk assessment
        gbm_features = self.prepare_features(validated, 'gbm')
        gbm_pred = self.models['gbm'].predict(gbm_features)[0]
        gbm_proba = self.models['gbm'].predict_proba(gbm_features)[0]
        results['risk'] = {
            'level': self.map_risk_level(gbm_pred),
            'score': float(gbm_proba[1]),  # Probability of high risk
            'confidence': float(max(gbm_proba))
        }
        
        # Add other model inferences...
        
        results['processing_time_ms'] = int((time.time() - start_time) * 1000)
        results['model_versions'] = self.model_versions
        
        return results
    
    def check_critical_conditions(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Illustrative, non-diagnostic wellness thresholds on synthetic data (not medical alerts)."""
        hr = features['heart_rate']
        hrv = features['hrv_mean']
        rr = features['respiratory_rate']
        activity = features['activity_level']
        
        # Heart-rate thresholds (illustrative, non-diagnostic)
        if hr > 150 and activity < 100:
            return {
                'is_critical': True,
                'message': 'Heart rate is well above the typical resting range'
            }
        
        if hr < 40:
            return {
                'is_critical': True,
                'message': 'Heart rate is well below the typical range'
            }
        
        # Respiratory-rate thresholds (illustrative, non-diagnostic)
        if rr > 25 or rr < 8:
            return {
                'is_critical': True,
                'message': 'Respiratory rate is outside the typical range'
            }
        
        # Low HRV (illustrative, non-diagnostic)
        if hrv < 10 and hr > 80:
            return {
                'is_critical': True,
                'message': 'Heart rate variability is below the typical range'
            }
        
        return {'is_critical': False}
    
    def map_risk_level(self, risk_class: int) -> str:
        """Map numeric risk class to human-readable level"""
        risk_mapping = {
            0: 'Low',
            1: 'Moderate',
            2: 'High',
            3: 'Critical'
        }
        return risk_mapping.get(risk_class, 'Unknown')
```

## Healthcare Dashboard

### Dashboard Architecture

```html
<!-- static/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TelemetryHealthCare Dashboard</title>
    <link rel="stylesheet" href="styles.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <header class="dashboard-header">
            <h1>TelemetryHealthCare Monitor</h1>
            <div class="connection-status" id="connectionStatus">
                <span class="status-indicator"></span>
                <span class="status-text">Connecting...</span>
            </div>
        </header>

        <!-- Patient Grid -->
        <div class="patient-grid" id="patientGrid">
            <!-- Patient cards will be dynamically inserted here -->
        </div>

        <!-- Detail View Modal -->
        <div class="modal" id="patientDetailModal">
            <div class="modal-content">
                <span class="close">&times;</span>
                <h2 id="patientName"></h2>
                
                <!-- Real-time Metrics -->
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Heart Rate</h3>
                        <div class="metric-value" id="heartRate">--</div>
                        <div class="metric-unit">bpm</div>
                    </div>
                    <div class="metric-card">
                        <h3>HRV</h3>
                        <div class="metric-value" id="hrv">--</div>
                        <div class="metric-unit">ms</div>
                    </div>
                    <div class="metric-card">
                        <h3>Risk Level</h3>
                        <div class="metric-value risk-level" id="riskLevel">--</div>
                    </div>
                    <div class="metric-card">
                        <h3>Fitness</h3>
                        <div class="metric-value" id="fitness">--</div>
                        <div class="metric-unit">VO2max</div>
                    </div>
                </div>

                <!-- Charts -->
                <div class="charts-container">
                    <div class="chart-wrapper">
                        <canvas id="heartRateChart"></canvas>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="riskChart"></canvas>
                    </div>
                </div>

                <!-- Alerts -->
                <div class="alerts-section" id="alertsSection">
                    <h3>Active Alerts</h3>
                    <div id="alertsList"></div>
                </div>
            </div>
        </div>
    </div>

    <script src="dashboard.js"></script>
</body>
</html>
```

### Dashboard JavaScript

```javascript
// static/dashboard.js
class TelemetryDashboard {
    constructor() {
        this.ws = null;
        this.patients = new Map();
        this.charts = {};
        this.selectedPatientId = null;
        this.apiUrl = window.location.origin;
        
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.loadPatients();
        this.setupEventListeners();
        this.startPeriodicRefresh();
    }
    
    connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/dashboard`);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus('connected');
        };
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('error');
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus('disconnected');
            // Reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        };
    }
    
    handleWebSocketMessage(message) {
        switch(message.type) {
            case 'health_update':
                this.updatePatientData(message.patient_id, message.data);
                break;
            case 'alert':
                this.handleAlert(message);
                break;
            case 'model_update':
                this.updateModelStatus(message);
                break;
        }
    }
    
    async loadPatients() {
        try {
            const response = await fetch(`${this.apiUrl}/api/dashboard/patients`);
            const data = await response.json();
            
            this.patients.clear();
            data.patients.forEach(patient => {
                this.patients.set(patient.id, patient);
            });
            
            this.renderPatientGrid();
        } catch (error) {
            console.error('Failed to load patients:', error);
        }
    }
    
    renderPatientGrid() {
        const grid = document.getElementById('patientGrid');
        grid.innerHTML = '';
        
        this.patients.forEach(patient => {
            const card = this.createPatientCard(patient);
            grid.appendChild(card);
        });
    }
    
    createPatientCard(patient) {
        const card = document.createElement('div');
        card.className = `patient-card ${this.getStatusClass(patient.status)}`;
        card.dataset.patientId = patient.id;
        
        card.innerHTML = `
            <div class="patient-header">
                <h3>${patient.name}</h3>
                <span class="patient-age">Age: ${patient.age}</span>
            </div>
            <div class="patient-metrics">
                <div class="metric">
                    <span class="metric-label">HR:</span>
                    <span class="metric-value">${patient.current_metrics?.heart_rate || '--'} bpm</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Risk:</span>
                    <span class="metric-value risk-${patient.current_metrics?.risk_level?.toLowerCase() || 'unknown'}">
                        ${patient.current_metrics?.risk_level || '--'}
                    </span>
                </div>
            </div>
            <div class="patient-footer">
                <span class="last-update">Updated: ${this.formatTime(patient.last_update)}</span>
                ${patient.alerts?.length > 0 ? `<span class="alert-badge">${patient.alerts.length}</span>` : ''}
            </div>
        `;
        
        card.addEventListener('click', () => this.showPatientDetail(patient.id));
        
        return card;
    }
    
    async showPatientDetail(patientId) {
        this.selectedPatientId = patientId;
        const patient = this.patients.get(patientId);
        
        // Update modal header
        document.getElementById('patientName').textContent = patient.name;
        
        // Load historical data
        await this.loadPatientHistory(patientId);
        
        // Update real-time metrics
        this.updateDetailMetrics(patient.current_metrics);
        
        // Show modal
        document.getElementById('patientDetailModal').style.display = 'block';
        
        // Initialize charts
        this.initializeCharts();
    }
    
    async loadPatientHistory(patientId) {
        try {
            const response = await fetch(
                `${this.apiUrl}/api/dashboard/patient/${patientId}/history?timeframe=24h`
            );
            const data = await response.json();
            
            this.updateCharts(data.data_points);
        } catch (error) {
            console.error('Failed to load patient history:', error);
        }
    }
    
    initializeCharts() {
        // Heart Rate Chart
        const hrCtx = document.getElementById('heartRateChart').getContext('2d');
        if (this.charts.heartRate) {
            this.charts.heartRate.destroy();
        }
        
        this.charts.heartRate = new Chart(hrCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Heart Rate',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 40,
                        max: 120
                    }
                }
            }
        });
        
        // Risk Score Chart
        const riskCtx = document.getElementById('riskChart').getContext('2d');
        if (this.charts.risk) {
            this.charts.risk.destroy();
        }
        
        this.charts.risk = new Chart(riskCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Risk Score',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
    
    updateCharts(dataPoints) {
        if (!dataPoints || dataPoints.length === 0) return;
        
        const labels = dataPoints.map(dp => this.formatTime(dp.timestamp));
        const heartRates = dataPoints.map(dp => dp.heart_rate);
        const riskScores = dataPoints.map(dp => dp.risk_score);
        
        // Update heart rate chart
        if (this.charts.heartRate) {
            this.charts.heartRate.data.labels = labels;
            this.charts.heartRate.data.datasets[0].data = heartRates;
            this.charts.heartRate.update();
        }
        
        // Update risk chart
        if (this.charts.risk) {
            this.charts.risk.data.labels = labels;
            this.charts.risk.data.datasets[0].data = riskScores;
            this.charts.risk.update();
        }
    }
    
    updateDetailMetrics(metrics) {
        if (!metrics) return;
        
        document.getElementById('heartRate').textContent = metrics.heart_rate || '--';
        document.getElementById('hrv').textContent = metrics.hrv_mean || '--';
        document.getElementById('riskLevel').textContent = metrics.risk_level || '--';
        document.getElementById('fitness').textContent = metrics.vo2max || '--';
        
        // Update risk level styling
        const riskElement = document.getElementById('riskLevel');
        riskElement.className = `metric-value risk-level risk-${(metrics.risk_level || 'unknown').toLowerCase()}`;
    }
    
    updatePatientData(patientId, data) {
        const patient = this.patients.get(patientId);
        if (!patient) return;
        
        // Update patient data
        patient.current_metrics = data;
        patient.last_update = new Date().toISOString();
        
        // Update grid card
        const card = document.querySelector(`[data-patient-id="${patientId}"]`);
        if (card) {
            // Update card metrics
            const hrElement = card.querySelector('.metric-value');
            if (hrElement) {
                hrElement.textContent = `${data.heart_rate || '--'} bpm`;
            }
        }
        
        // Update detail view if this patient is selected
        if (this.selectedPatientId === patientId) {
            this.updateDetailMetrics(data);
            
            // Add new data point to charts
            if (this.charts.heartRate && data.heart_rate) {
                this.addChartDataPoint(this.charts.heartRate, 
                    this.formatTime(new Date()), 
                    data.heart_rate);
            }
            
            if (this.charts.risk && data.risk_score !== undefined) {
                this.addChartDataPoint(this.charts.risk, 
                    this.formatTime(new Date()), 
                    data.risk_score);
            }
        }
    }
    
    addChartDataPoint(chart, label, value) {
        // Keep only last 50 points for performance
        if (chart.data.labels.length > 50) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.data.labels.push(label);
        chart.data.datasets[0].data.push(value);
        chart.update();
    }
    
    handleAlert(alert) {
        // Show notification
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification('TelemetryHealthCare Alert', {
                body: alert.message,
                icon: '/favicon.ico',
                requireInteraction: true
            });
        }
        
        // Update alerts section
        const alertsList = document.getElementById('alertsList');
        if (alertsList) {
            const alertElement = document.createElement('div');
            alertElement.className = `alert alert-${alert.severity}`;
            alertElement.innerHTML = `
                <div class="alert-time">${this.formatTime(new Date())}</div>
                <div class="alert-message">${alert.message}</div>
                <button class="alert-acknowledge" data-alert-id="${alert.id}">Acknowledge</button>
            `;
            alertsList.prepend(alertElement);
        }
    }
    
    formatTime(timestamp) {
        if (!timestamp) return '--';
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }
    
    getStatusClass(status) {
        const statusClasses = {
            'stable': 'status-stable',
            'warning': 'status-warning',
            'critical': 'status-critical',
            'offline': 'status-offline'
        };
        return statusClasses[status] || 'status-unknown';
    }
    
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connectionStatus');
        const indicator = statusElement.querySelector('.status-indicator');
        const text = statusElement.querySelector('.status-text');
        
        switch(status) {
            case 'connected':
                indicator.className = 'status-indicator connected';
                text.textContent = 'Connected';
                break;
            case 'disconnected':
                indicator.className = 'status-indicator disconnected';
                text.textContent = 'Disconnected';
                break;
            case 'error':
                indicator.className = 'status-indicator error';
                text.textContent = 'Connection Error';
                break;
        }
    }
    
    setupEventListeners() {
        // Close modal
        document.querySelector('.close').addEventListener('click', () => {
            document.getElementById('patientDetailModal').style.display = 'none';
            this.selectedPatientId = null;
        });
        
        // Request notification permission
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
    }
    
    startPeriodicRefresh() {
        // Refresh patient list every 30 seconds
        setInterval(() => {
            this.loadPatients();
        }, 30000);
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new TelemetryDashboard();
});
```

### Dashboard Styles

```css
/* static/styles.css */
:root {
    --primary-color: #007AFF;
    --success-color: #34C759;
    --warning-color: #FF9500;
    --danger-color: #FF3B30;
    --background: #F2F2F7;
    --card-background: #FFFFFF;
    --text-primary: #000000;
    --text-secondary: #8E8E93;
    --border-color: #C6C6C8;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background-color: var(--background);
    color: var(--text-primary);
    line-height: 1.6;
}

.dashboard-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
    padding: 20px;
    background-color: var(--card-background);
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.dashboard-header h1 {
    font-size: 28px;
    font-weight: 600;
    color: var(--primary-color);
}

.connection-status {
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: var(--text-secondary);
}

.status-indicator.connected {
    background-color: var(--success-color);
    animation: pulse 2s infinite;
}

.status-indicator.disconnected {
    background-color: var(--danger-color);
}

.status-indicator.error {
    background-color: var(--warning-color);
}

@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(52, 199, 89, 0.7);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(52, 199, 89, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(52, 199, 89, 0);
    }
}

.patient-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.patient-card {
    background-color: var(--card-background);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    cursor: pointer;
    transition: all 0.3s ease;
    border-left: 4px solid transparent;
}

.patient-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.patient-card.status-stable {
    border-left-color: var(--success-color);
}

.patient-card.status-warning {
    border-left-color: var(--warning-color);
}

.patient-card.status-critical {
    border-left-color: var(--danger-color);
    animation: critical-pulse 1s infinite;
}

@keyframes critical-pulse {
    0%, 100% {
        background-color: var(--card-background);
    }
    50% {
        background-color: rgba(255, 59, 48, 0.1);
    }
}

.patient-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.patient-header h3 {
    font-size: 18px;
    font-weight: 600;
}

.patient-age {
    color: var(--text-secondary);
    font-size: 14px;
}

.patient-metrics {
    display: flex;
    gap: 20px;
    margin-bottom: 15px;
}

.metric {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.metric-label {
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
}

.metric-value {
    font-size: 16px;
    font-weight: 600;
}

.risk-low {
    color: var(--success-color);
}

.risk-moderate {
    color: var(--warning-color);
}

.risk-high,
.risk-critical {
    color: var(--danger-color);
}

.patient-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 15px;
    border-top: 1px solid var(--border-color);
}

.last-update {
    font-size: 12px;
    color: var(--text-secondary);
}

.alert-badge {
    background-color: var(--danger-color);
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}

/* Modal Styles */
.modal {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.4);
    animation: fadeIn 0.3s;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.modal-content {
    background-color: var(--card-background);
    margin: 50px auto;
    padding: 30px;
    border-radius: 12px;
    width: 90%;
    max-width: 1200px;
    max-height: 90vh;
    overflow-y: auto;
    position: relative;
    animation: slideIn 0.3s;
}

@keyframes slideIn {
    from {
        transform: translateY(-50px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

.close {
    position: absolute;
    right: 20px;
    top: 20px;
    font-size: 28px;
    font-weight: bold;
    color: var(--text-secondary);
    cursor: pointer;
    transition: color 0.3s;
}

.close:hover {
    color: var(--text-primary);
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 20px;
    margin: 30px 0;
}

.metric-card {
    background-color: var(--background);
    padding: 20px;
    border-radius: 8px;
    text-align: center;
}

.metric-card h3 {
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 10px;
    text-transform: uppercase;
}

.metric-card .metric-value {
    font-size: 32px;
    font-weight: 600;
    color: var(--primary-color);
}

.metric-unit {
    font-size: 14px;
    color: var(--text-secondary);
    margin-top: 5px;
}

.charts-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 30px;
    margin: 30px 0;
}

.chart-wrapper {
    background-color: var(--background);
    padding: 20px;
    border-radius: 8px;
}

.alerts-section {
    margin-top: 30px;
}

.alerts-section h3 {
    margin-bottom: 15px;
    font-size: 18px;
}

.alert {
    padding: 15px;
    margin-bottom: 10px;
    border-radius: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.alert-warning {
    background-color: rgba(255, 149, 0, 0.1);
    border-left: 4px solid var(--warning-color);
}

.alert-critical {
    background-color: rgba(255, 59, 48, 0.1);
    border-left: 4px solid var(--danger-color);
}

.alert-time {
    font-size: 12px;
    color: var(--text-secondary);
}

.alert-message {
    flex: 1;
    margin: 0 20px;
}

.alert-acknowledge {
    background-color: var(--primary-color);
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.3s;
}

.alert-acknowledge:hover {
    background-color: #0051D5;
}

/* Responsive Design */
@media (max-width: 768px) {
    .patient-grid {
        grid-template-columns: 1fr;
    }
    
    .charts-container {
        grid-template-columns: 1fr;
    }
    
    .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .modal-content {
        width: 95%;
        margin: 20px auto;
        padding: 20px;
    }
}
```

## iOS App Integration

### Modified HealthKitManager

```swift
// HealthKitManager.swift - Modified for API integration
import Foundation
import HealthKit
import Combine

class HealthKitManager: ObservableObject {
    static let shared = HealthKitManager()
    
    private let healthStore = HKHealthStore()
    private let apiClient = TelemetryAPIClient()
    
    @Published var isAuthorized = false
    @Published var lastSyncTime: Date?
    @Published var syncStatus: SyncStatus = .idle
    
    enum SyncStatus {
        case idle
        case syncing
        case success
        case error(String)
    }
    
    // MARK: - API Integration
    
    func syncHealthData() async {
        await MainActor.run {
            self.syncStatus = .syncing
        }
        
        do {
            // Collect health data
            let healthData = try await collectHealthData()
            
            // Send to API
            let assessment = try await apiClient.submitHealthData(healthData)
            
            // Update local state
            await MainActor.run {
                self.lastSyncTime = Date()
                self.syncStatus = .success
                
                // Store assessment locally for offline access
                DataManager.shared.saveAssessment(assessment)
                
                // Check for critical alerts
                if let alert = assessment.alerts.first(where: { $0.severity == .critical }) {
                    self.showCriticalAlert(alert)
                }
            }
        } catch {
            await MainActor.run {
                self.syncStatus = .error(error.localizedDescription)
            }
        }
    }
    
    private func showCriticalAlert(_ alert: Alert) {
        // Show local notification
        let content = UNMutableNotificationContent()
        content.title = "Critical Health Alert"
        content.body = alert.message
        content.sound = .defaultCritical
        
        let request = UNNotificationRequest(
            identifier: alert.id,
            content: content,
            trigger: nil
        )
        
        UNUserNotificationCenter.current().add(request)
    }
}

// MARK: - API Client

class TelemetryAPIClient {
    private let baseURL = "https://api.telemetryhealth.com"
    private let apiKey = "your-api-key-here"  // Store securely in keychain
    private let session = URLSession.shared
    
    func submitHealthData(_ data: HealthKitData) async throws -> HealthAssessment {
        let url = URL(string: "\(baseURL)/api/health/data")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        
        let payload = HealthDataPayload(
            patientId: getUserId(),
            timestamp: Date(),
            metrics: data.toMetrics(),
            deviceInfo: getDeviceInfo()
        )
        
        request.httpBody = try JSONEncoder().encode(payload)
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw APIError.invalidResponse
        }
        
        return try JSONDecoder().decode(HealthAssessment.self, from: data)
    }
    
    private func getUserId() -> String {
        // Get or create unique user ID
        if let userId = UserDefaults.standard.string(forKey: "userId") {
            return userId
        } else {
            let newId = UUID().uuidString
            UserDefaults.standard.set(newId, forKey: "userId")
            return newId
        }
    }
    
    private func getDeviceInfo() -> DeviceInfo {
        return DeviceInfo(
            model: UIDevice.current.model,
            osVersion: UIDevice.current.systemVersion
        )
    }
}

// MARK: - Data Models

struct HealthDataPayload: Codable {
    let patientId: String
    let timestamp: Date
    let metrics: HealthMetrics
    let deviceInfo: DeviceInfo
}

struct HealthMetrics: Codable {
    let heartRate: Double
    let heartRateStd: Double
    let hrvMean: Double
    let pnn50: Double
    let respiratoryRate: Double
    let activityLevel: Double
    let sleepQuality: Double
}

struct DeviceInfo: Codable {
    let model: String
    let osVersion: String
}

enum APIError: Error {
    case invalidResponse
    case networkError
    case authenticationFailed
}
```

### Modified AIAnalysisView

```swift
// AIAnalysisView.swift - Modified for API integration
import SwiftUI

struct AIAnalysisView: View {
    @StateObject private var healthManager = HealthKitManager.shared
    @StateObject private var dataManager = DataManager.shared
    @State private var isLoading = false
    @State private var lastAssessment: HealthAssessment?
    @State private var syncTimer: Timer?
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Sync Status Header
                SyncStatusView(
                    status: healthManager.syncStatus,
                    lastSync: healthManager.lastSyncTime
                )
                
                // Loading State
                if isLoading {
                    ProgressView("Analyzing health data...")
                        .padding()
                }
                
                // Assessment Results
                if let assessment = lastAssessment {
                    // Rhythm Analysis
                    AnalysisCardView(
                        title: "Heart Rhythm",
                        status: assessment.predictions.rhythmStatus.prediction,
                        confidence: assessment.predictions.rhythmStatus.confidence,
                        icon: "heart.fill",
                        color: rhythmColor(for: assessment.predictions.rhythmStatus.prediction)
                    )
                    
                    // Risk Assessment
                    AnalysisCardView(
                        title: "Health Risk",
                        status: assessment.predictions.healthRisk.level,
                        confidence: assessment.predictions.healthRisk.confidence,
                        icon: "shield.fill",
                        color: riskColor(for: assessment.predictions.healthRisk.level)
                    )
                    
                    // HRV Pattern
                    AnalysisCardView(
                        title: "HRV Pattern",
                        status: assessment.predictions.hrvPattern.pattern,
                        confidence: assessment.predictions.hrvPattern.confidence,
                        icon: "waveform.path.ecg",
                        color: .green
                    )
                    
                    // Fitness Level
                    FitnessCardView(
                        vo2max: assessment.predictions.fitnessLevel.vo2max,
                        cardiovascularAge: assessment.predictions.fitnessLevel.cardiovascularAge,
                        fitnessCategory: assessment.predictions.fitnessLevel.fitnessCategory
                    )
                    
                    // Alerts
                    if !assessment.alerts.isEmpty {
                        AlertsView(alerts: assessment.alerts)
                    }
                    
                    // Recommendations
                    if !assessment.recommendations.isEmpty {
                        RecommendationsView(recommendations: assessment.recommendations)
                    }
                } else {
                    // Empty State
                    EmptyStateView(
                        icon: "heart.text.square",
                        title: "No Assessment Available",
                        description: "Tap refresh to analyze your health data"
                    )
                }
                
                // Manual Refresh Button
                Button(action: {
                    Task {
                        await refreshHealthData()
                    }
                }) {
                    Label("Refresh Analysis", systemImage: "arrow.clockwise")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isLoading)
            }
            .padding()
        }
        .navigationTitle("AI Analysis")
        .task {
            await loadInitialData()
            startAutoRefresh()
        }
        .onDisappear {
            stopAutoRefresh()
        }
    }
    
    // MARK: - Data Loading
    
    private func loadInitialData() async {
        // Load last cached assessment
        if let cached = dataManager.getLatestAssessment() {
            await MainActor.run {
                self.lastAssessment = cached
            }
        }
        
        // Fetch fresh data
        await refreshHealthData()
    }
    
    private func refreshHealthData() async {
        await MainActor.run {
            isLoading = true
        }
        
        await healthManager.syncHealthData()
        
        await MainActor.run {
            isLoading = false
            
            // Update with latest assessment
            if let latest = dataManager.getLatestAssessment() {
                self.lastAssessment = latest
            }
        }
    }
    
    // MARK: - Auto Refresh
    
    private func startAutoRefresh() {
        syncTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { _ in
            Task {
                await refreshHealthData()
            }
        }
    }
    
    private func stopAutoRefresh() {
        syncTimer?.invalidate()
        syncTimer = nil
    }
    
    // MARK: - Helper Functions
    
    private func rhythmColor(for status: String) -> Color {
        switch status.lowercased() {
        case "normal":
            return .green
        case "irregular":
            return .orange
        default:
            return .gray
        }
    }
    
    private func riskColor(for level: String) -> Color {
        switch level.lowercased() {
        case "low":
            return .green
        case "moderate":
            return .yellow
        case "high":
            return .orange
        case "critical":
            return .red
        default:
            return .gray
        }
    }
}

// MARK: - Supporting Views

struct SyncStatusView: View {
    let status: HealthKitManager.SyncStatus
    let lastSync: Date?
    
    var body: some View {
        HStack {
            statusIcon
            VStack(alignment: .leading) {
                Text(statusText)
                    .font(.headline)
                if let lastSync = lastSync {
                    Text("Last synced: \(lastSync, style: .relative) ago")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            Spacer()
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    @ViewBuilder
    private var statusIcon: some View {
        switch status {
        case .idle:
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
        case .syncing:
            ProgressView()
        case .success:
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
        case .error:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)
        }
    }
    
    private var statusText: String {
        switch status {
        case .idle:
            return "Ready"
        case .syncing:
            return "Syncing..."
        case .success:
            return "Synced Successfully"
        case .error(let message):
            return "Error: \(message)"
        }
    }
}
```

## Security (design notes)

> These are **design intentions** for a demo backend, not implemented or audited guarantees. No
> HIPAA/GDPR compliance is claimed. The system processes synthetic data only.

### Planned MVP Security Measures

1. **API Authentication**
   - API key authentication for MVP
   - Rate limiting (100 requests/minute per key)
   - IP whitelisting option

2. **Data Encryption**
   - TLS 1.3 for all API communications
   - Database encryption at rest
   - Sensitive data field encryption

3. **Access Control**
   - Basic role-based access (admin, viewer)
   - Session management with timeout
   - Audit logging for all data access

### MVP Security (What We're Implementing)

- ✅ Encrypted data transmission (TLS via Railway)
- ✅ Basic API key authentication
- ✅ Simple audit logging to database
- ✅ Automatic backups (Railway PostgreSQL)
- ✅ Basic password protection for dashboard

## Deployment Strategy

### MVP Deployment (Railway.app)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# railway.json
{
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "numReplicas": 1,
    "healthcheckPath": "/health",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

### Environment Configuration

```bash
# .env.example
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/telemetryhealth
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# API Configuration
API_KEY=generate-secure-key-here
SECRET_KEY=generate-secret-key-here
ENVIRONMENT=production
DEBUG=false

# ML Models
MODEL_DIR=/app/ml_models
MODEL_CACHE_SIZE=4

# Dashboard
DASHBOARD_USER=admin
DASHBOARD_PASSWORD=secure-password-here

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```


## Implementation Roadmap

### Week 1: Backend Development
- [ ] Day 1-2: Project setup and database schema
- [ ] Day 3-4: FastAPI endpoints and ML model integration
- [ ] Day 5: WebSocket implementation
- [ ] Day 6-7: Testing and error handling

### Week 2: Dashboard & Integration
- [ ] Day 8-9: Dashboard UI development
- [ ] Day 10-11: iOS app API integration
- [ ] Day 12: Deployment setup
- [ ] Day 13-14: End-to-end testing and bug fixes


## Cost Analysis

### MVP Costs (10 users)
| Service | Provider | Monthly Cost |
|---------|----------|--------------|
| Backend Hosting | Railway.app | $5-20 |
| Database | Railway PostgreSQL | $5-10 |
| Domain | Namecheap | $1 |
| SSL | Railway (included) | $0 |
| **Total** | | **$11-31/month** |

## Next Steps

### Immediate Actions (Week 1)
1. **Set up development environment**
   ```bash
   # Clone and setup
   git clone https://github.com/yourorg/telemetryhealth-backend
   cd telemetryhealth-backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Initialize database**
   ```bash
   # Create PostgreSQL database
   createdb telemetryhealth
   
   # Run migrations
   alembic upgrade head
   ```

3. **Start development**
   ```bash
   # Run FastAPI server
   uvicorn app.main:app --reload --port 8000
   ```

### Repository Structure (Option A: Separate Repos - Recommended)
Create one new repository:

1. **telemetryhealth-backend**
   - FastAPI application with ML models
   - Dashboard static files (HTML/JS/CSS)
   - Database schemas
   - Deployment configurations (Dockerfile, railway.json)

2. **TelemetryHealthCare** (existing iOS repo)
   - Keep iOS app in current repository
   - Update to use API endpoints

### Repository Structure (Option B: Monorepo)
Keep everything in the current repository:

```
TelemetryHealthCare/
├── ios/                    # Move existing iOS code here
│   ├── TelemetryHealthCare/
│   ├── TelemetryHealthCare.xcodeproj
│   └── ...
├── backend/                # New backend code
│   ├── app/
│   ├── ml_models/
│   ├── static/            # Dashboard files
│   ├── requirements.txt
│   └── Dockerfile
└── README.md
```

### Development Workflow
1. **Backend First**: Complete API endpoints
2. **iOS Integration**: Modify app to use API
3. **Dashboard MVP**: Simple HTML dashboard
4. **Testing**: End-to-end testing
5. **Deploy**: Railway.app deployment

## Conclusion

This MVP architecture provides everything needed for a rapid deployment to test with ~10 users:

Key benefits:
- **Speed**: 2-week deployment timeline
- **Cost**: $11-31/month total infrastructure cost
- **Simplicity**: Single FastAPI backend serving both API and dashboard
- **Reuses existing models**: the same models trained on synthetic data (not validated on real data)
- **Easy Testing**: Simple deployment to Railway.app with one command

The monolithic approach minimizes complexity while providing all essential features for testing and demonstration.