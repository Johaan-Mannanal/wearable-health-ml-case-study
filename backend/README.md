# TelemetryHealthCare Backend

> ⚠️ Synthetic-data research project — not a medical device. See the root README.md and MODEL_CARD.md.

Work-in-progress FastAPI backend that serves the project's ML models (trained on **synthetic** data)
plus a small demo dashboard. It is a portfolio/learning exercise, not a healthcare product; "patient"
in the API just means "app user", and no real patient data is involved.

## Quick Start

### 1. Install Dependencies
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Up Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Run Locally
```bash
uvicorn app.main:app --reload --port 8000
```

Visit:
- API: http://localhost:8000/api/status
- Dashboard: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Deployment to Railway

1. Push to GitHub
2. Connect repo to Railway
3. Set root directory to `/backend` in Railway settings
4. Add PostgreSQL database
5. Deploy!

## API Endpoints

- `POST /api/health/data` - Submit health data from iOS app
- `GET /api/ml/analyze` - Run ML analysis
- `GET /api/dashboard/patients` - Get patient list
- `WS /ws/dashboard` - WebSocket for real-time updates

## Testing

```bash
# Test API endpoint
curl -X POST http://localhost:8000/api/health/data \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-api-key-here" \
  -d '{
    "patient_id": "test-user-001",
    "metrics": {
      "heart_rate": 72,
      "heart_rate_std": 5.2,
      "hrv_mean": 45,
      "pnn50": 0.25,
      "respiratory_rate": 16,
      "activity_level": 150,
      "sleep_quality": 0.85
    }
  }'
```