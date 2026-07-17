# Architecture

> Synthetic-data research project: not a medical device. See the [README](README.md) and
> [MODEL_CARD.md](MODEL_CARD.md).

This repository is full-stack: the ML research is the centerpiece, with a FastAPI backend and a
SwiftUI iOS app as applied extras (both work-in-progress). This document is a high-level map;
the implementation lives in the code, not pasted here.

## System overview

```
 iOS app (SwiftUI)                 Backend (FastAPI)                ML pipeline (Python)
 -----------------                 -----------------                --------------------
 HealthKit-style input   --HTTP--> REST API + WebSocket   --loads--> scikit-learn models
 on-device display                 model inference                   (trained offline on
 (charts, disclaimers)   <-------   alerts / results        results  synthetic data)
```

All model inputs are **synthetic** (generated in `src/data_processing.py`); no real health data
flows through any layer.

## Components

**ML pipeline (`src/`, `scripts/`)**, the maintained, tested core.
- Synthetic data generation, feature engineering, training, evaluation, visualization.
- Four modeling tasks: SVM ensemble (rhythm), gradient boosting (risk), MLP (4-class HRV), and a
  regression ensemble (fitness / VO2max / cardiovascular age).
- One-command CLIs: `python -m src.train` / `python -m src.evaluate`; results in
  [`results/metrics.csv`](results/metrics.csv).

**Backend (`backend/`)**, FastAPI service that loads the trained `.pkl` models and exposes
inference over REST plus a WebSocket for streaming updates. Threshold-based alerts are
illustrative and non-diagnostic. Not a production or clinical system.

**iOS app (`TelemetryHealthCare/`)**, SwiftUI client that collects HealthKit-style metrics,
calls the backend, and displays trends with a prominent medical disclaimer. Work-in-progress.

## Technology stack

| Layer | Technology |
|-------|-----------|
| ML | Python, scikit-learn, XGBoost, NumPy, pandas, Matplotlib |
| Backend | FastAPI, Uvicorn, Pydantic, SQLAlchemy, WebSockets |
| Data | PostgreSQL (backend), synthetic in-memory for ML |
| iOS | Swift, SwiftUI, HealthKit, Core Data |
| Deploy | Docker, Railway |
| Tests | pytest (ML), XCTest (iOS) |

## Data flow (research pipeline)
1. Generate synthetic samples with physiological clipping (`src/data_processing.py`).
2. Engineer model-specific features; split into train/test **before** scaling.
3. Fit inside a scikit-learn `Pipeline` (scaler + estimator), no leakage.
4. Evaluate on a held-out split plus 5-fold cross-validation.
5. Persist models (`.pkl`) and metrics (`results/metrics.csv`); the backend loads the models.

## API surface (backend)
A small REST surface for health metrics and ML inference, plus a WebSocket for live updates.
Auth is a simple API key for the MVP. Exact routes and schemas live in `backend/app/`; they are
intentionally not duplicated here.

## Deployment
Containerized (`Dockerfile`) and deployable to Railway (`railway.json`). Configuration is via
environment variables; see [`backend/.env.example`](backend/.env.example). Runtime secrets are
never committed.

## Design decisions & honest limitations
- **Synthetic-only data** keeps the project free of privacy/licensing constraints, but the
  metrics measure separability of generated distributions, not real-world performance.
- Preprocessing is fit on training data only (leakage-safe); seeds are fixed for reproducibility.
- No participant structure exists, so nothing can be claimed about cross-individual generalization.
- The backend and iOS app show how the models could be served; they are not audited,
  HIPAA-compliant, or production-hardened.

For research methodology and results, see [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) and
[`docs/RESEARCH_DOCUMENTATION.md`](docs/RESEARCH_DOCUMENTATION.md).
