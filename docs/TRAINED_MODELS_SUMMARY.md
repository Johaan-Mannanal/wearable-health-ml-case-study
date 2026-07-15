# Trained ML Models Summary

> ⚠️ Synthetic-data research project — not a medical device. See the root README.md and MODEL_CARD.md.

## Overview
The models below are trained and evaluated **entirely on synthetic data**. Input feature sets are
designed to resemble what a watch could expose via HealthKit, but no real Apple Watch or patient
data was used. Metrics are reproduced from the shipped code — source of truth:
[`results/metrics.csv`](../results/metrics.csv). Scores are high mainly because the synthetic
classes are separable by design.

## Model Performance Summary (synthetic data — verified)

### 1. SVM ensemble - Rhythm Classification
- **Accuracy**: 93.9% / **ROC-AUC**: 0.987
- **Model File**: `svm_heart_rhythm_model.pkl`
- **Purpose**: binary classification of a synthetic Normal/Irregular label (not diagnosis)
- **Input Features**: mean_heart_rate, std_heart_rate, pnn50

### 2. GBM - Health-Risk Classification
- **Accuracy**: 99.4% / **ROC-AUC**: 1.000
- **Model File**: `gbm_health_risk_model.pkl`
- **Purpose**: binary Low/High synthetic "risk" label, using only wearable-style features (no BP)
- **Input Features**: average_heart_rate, hrv_mean, respiratory_rate, activity_level,
  sleep_quality, stress_indicator, hr_hrv_ratio, recovery_score

### 3. Neural Network (MLP) - HRV Pattern Classification
- **Accuracy**: 99.0%
- **Model File**: `hrv_pattern_nn_model.pkl`
- **Purpose**: 4-class classification of synthetic HRV labels (normal / afib / bradycardia / tachycardia)
- **Input**: 13 features extracted from 50 HR readings

### 4. Cardio regression ensemble
- **Held-out R²**: fitness 0.918, VO₂max 0.563, cardiovascular age 0.970
- **Model Files**: `cardiovascular_*_model.pkl`
- **Purpose**: regress synthetic fitness/VO₂max/cardiovascular-age targets

## Next Steps for Real Data Integration

### 1. Test with Synthetic Apple Watch Data
Run the test scripts to verify model performance:
```bash
python3 scripts/testing/test_improved_model.py  # For SVM
```

### 2. HealthKit Integration
Use the provided `scripts/utilities/healthkit_data_processor.py` for real data processing.

### 3. Core ML Conversion
Convert models for iOS deployment:
- SVM and GBM: Use coremltools with scikit-learn converter
- Neural Network: Direct conversion supported

### 4. iOS App Integration
Models are ready to be integrated into the Swift app with HealthKit data pipeline.

## Model Files
- `ML_Models/svm_heart_rhythm_model.pkl` - Heart rhythm classifier
- `ML_Models/gbm_health_risk_model.pkl` - Health risk assessment
- `ML_Models/hrv_pattern_nn_model.pkl` - HRV pattern analyzer
- Metadata JSON files in `ML_Models/` with full specifications

All models are research/education artifacts trained on synthetic data — not for deployment or any
health use.