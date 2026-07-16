# Model Testing Report - Synthetic Data

> ⚠️ Synthetic-data research project — not a medical device. See the root README.md and MODEL_CARD.md.

## Executive Summary
The models were exercised against **synthetic** wearable-style inputs (no real Apple Watch or
patient data). They behave as expected on that synthetic data. This is a smoke/behavior check of the
pipeline — **not** evidence of real-world accuracy or readiness for any deployment. Verified metrics:
[`results/metrics.csv`](../results/metrics.csv).

## Test Results

### 1. SVM Heart Rhythm Model
**Purpose:** classify a synthetic Normal/Irregular rhythm label (not detection)

**Behavior on synthetic inputs:**
- **Verified test-split accuracy:** 93.9% (ROC-AUC 0.987) — see `results/metrics.csv`
- Assigns the "Normal" synthetic label to low-variability inputs
- Assigns the "Irregular" synthetic label to high-variability inputs
- **Edge cases:** high-intensity-exercise-like inputs sometimes get the "Irregular" label (expected)

**Illustrative synthetic scenarios** (hand-picked inputs; labels/confidences are model outputs on
synthetic data, not clinical findings):
| Scenario (synthetic) | HR | Std Dev | Label | Model confidence |
|----------|-----|---------|---------|------------|
| Resting-like | 65 | 5.2 | Normal | 99.8% |
| High-variability | 88 | 18.5 | Irregular | 98.5% |
| Athlete-rest-like | 45 | 3.8 | Normal | 95.7% |
| High-intensity-like | 165 | 4.2 | Irregular | 94.5% |

### 2. GBM Health Risk Model
**Purpose:** Assess overall health risk without blood pressure

**Test Performance:**
- **Verified test accuracy:** 99.4% / ROC-AUC 1.000 (synthetic, separable-by-design)
- Integrates activity, sleep, and stress synthetic features
- Produces a Low/High synthetic "risk" label (not a health assessment)

**Key Features Tested:**
- Heart rate and HRV
- Respiratory rate
- Activity levels
- Sleep quality
- Stress indicators

### 3. Neural Network HRV Model
**Purpose:** classify synthetic HRV pattern labels

**Test Performance:**
- **Verified test accuracy:** 99.0% (synthetic, 4-class macro-averaged)
- 4 synthetic classes labeled: normal, afib, bradycardia, tachycardia
- Separates the synthetic classes well (they are separable by design)

**Model outputs on synthetic class examples** (confidence = model output on synthetic data, not clinical certainty):
- "normal" class: 90.8%
- "afib" class: 100%
- "bradycardia" class: 99.9%
- "tachycardia" class: 99.4%

## Continuous Monitoring Simulation

### 24-Hour Simulation Results:
- **Total readings:** 1,440 (1 per minute)
- **Patterns detected:** Sleep, rest, activity, exercise
- **Circadian rhythm:** Models correctly adapted to time-of-day variations

### 1-Hour Real-Time Test:
```
00:00-10:00  Sitting/Working  → Normal rhythm, Low risk
10:00-20:00  Walking         → Slight irregularity (normal for activity)
30:00-40:00  Stairs          → Elevated HR, appropriate response
40:00-60:00  Rest            → Return to normal baseline
```

## Apple Watch Compatibility

### Fully Compatible Features:
- ✅ Heart rate (continuous)
- ✅ Heart rate variability
- ✅ Activity detection
- ✅ Sleep monitoring
- ✅ Respiratory rate

### Data Pipeline Ready:
- HealthKit integration documented
- Feature extraction implemented
- Real-time processing capable

## Observations (on synthetic data only)

### What the pipeline does well
1. Cleanly separates the synthetic classes (which are separable by design)
2. Uses only wearable-style features (no blood pressure input)
3. Fast inference; small footprint

### Limitations
1. **Synthetic data only** — nothing here transfers to real physiological signals
2. Outputs are labels/probabilities, **not** diagnoses
3. No validation against any real or public dataset

## Next steps (engineering, research-only)

1. Continue wiring the demo pipeline (`healthkit_data_processor.py`) and app UI
2. Keep thresholds clearly marked as illustrative
3. Before any real-data work: obtain a consented/public dataset, use group-aware validation, and
   compare against trivial baselines (see `RESEARCH_NOTES.md`)

## Conclusion
The models behave as expected on synthetic data. This confirms the pipeline runs end-to-end — it does
**not** establish real-world accuracy, and the project is **not** ready for HealthKit deployment,
app release, or any clinical use.