# Technical Documentation

A concise technical companion to the README. It records the methodology and per-model detail
without duplicating the model card or research notes.

## Research question
Can standard supervised-learning models separate labeled patterns in **synthetic** wearable-style
cardiovascular features (heart rate and heart-rate variability), and how do they compare?

## Data
All data is generated in code (`src/data_processing.py`) with a fixed seed. There is no real
Apple Watch, HealthKit, or patient data. Classes are drawn from deliberately different
distributions with physiological clipping, so they are separable by construction, high scores
verify the pipeline, not real-world performance. Schemas are in
[`data/README.md`](../data/README.md).

## Methodology
- **Split then preprocess:** stratified train/test split, with scaling done inside a
  scikit-learn `Pipeline` fit on training data only (no leakage).
- **Fixed seeds** (`random_state=42`) for reproducibility.
- **Evaluation:** held-out test metrics plus 5-fold cross-validation.
- Metrics computed with standard scikit-learn functions; macro-averaging for the 4-class model.

## Models
| Model | Task | Algorithm | Key features |
|-------|------|-----------|--------------|
| SVM ensemble | Binary rhythm (Normal / Irregular) | Soft-voting SVM + Logistic Regression + Random Forest, RobustScaler | mean HR, HR std, pNN50 |
| Gradient boosting | Binary health-risk (Low / High) | `GradientBoostingClassifier`, StandardScaler | 6 base + 2 derived (hr_hrv_ratio, recovery_score) |
| Neural network | 4-class HRV (normal/afib/bradycardia/tachycardia) | `MLPClassifier` (64→32→16), StandardScaler | 13 HRV summary features |
| Regression ensemble | Fitness / VO2max / cardiovascular age | RandomForest + GradientBoosting + XGBoost | age, resting HR, HRR, RMSSD |

Model construction and hyperparameters live in `src/train.py`; feature sets in
`src/feature_engineering.py`.

## Verified results
Reproduce with `python -m src.evaluate --model all` (writes
[`results/metrics.csv`](../results/metrics.csv)). Classification metrics are macro-averaged for
the 4-class model.

| Model | Accuracy | ROC-AUC | 5-fold CV | Train/Test |
|-------|----------|---------|-----------|------------|
| SVM ensemble | 0.939 | 0.987 | 0.932 ± 0.009 | 4000 / 1000 |
| Gradient boosting | 0.994 | 1.000 | 0.995 ± 0.002 | 8000 / 2000 |
| Neural network | 0.990 | N/A | 0.995 ± 0.003 | 3200 / 800 |

Regression (held-out R²): fitness **0.918**, VO2max **0.563**, cardiovascular age **0.970**. The
weaker VO2max result is a useful reminder that not every target is trivially recoverable even
from synthetic data. Confusion matrices and charts are in [`assets/`](../assets/).

## Illustrative application logic (non-diagnostic)
The backend and iOS app apply simple threshold rules to model outputs (e.g., flagging values
outside a typical range) purely to demonstrate how results could surface in a UI. These are
**illustrative wellness cues on synthetic data**, not diagnoses, and are labeled as such in the
code.

## Limitations
- Synthetic-only data with no participant structure; results do not transfer to real signals and
  say nothing about cross-individual generalization.
- Separable-by-design classes inflate classification scores.
- No external validation against a real or public benchmark.
- Not a clinical tool; see [`MODEL_CARD.md`](../MODEL_CARD.md) for out-of-scope uses and risks.

## What would make this publication-quality
A consented or public benchmark dataset, group-aware (participant-level) splitting, confidence
intervals and calibration, comparison against trivial baselines, and independent validation.
Details in [`RESEARCH_NOTES.md`](../RESEARCH_NOTES.md).
