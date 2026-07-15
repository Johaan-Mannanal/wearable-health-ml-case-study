# Wearable Health Telemetry — ML Research (Synthetic Data)

> **Exploratory machine-learning study of whether standard models can classify labeled
> patterns in *synthetic* wearable-style cardiovascular signals (heart rate & heart-rate
> variability).** Unpublished student research. **Not a medical device.**

[![Made with Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-orange)](https://scikit-learn.org/)
[![Data: synthetic](https://img.shields.io/badge/data-synthetic-lightgrey)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What this is (in one sentence)

This project explores whether machine-learning models can separate **labeled patterns in
wearable cardiovascular telemetry** — using **synthetically generated** heart-rate and
heart-rate-variability features designed to resemble what an Apple Watch exposes through
HealthKit.

## What this is *not*

It does **not** diagnose disease, detect heart conditions, or provide medical advice, and it
has **no clinical validation**. All results come from **synthetic data**, so the scores below
measure how separable the generated classes are — **not** real-world performance.

## Research question

> Can common supervised-learning models (an SVM ensemble, gradient boosting, a small neural
> network, and a regression ensemble) recover the class/label structure embedded in
> synthetically generated wearable-style cardiovascular features, and how do they compare?

## Motivation

Wearables expose rich cardiovascular signals (heart rate, HRV, respiratory rate, activity).
As a learning exercise, this project builds an **end-to-end, reproducible ML pipeline** —
data generation → feature engineering → training → held-out evaluation → visualization —
around those signal types, while being explicit about the limits of using synthetic data.

## ⚠️ Important medical disclaimer

This is a **student research prototype for educational purposes only**. It is **not** a
medical device, is **not** FDA-cleared, has **not** been clinically validated, and must
**never** be used to make health decisions. Labels such as "normal / irregular / afib /
bradycardia / tachycardia" refer to **synthetic classes**, not diagnoses. Consult a qualified
clinician for any medical concern.

## Dataset overview

**There is no real dataset.** Every sample is generated in code
([`src/data_processing.py`](src/data_processing.py)) from NumPy distributions with a fixed
seed (`42`). Because the classes are drawn from deliberately different distributions, they are
separable by construction. See [`data/README.md`](data/README.md) for schemas and
[`DATA_ACCESS.md`](DATA_ACCESS.md) for why no real data is used.

## Repository structure

```
.
├── README.md                 # this file
├── LICENSE                   # MIT
├── requirements.txt          # ML pipeline dependencies
├── MODEL_CARD.md             # intended use, inputs/outputs, limitations, risks
├── RESEARCH_NOTES.md         # honest notes: what worked / didn't / weaknesses
├── DATA_ACCESS.md            # why the data is synthetic + how to regenerate
├── configs/default.yaml      # reproducible run configuration
├── src/                      # importable pipeline (data, features, train, evaluate, viz)
├── notebooks/                # exploratory notebooks
├── tests/                    # pytest: data processing + metric correctness
├── results/                  # verified metrics.csv + explanation
├── assets/                   # generated charts (comparison, confusion, distribution, workflow)
├── models/                   # trained pipelines (.pkl), regenerable via src.train
├── backend/                  # FastAPI service that serves the models (full-stack extra)
├── TelemetryHealthCare/      # SwiftUI iOS app (full-stack extra, work-in-progress)
└── docs/ , guides/           # supplementary documentation
```

> This repo is intentionally **full-stack**: the ML research is the centerpiece, and a FastAPI
> backend and a SwiftUI iOS app are included as applied extras. The app and backend are
> works in progress (see `docs/` and `guides/`).

## Methodology

1. **Synthetic data generation** — per-class NumPy distributions with physiological clipping.
2. **Feature engineering** — model-specific feature sets; two derived features for the
   health-risk model (`hr_hrv_ratio`, `recovery_score`).
3. **Split then preprocess** — stratified train/test split, with scaling done **inside a
   scikit-learn `Pipeline`** so it is fit on training data only (no leakage).
4. **Training** — fixed seeds; one command trains any/all models.
5. **Evaluation** — held-out test metrics + 5-fold cross-validation.
6. **Visualization** — honest charts (bars start at 0).

## Models compared

| Model | Task | Algorithm |
|-------|------|-----------|
| SVM ensemble | Binary rhythm (Normal vs Irregular) | Soft-voting SVM + Logistic Regression + Random Forest, RobustScaler |
| Gradient Boosting | Binary health-risk (Low vs High) | `GradientBoostingClassifier`, StandardScaler |
| Neural network | 4-class HRV (normal/afib/bradycardia/tachycardia) | `MLPClassifier` (64→32→16), StandardScaler |
| Cardio regression | Fitness / VO₂max / cardiovascular age | Voting ensemble (RandomForest + GradientBoosting + XGBoost) |

## Evaluation approach

Metrics are computed on a **held-out 20% test split** and cross-checked with **5-fold CV**.
For the imbalanced binary tasks, accuracy alone is misleading, so precision, recall, F1, ROC-AUC,
and the **confusion matrix** are all reported — with attention to **false negatives** (missed
"irregular"/"high-risk" cases), which matter most in any health-adjacent setting.

## Verified results

Reproduced from the shipped code (`python -m src.evaluate --model all`) on synthetic data with
seed `42`. These **replace** the older, inconsistent numbers in previous documentation.
Machine-readable source of truth: [`results/metrics.csv`](results/metrics.csv).

**Classification**

| Model | Task | Accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV | Train / Test |
|-------|------|----------|-----------|--------|----|---------|-----------|--------------|
| SVM ensemble | Binary rhythm | **93.9%** | 0.910 | 0.940 | 0.925 | 0.987 | 93.2% ± 0.9% | 4000 / 1000 |
| Gradient Boosting | Binary health-risk | **99.4%** | 0.994 | 0.990 | 0.992 | 1.000 | 99.5% ± 0.2% | 8000 / 2000 |
| Neural network (MLP) | 4-class HRV | **99.0%** | 0.990 | 0.990 | 0.990 | — (multiclass) | 99.5% ± 0.3% | 3200 / 800 |

*Precision/recall/F1 are macro-averaged for the 4-class model.*

**Regression (cardiovascular targets), held-out test**

| Target | R² | MAE | RMSE | 5-fold CV R² |
|--------|----|----|------|--------------|
| Fitness level | 0.918 | 3.51 | 4.40 | 0.918 ± 0.012 |
| VO₂max | **0.563** | 5.83 | 7.54 | 0.598 ± 0.034 |
| Cardiovascular age | 0.970 | 2.72 | 3.47 | 0.967 ± 0.003 |

Confusion matrices and per-class breakdowns are in [`assets/`](assets/).

> ⚠️ **How to read these.** The classification scores are high **because the synthetic classes
> are separable by design** — they verify the pipeline works, not that the models predict real
> cardiovascular states. The weaker VO₂max R² (0.56) is a useful reminder that not every target
> is trivially recoverable even from synthetic data.

## Reproduction instructions

```bash
# 1. Environment (Python 3.11+). On macOS, XGBoost needs libomp: brew install libomp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Train (any of: svm | gbm | nn | cardio | all)
python -m src.train --config configs/default.yaml --model all

# 3. Evaluate -> writes results/metrics.csv
python -m src.evaluate --config configs/default.yaml --model all

# 4. Tests
pytest tests/ -q
```

All data is generated on the fly, so no downloads are required.

## Limitations

- **Synthetic data only** — no real or public benchmark; results do not transfer to reality.
- **No participant structure** — cannot assess cross-individual generalization.
- **Separable-by-design classes** inflate the classification metrics.
- **Class imbalance** on binary tasks — accuracy flatters the majority class.
- The iOS app and backend are **works in progress**, not production systems.

## Ethical and privacy considerations

- No human/patient data was collected or used; nothing to de-identify.
- Health-adjacent ML must avoid overclaiming — hence the disclaimers and the synthetic framing.
- Using real wearable data would require consent, an ethics/IRB review, a lawful basis for
  processing health data, and group-aware splitting (see [`DATA_ACCESS.md`](DATA_ACCESS.md)).

## Future work

Replace synthetic data with a consented or public benchmark; add group-aware CV; report
confidence intervals and calibration; compare against trivial baselines; and add external
validation before making any performance claim. See [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md).

## My contribution

<!-- TODO: Johaan to confirm/personalize the exact split of work. -->
Designed and implemented the end-to-end ML pipeline (synthetic data generation, feature
engineering, model training/evaluation, and visualization), plus the FastAPI backend and
SwiftUI iOS app scaffolding. This project had a second contributor (see Acknowledgments);
please edit this section to describe who did what.

## Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.ai/),
  NumPy, pandas, and Matplotlib.
- Co-contributor: **Yash Piratla** (early ML implementation). <!-- TODO: confirm attribution -->

## Contact

<!-- TODO: Johaan to add preferred public contact (GitHub profile or a role email). -->
Maintainer: [@Johaan-Mannanal](https://github.com/Johaan-Mannanal)

---

*Educational research project. Not affiliated with Apple. "Apple Watch" and "HealthKit" are
trademarks of Apple Inc., referenced only to describe the signal types this synthetic data
imitates.*
