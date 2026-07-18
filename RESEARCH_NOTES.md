# Research Notes

Honest, first-person notes on what this project actually is, what worked, what didn't, and
what would have to change before any of it counts as real research.

## What was investigated

Whether standard machine-learning classifiers and regressors can separate **labeled
patterns in wearable-style cardiovascular signals** (heart rate, heart-rate variability,
and derived metrics), using features that would in principle be obtainable from an Apple
Watch via HealthKit. Four modeling tasks were built:

1. **Heart-rhythm classification** (binary: Normal vs Irregular): SVM + Logistic
   Regression + Random Forest soft-voting ensemble.
2. **Health-risk classification** (binary: Low vs High): Gradient Boosting.
3. **HRV-pattern classification** (4-class: normal / afib / bradycardia / tachycardia) -
   a multi-layer perceptron (scikit-learn `MLPClassifier`).
4. **Cardiovascular-fitness regression** (fitness level, VO₂max, cardiovascular age) -
   a Random Forest + Gradient Boosting + XGBoost voting ensemble.

## The single most important caveat

**All data is synthetic.** Every sample is drawn from NumPy distributions that were
hand-designed per class (see `src/data_processing.py`). The classes were intentionally
generated from *different* distributions, so the high accuracies (verified in
`results/metrics.csv`) largely measure **how separable the synthetic generator made the
classes**, not whether the models would work on real physiological data.

Concretely: the Gradient Boosting task reaches ~99% accuracy and ~1.00 AUC because
low-risk and high-risk profiles were generated from well-separated distributions across
eight features. That is a property of the data generator, not evidence of real predictive
power.

## What worked

- **Clean, reproducible pipelines.** Preprocessing (scalers) is inside scikit-learn
  `Pipeline` objects and fit only on the training split, so there is **no scaler leakage**.
- **Fixed seeds (`random_state=42`)** make every result reproducible.
- **The refactored code runs end-to-end** via `python -m src.train` / `python -m src.evaluate`,
  and the metrics reproduce (with small differences due to newer library versions, see below).
- Standard metrics (accuracy, precision, recall, F1, ROC-AUC, R²) are computed correctly on a
  held-out test split.

## What did not work / what is misleading

- **The numbers do not transfer to reality.** High accuracy on separable synthetic data says
  nothing about real Apple Watch performance. This was the central weakness.
- **The original saved metadata numbers did not all reproduce exactly.** The maintained
  pipeline evaluates the SVM at 93.9% and the GBM at 99.4% (`results/metrics.csv`), while the
  original saved metadata claimed 92.4% and 99.35%. Small shifts are expected across library
  versions; the lesson is that metrics should not be quoted to three decimals as if fixed.
- **The HRV neural-network training script did not run to completion as committed**: it
  referenced an undefined module-scope variable `n_samples` (twice), raising `NameError`
  before finishing. The model still exists because an earlier version was run; the committed
  script was broken. This is fixed in the refactor.
- **Documentation drift.** Earlier documentation contained numbers, clinical claims, and
  deployment statistics that the code does not support and that contradicted each other across
  files. Those documents were removed; `results/metrics.csv` is the single source of truth.

## Methodological weaknesses

- **Synthetic-only data** with **no participant structure**, so nothing can be said about
  cross-individual generalization.
- **Accuracy-first framing** on **imbalanced** binary tasks (e.g., 60/40): accuracy flatters
  the majority class; precision/recall/AUC and the confusion matrix matter more, especially for
  anything health-adjacent where false negatives are costly.
- **No external validation** against any real or public dataset (e.g., MIT-BIH / PhysioNet).
- The cardiovascular-regression targets are themselves generated from the same latent variables
  used as features, which can inflate R².

## Questions that remain unanswered

- Would any of these models retain signal on **real** HRV/heart-rate data?
- How would performance degrade under realistic sensor noise, missingness, and motion artifacts?
- Are the four "conditions" (normal/afib/brady/tachy) separable from features an Apple Watch can
  actually derive, as opposed to from the idealized 13 features used here?

## What would be needed before this is publication-quality research

1. Replace synthetic data with a **real, ethically sourced, consented** dataset (or a recognized
   public benchmark), and cite it.
2. Use **group-aware splitting** (`GroupKFold`) so no participant appears in both train and test.
3. Report performance with **confidence intervals**, calibrated probabilities, and error analysis
   focused on false negatives.
4. Compare against **baselines** (majority-class, simple thresholds) to show the models add value.
5. Independent/external validation and a clear statement of clinical scope (which, today, is
   **none**, this is not a medical device).
