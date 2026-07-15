"""TelemetryHealthCare — exploratory ML on synthetic wearable-style data.

This package consolidates a set of exploratory machine-learning models that were
originally spread across standalone training scripts. **Every model in this
package is trained exclusively on SYNTHETIC data generated with NumPy.** No real
patient data, Apple Watch recordings, or other measured physiological signals are
used anywhere. The synthetic generators mimic plausible distributions for
wearable-style metrics (heart rate, HRV, activity, sleep, etc.) so that the
modelling pipeline can be exercised end-to-end, but the data — and therefore any
resulting metric — should be treated as illustrative only, never as clinical
evidence.

Sub-modules
-----------
data_processing
    Reproducible synthetic dataset generators and a train/test split helper.
feature_engineering
    Derived-feature helpers and the canonical per-model feature sets.
train
    Model factory functions, a model registry, and a training CLI.
evaluate
    Metric helpers, cross-validation, and an evaluation CLI.
visualization
    Honest matplotlib plotting helpers (never invoked at import time).

Nothing in this package performs training, file I/O, or plotting at import time.
"""

__version__ = "0.1.0"

# Global default seed used to make every synthetic dataset and model reproducible.
RANDOM_SEED: int = 42

__all__ = ["__version__", "RANDOM_SEED"]
