"""Feature engineering for the TelemetryHealthCare models.

Contains the derived-feature transform used by the health-risk (GBM) model and
the canonical mapping of model name to feature columns. As with the rest of the
package, all inputs are expected to be **synthetic** wearable-style data.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

__all__ = ["add_health_risk_derived_features", "FEATURE_SETS"]


def add_health_risk_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the two derived health-risk features used by the GBM model.

    Implements the same formulas as ``train_gbm_model.py``:

    * ``hr_hrv_ratio = average_heart_rate / (hrv_mean + 1)`` — a cardiovascular
      stress / autonomic-balance proxy (the ``+ 1`` guards against division by
      zero at very low HRV).
    * ``recovery_score = sleep_quality * hrv_mean / 50`` — sleep quality weighted
      by HRV.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame containing at least ``average_heart_rate``, ``hrv_mean`` and
        ``sleep_quality``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with ``hr_hrv_ratio`` and ``recovery_score`` added.

    Raises
    ------
    KeyError
        If any required base column is missing.
    """
    required = {"average_heart_rate", "hrv_mean", "sleep_quality"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for derived features: {sorted(missing)}")

    out = df.copy()
    out["hr_hrv_ratio"] = out["average_heart_rate"] / (out["hrv_mean"] + 1)
    out["recovery_score"] = out["sleep_quality"] * out["hrv_mean"] / 50
    return out


#: Canonical feature columns for each model key.
FEATURE_SETS: Dict[str, List[str]] = {
    # SVM soft-voting ensemble — binary heart rhythm.
    "svm": ["mean_heart_rate", "std_heart_rate", "pnn50"],
    # Gradient boosting — binary health risk (includes 2 derived features).
    "gbm": [
        "average_heart_rate",
        "hrv_mean",
        "respiratory_rate",
        "activity_level",
        "sleep_quality",
        "stress_indicator",
        "hr_hrv_ratio",
        "recovery_score",
    ],
    # MLP — 4-class HRV pattern.
    "nn": [
        "mean_rr",
        "std_rr",
        "min_rr",
        "max_rr",
        "q25_rr",
        "q75_rr",
        "mean_diff_rr",
        "std_diff_rr",
        "rmssd",
        "pnn50",
        "low_freq_power",
        "mid_freq_power",
        "high_freq_power",
    ],
    # Cardiovascular fitness regressors — shared 4-feature input.
    "cardio": ["age", "resting_hr", "hrr_1min", "rmssd"],
}
