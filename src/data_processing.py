"""Synthetic dataset generators for the TelemetryHealthCare models.

.. warning::
    All data produced by this module is **synthetic**. It is sampled from NumPy
    probability distributions chosen to loosely resemble wearable-style
    physiological metrics. None of it is measured from real people or devices,
    and it must not be used to draw clinical conclusions.

Each generator faithfully reproduces the sampling distributions used by the
original standalone training scripts (``scripts/training/*.py``) while sharing a
single, reproducible interface: every function accepts ``n_samples`` and
``random_state`` and returns a :class:`pandas.DataFrame`. Reproducibility is
achieved with a per-call :func:`numpy.random.default_rng` instance, so the
generators are independent of global NumPy state and of one another.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import RANDOM_SEED

__all__ = [
    "generate_rhythm_data",
    "generate_health_risk_data",
    "generate_hrv_pattern_data",
    "generate_cardio_fitness_data",
    "train_test_split_Xy",
    "HRV_CLASS_NAMES",
]

#: Ordered class names for the 4-class HRV pattern task (index == integer label).
HRV_CLASS_NAMES: Tuple[str, ...] = ("normal", "afib", "bradycardia", "tachycardia")

#: Minimum number of samples any generator will accept.
_MIN_SAMPLES: int = 10


def _validate_n_samples(n_samples: int) -> None:
    """Validate a requested sample count.

    Parameters
    ----------
    n_samples : int
        Requested number of samples.

    Raises
    ------
    ValueError
        If ``n_samples`` is smaller than :data:`_MIN_SAMPLES`.
    """
    if n_samples < _MIN_SAMPLES:
        raise ValueError(
            f"n_samples must be >= {_MIN_SAMPLES}, got {n_samples}. "
            "The generators need enough rows to form balanced, splittable classes."
        )


def generate_rhythm_data(
    n_samples: int = 5000, random_state: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate synthetic heart-rhythm classification data (binary).

    Reproduces the distributions from ``train_svm_model.py``. Roughly 60% of the
    rows are "Normal" rhythm and 40% "Irregular", using physiologically-flavoured
    (but entirely synthetic) distributions for the three features.

    Parameters
    ----------
    n_samples : int, default=5000
        Total number of samples to generate.
    random_state : int, default=42
        Seed for the local :func:`numpy.random.default_rng` instance.

    Returns
    -------
    pandas.DataFrame
        Columns ``mean_heart_rate``, ``std_heart_rate``, ``pnn50`` and the binary
        target ``label`` (0 = Normal, 1 = Irregular).

    Raises
    ------
    ValueError
        If ``n_samples`` < 10.
    """
    _validate_n_samples(n_samples)
    rng = np.random.default_rng(random_state)

    n_normal = int(n_samples * 0.6)
    n_irregular = n_samples - n_normal
    n_tachy = n_irregular // 2
    n_brady = n_irregular - n_tachy

    normal = pd.DataFrame(
        {
            "mean_heart_rate": rng.normal(70, 10, n_normal),
            "std_heart_rate": rng.gamma(2, 1.5, n_normal),
            "pnn50": rng.beta(2, 5, n_normal) * 0.5,
        }
    )
    normal["label"] = 0

    irregular = pd.DataFrame(
        {
            "mean_heart_rate": np.concatenate(
                [rng.normal(95, 15, n_tachy), rng.normal(50, 8, n_brady)]
            ),
            "std_heart_rate": rng.gamma(4, 2, n_irregular),
            "pnn50": rng.beta(1, 8, n_irregular) * 0.3,
        }
    )
    irregular["label"] = 1

    data = pd.concat([normal, irregular], ignore_index=True)
    data["mean_heart_rate"] = np.clip(data["mean_heart_rate"], 40, 120)
    data["std_heart_rate"] = np.clip(data["std_heart_rate"], 1, 20)
    data["pnn50"] = np.clip(data["pnn50"], 0, 0.5)

    return data.sample(frac=1, random_state=random_state).reset_index(drop=True)


def generate_health_risk_data(
    n_samples: int = 10000, random_state: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate synthetic health-risk classification data (binary).

    Reproduces the base-feature distributions from ``train_gbm_model.py`` (roughly
    60% "Low risk" / 40% "High risk"). The two derived features used by the GBM
    model (``hr_hrv_ratio`` and ``recovery_score``) are intentionally **not**
    added here; use
    :func:`src.feature_engineering.add_health_risk_derived_features` for that so
    that feature engineering lives in one place.

    Parameters
    ----------
    n_samples : int, default=10000
        Total number of samples to generate.
    random_state : int, default=42
        Seed for the local :func:`numpy.random.default_rng` instance.

    Returns
    -------
    pandas.DataFrame
        Columns ``average_heart_rate``, ``hrv_mean``, ``respiratory_rate``,
        ``activity_level``, ``sleep_quality``, ``stress_indicator`` and the binary
        target ``risk_level`` (0 = Low, 1 = High).

    Raises
    ------
    ValueError
        If ``n_samples`` < 10.
    """
    _validate_n_samples(n_samples)
    rng = np.random.default_rng(random_state)

    n_low = int(n_samples * 0.6)
    n_high = n_samples - n_low
    n_tachy = n_high // 2
    n_brady = n_high - n_tachy

    low = pd.DataFrame(
        {
            "average_heart_rate": rng.normal(70, 8, n_low),
            "hrv_mean": rng.normal(50, 15, n_low),
            "respiratory_rate": rng.normal(14, 2, n_low),
            "activity_level": rng.gamma(3, 100, n_low),
            "sleep_quality": rng.beta(7, 3, n_low),
            "stress_indicator": rng.beta(2, 5, n_low),
        }
    )
    low["risk_level"] = 0

    high = pd.DataFrame(
        {
            "average_heart_rate": np.concatenate(
                [rng.normal(90, 12, n_tachy), rng.normal(55, 8, n_brady)]
            ),
            "hrv_mean": rng.normal(30, 10, n_high),
            "respiratory_rate": rng.normal(18, 3, n_high),
            "activity_level": rng.gamma(1, 50, n_high),
            "sleep_quality": rng.beta(3, 7, n_high),
            "stress_indicator": rng.beta(5, 2, n_high),
        }
    )
    high["risk_level"] = 1

    data = pd.concat([low, high], ignore_index=True)
    data["average_heart_rate"] = np.clip(data["average_heart_rate"], 40, 120)
    data["hrv_mean"] = np.clip(data["hrv_mean"], 10, 100)
    data["respiratory_rate"] = np.clip(data["respiratory_rate"], 8, 25)
    data["activity_level"] = np.clip(data["activity_level"], 0, 1000)
    data["sleep_quality"] = np.clip(data["sleep_quality"], 0, 1)
    data["stress_indicator"] = np.clip(data["stress_indicator"], 0, 1)

    return data.sample(frac=1, random_state=random_state).reset_index(drop=True)


def _extract_hrv_features(rr_intervals: np.ndarray, sequence_length: int) -> list[float]:
    """Extract the 13 HRV features from one RR-interval sequence.

    Mirrors the time-domain and frequency-domain feature extraction in
    ``train_hrv_nn_model.py``.

    Parameters
    ----------
    rr_intervals : numpy.ndarray
        RR intervals in milliseconds.
    sequence_length : int
        Length of the source heart-rate sequence (used to slice the FFT).

    Returns
    -------
    list of float
        The 13 engineered features, with any non-finite value replaced by 0.0.
    """
    rr_diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diff**2))
    pnn50 = len(np.where(np.abs(rr_diff) > 50)[0]) / len(rr_intervals)
    fft_values = np.abs(np.fft.fft(rr_intervals))[: sequence_length // 2]

    features = [
        np.mean(rr_intervals),
        np.std(rr_intervals),
        np.min(rr_intervals),
        np.max(rr_intervals),
        np.percentile(rr_intervals, 25),
        np.percentile(rr_intervals, 75),
        np.mean(rr_diff),
        np.std(rr_diff),
        rmssd,
        pnn50,
        np.mean(fft_values[:5]),
        np.mean(fft_values[5:15]),
        np.mean(fft_values[15:]),
    ]
    return [float(f) if np.isfinite(f) else 0.0 for f in features]


#: Ordered feature names produced by :func:`_extract_hrv_features`.
HRV_FEATURE_NAMES: Tuple[str, ...] = (
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
)


def generate_hrv_pattern_data(
    n_samples: int = 4000,
    random_state: int = RANDOM_SEED,
    sequence_length: int = 50,
) -> pd.DataFrame:
    """Generate synthetic HRV-pattern classification data (4 classes, balanced).

    Reproduces the synthetic RR-interval simulation and feature extraction from
    ``train_hrv_nn_model.py``. Four cardiac-rhythm patterns are simulated in equal
    proportion (``normal``, ``afib``, ``bradycardia``, ``tachycardia``); 13 HRV
    features are extracted from each simulated sequence.

    Parameters
    ----------
    n_samples : int, default=4000
        Total number of samples; split evenly across the four classes
        (``n_samples // 4`` each, so a few trailing rows are dropped if
        ``n_samples`` is not a multiple of four).
    random_state : int, default=42
        Seed for the local :func:`numpy.random.default_rng` instance.
    sequence_length : int, default=50
        Number of heart-rate samples simulated per sequence.

    Returns
    -------
    pandas.DataFrame
        The 13 columns in :data:`HRV_FEATURE_NAMES` plus the integer target
        ``condition`` (0=normal, 1=afib, 2=bradycardia, 3=tachycardia; see
        :data:`HRV_CLASS_NAMES`).

    Raises
    ------
    ValueError
        If ``n_samples`` < 10.
    """
    _validate_n_samples(n_samples)
    rng = np.random.default_rng(random_state)

    samples_per_condition = n_samples // 4
    rows: list[list[float]] = []
    labels: list[int] = []

    for condition_idx, condition in enumerate(HRV_CLASS_NAMES):
        for _ in range(samples_per_condition):
            if condition == "normal":
                base_hr = rng.normal(70, 10)
                variability = rng.normal(0, 5, sequence_length)
            elif condition == "afib":
                base_hr = rng.normal(80, 15)
                variability = rng.normal(0, 15, sequence_length)
                n_spikes = int(rng.integers(5, 9))
                spike_indices = rng.choice(sequence_length, n_spikes, replace=False)
                variability[spike_indices] += rng.normal(20, 5, n_spikes)
            elif condition == "bradycardia":
                base_hr = rng.normal(50, 5)
                variability = rng.normal(0, 3, sequence_length)
            else:  # tachycardia
                base_hr = rng.normal(100, 10)
                variability = rng.normal(0, 2, sequence_length)

            hr_sequence = np.clip(base_hr + variability, 40, 150)

            if condition == "afib":
                ectopic_mask = rng.random(sequence_length) < 0.1
                hr_sequence[ectopic_mask] += rng.normal(0, 10, int(np.sum(ectopic_mask)))
                hr_sequence = np.clip(hr_sequence, 40, 150)

            rr_intervals = np.clip(60000 / hr_sequence, 400, 1500)
            rows.append(_extract_hrv_features(rr_intervals, sequence_length))
            labels.append(condition_idx)

    data = pd.DataFrame(rows, columns=list(HRV_FEATURE_NAMES))
    data["condition"] = labels
    return data.sample(frac=1, random_state=random_state).reset_index(drop=True)


def generate_cardio_fitness_data(
    n_samples: int = 2000, random_state: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate synthetic cardiovascular-fitness regression data.

    Reproduces the profile/metric distributions from
    ``train_cardiovascular_fitness_model.py`` (vectorised for speed). The returned
    frame contains the four features used by the consolidated cardio models plus
    the three regression targets and a handful of intermediate metrics.

    Parameters
    ----------
    n_samples : int, default=2000
        Total number of synthetic user profiles to generate.
    random_state : int, default=42
        Seed for the local :func:`numpy.random.default_rng` instance.

    Returns
    -------
    pandas.DataFrame
        Feature columns ``age``, ``resting_hr``, ``hrr_1min``, ``rmssd``;
        intermediate columns ``max_hr``, ``hr_reserve``, ``hrr_2min``, ``sdnn``,
        ``pnn50``, ``lf_hf_ratio``; and target columns ``fitness_level``,
        ``vo2max``, ``cardiovascular_age``.

    Raises
    ------
    ValueError
        If ``n_samples`` < 10.
    """
    _validate_n_samples(n_samples)
    rng = np.random.default_rng(random_state)
    n = n_samples

    age = rng.integers(18, 80, n).astype(float)

    base_fitness = np.clip(70 - (age - 40) * 0.5 + rng.normal(0, 10, n), 10, 95)
    activities = np.array(["sedentary", "light", "moderate", "active", "athlete"])
    activity_level = rng.choice(activities, size=n, p=[0.25, 0.30, 0.25, 0.15, 0.05])
    multiplier_map = {
        "sedentary": 0.7,
        "light": 0.85,
        "moderate": 1.0,
        "active": 1.15,
        "athlete": 1.3,
    }
    multiplier = np.array([multiplier_map[a] for a in activity_level])
    fitness = np.clip(base_fitness * multiplier, 10, 95)

    resting_hr = np.clip(85 - fitness * 0.35 + rng.normal(0, 3, n), 40, 95)
    max_hr = 220 - age + rng.normal(0, 5, n)
    hr_reserve = max_hr - resting_hr

    hrr_high = 30 + (fitness - 70) * 0.5 + rng.normal(0, 3, n)
    hrr_mid = 20 + (fitness - 40) * 0.33 + rng.normal(0, 2, n)
    hrr_low = 12 + (fitness - 20) * 0.4 + rng.normal(0, 2, n)
    hrr_1min = np.clip(
        np.where(fitness > 70, hrr_high, np.where(fitness > 40, hrr_mid, hrr_low)), 5, 50
    )
    hrr_2min = np.clip(hrr_1min * 1.5 + rng.normal(0, 3, n), 10, 70)

    rmssd = np.clip(20 + fitness * 0.6 + rng.normal(0, 5, n), 10, 100)
    sdnn = np.clip(30 + fitness * 0.7 + rng.normal(0, 8, n), 20, 120)
    pnn50 = np.clip(fitness * 0.3 + rng.normal(0, 3, n), 0, 40)

    lf_hf_high = 0.8 + rng.normal(0, 0.2, n)
    lf_hf_low = 1.5 + (60 - fitness) * 0.02 + rng.normal(0, 0.3, n)
    lf_hf_ratio = np.clip(np.where(fitness > 60, lf_hf_high, lf_hf_low), 0.3, 4.0)

    vo2_athlete = 50 + fitness * 0.3 + rng.normal(0, 3, n)
    vo2_active = 40 + fitness * 0.25 + rng.normal(0, 3, n)
    vo2_moderate = 35 + fitness * 0.2 + rng.normal(0, 3, n)
    vo2_other = 25 + fitness * 0.2 + rng.normal(0, 3, n)
    vo2max = np.select(
        [
            activity_level == "athlete",
            activity_level == "active",
            activity_level == "moderate",
        ],
        [vo2_athlete, vo2_active, vo2_moderate],
        default=vo2_other,
    )
    vo2max = np.clip(vo2max, 15, 75)

    cardiovascular_age = np.clip(
        age + (fitness - 50) * -0.3 + rng.normal(0, 3, n), 18, 90
    )

    return pd.DataFrame(
        {
            "age": age,
            "resting_hr": resting_hr,
            "max_hr": max_hr,
            "hr_reserve": hr_reserve,
            "hrr_1min": hrr_1min,
            "hrr_2min": hrr_2min,
            "rmssd": rmssd,
            "sdnn": sdnn,
            "pnn50": pnn50,
            "lf_hf_ratio": lf_hf_ratio,
            "fitness_level": fitness,
            "vo2max": vo2max,
            "cardiovascular_age": cardiovascular_age,
        }
    )


def train_test_split_Xy(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into stratified (or plain) train/test feature-target sets.

    Every column except ``target`` is treated as a feature, so callers should pass
    a frame already reduced to ``feature_cols + [target]``.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame containing the feature columns and the ``target`` column.
    target : str
        Name of the target column.
    test_size : float, default=0.2
        Proportion of rows held out for the test split.
    random_state : int, default=42
        Seed passed to :func:`sklearn.model_selection.train_test_split`.
    stratify : bool, default=True
        If True, stratify the split on the target (use for classification only).

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )
