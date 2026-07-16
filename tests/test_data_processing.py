"""Tests for the synthetic data generators.

Run from the repository root so that ``import src`` resolves::

    python -m pytest tests/ -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import data_processing as dp
from src.feature_engineering import FEATURE_SETS, add_health_risk_derived_features

RHYTHM_COLUMNS = {"mean_heart_rate", "std_heart_rate", "pnn50", "label"}
HEALTH_COLUMNS = {
    "average_heart_rate", "hrv_mean", "respiratory_rate", "activity_level",
    "sleep_quality", "stress_indicator", "risk_level",
}


def test_rhythm_shape_and_columns() -> None:
    df = dp.generate_rhythm_data(n_samples=500, random_state=42)
    assert len(df) == 500
    assert set(df.columns) == RHYTHM_COLUMNS
    assert not df.isnull().values.any()


def test_health_risk_shape_columns_and_derived() -> None:
    df = dp.generate_health_risk_data(n_samples=800, random_state=42)
    assert len(df) == 800
    assert set(df.columns) == HEALTH_COLUMNS
    assert not df.isnull().values.any()
    # Derived features can be added and cover the gbm feature set.
    derived = add_health_risk_derived_features(df)
    assert "hr_hrv_ratio" in derived.columns
    assert "recovery_score" in derived.columns
    assert set(FEATURE_SETS["gbm"]).issubset(derived.columns)


def test_hrv_shape_columns_and_balance() -> None:
    df = dp.generate_hrv_pattern_data(n_samples=400, random_state=42)
    assert len(df) == 400  # 400 is a multiple of 4
    assert set(FEATURE_SETS["nn"]).issubset(df.columns)
    assert "condition" in df.columns
    assert not df.isnull().values.any()
    # Perfectly balanced across the 4 classes.
    counts = df["condition"].value_counts().to_dict()
    assert set(counts.keys()) == {0, 1, 2, 3}
    assert all(count == 100 for count in counts.values())


def test_cardio_shape_and_targets() -> None:
    df = dp.generate_cardio_fitness_data(n_samples=500, random_state=42)
    assert len(df) == 500
    for col in FEATURE_SETS["cardio"]:
        assert col in df.columns
    for target in ("fitness_level", "vo2max", "cardiovascular_age"):
        assert target in df.columns
    assert not df.isnull().values.any()


@pytest.mark.parametrize(
    "generator",
    [
        dp.generate_rhythm_data,
        dp.generate_health_risk_data,
        dp.generate_hrv_pattern_data,
        dp.generate_cardio_fitness_data,
    ],
)
def test_reproducibility(generator) -> None:
    first = generator(n_samples=300, random_state=42)
    second = generator(n_samples=300, random_state=42)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    "generator",
    [
        dp.generate_rhythm_data,
        dp.generate_health_risk_data,
        dp.generate_hrv_pattern_data,
        dp.generate_cardio_fitness_data,
    ],
)
def test_value_error_on_too_few_samples(generator) -> None:
    with pytest.raises(ValueError):
        generator(n_samples=9, random_state=42)


def test_rhythm_class_balance_roughly_60_40() -> None:
    df = dp.generate_rhythm_data(n_samples=5000, random_state=42)
    normal_frac = (df["label"] == 0).mean()
    assert 0.55 <= normal_frac <= 0.65  # designed at 0.60


def test_health_risk_class_balance_roughly_60_40() -> None:
    df = dp.generate_health_risk_data(n_samples=5000, random_state=42)
    low_frac = (df["risk_level"] == 0).mean()
    assert 0.55 <= low_frac <= 0.65  # designed at 0.60


def test_train_test_split_Xy_stratified() -> None:
    df = dp.generate_rhythm_data(n_samples=1000, random_state=42)
    X_train, X_test, y_train, y_test = dp.train_test_split_Xy(
        df, "label", test_size=0.2, random_state=42, stratify=True
    )
    assert len(X_train) == 800
    assert len(X_test) == 200
    assert "label" not in X_train.columns
    # Stratification preserves the class ratio within a small tolerance.
    assert abs(y_train.mean() - y_test.mean()) < 0.05
