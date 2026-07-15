"""Tests for the metric helpers, using small hand-verified examples.

Run from the repository root::

    python -m pytest tests/ -q
"""

from __future__ import annotations

import math

from src.evaluate import (
    compute_classification_metrics,
    compute_regression_metrics,
)


def test_binary_classification_metrics_hand_verified() -> None:
    # y_true / y_pred chosen so metrics are easy to verify by hand.
    #   index: 0    1    2    3
    #   true:  1    1    0    0
    #   pred:  1    0    0    0
    # TP=1 (idx0), FN=1 (idx1), TN=2 (idx2,3), FP=0
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    m = compute_classification_metrics(y_true, y_pred)

    assert m["averaging"] == "binary"
    assert math.isclose(m["accuracy"], 0.75)          # 3/4 correct
    assert math.isclose(m["precision"], 1.0)          # TP/(TP+FP) = 1/1
    assert math.isclose(m["recall"], 0.5)             # TP/(TP+FN) = 1/2
    assert math.isclose(m["f1"], 2 / 3, rel_tol=1e-9) # 2*1*0.5/(1+0.5)
    # 2x2 confusion matrix, [[TN, FP], [FN, TP]] = [[2, 0], [1, 1]]
    assert m["confusion_matrix"] == [[2, 0], [1, 1]]
    assert m["roc_auc"] is None  # no probabilities supplied


def test_binary_roc_auc_perfect_separation() -> None:
    y_true = [0, 0, 1, 1]
    proba = [0.1, 0.2, 0.8, 0.9]  # positive-class probabilities
    m = compute_classification_metrics(y_true, [0, 0, 1, 1], proba)
    assert math.isclose(m["roc_auc"], 1.0)


def test_multiclass_uses_macro_averaging() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 1]
    m = compute_classification_metrics(y_true, y_pred)
    assert m["averaging"] == "macro"
    assert math.isclose(m["accuracy"], 4 / 6, rel_tol=1e-9)
    # 3x3 confusion matrix.
    assert len(m["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in m["confusion_matrix"])


def test_regression_metrics_hand_verified() -> None:
    # Classic small example.
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    m = compute_regression_metrics(y_true, y_pred)

    # MAE = mean(|0.5|, |0.5|, 0, |1|) = 0.5
    assert math.isclose(m["mae"], 0.5)
    # RMSE = sqrt(mean(0.25, 0.25, 0, 1)) = sqrt(0.375)
    assert math.isclose(m["rmse"], math.sqrt(0.375), rel_tol=1e-9)
    # R^2 for this example is ~0.9486 (variance-weighted).
    assert math.isclose(m["r2"], 0.9486081370449679, rel_tol=1e-6)


def test_confusion_matrix_shape_binary() -> None:
    m = compute_classification_metrics([0, 1, 0, 1], [0, 1, 1, 0])
    cm = m["confusion_matrix"]
    assert len(cm) == 2 and all(len(row) == 2 for row in cm)
