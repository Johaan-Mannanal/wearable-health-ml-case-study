"""Evaluation metrics, cross-validation, and the evaluation CLI.

All metrics are computed on **synthetic** held-out data; treat them as
illustrative of the modelling pipeline, not as evidence of real-world skill.

Run from the repository root::

    python -m src.evaluate --config configs/default.yaml --model svm
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "cross_validate_model",
    "METRICS_COLUMNS",
    "update_metrics_csv",
    "main",
]

#: Column order for ``results/metrics.csv``.
METRICS_COLUMNS: List[str] = [
    "model",
    "task",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "r2",
    "mae",
    "rmse",
    "cv_mean",
    "cv_std",
    "n_train",
    "n_test",
    "notes",
]


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute standard classification metrics.

    Binary tasks use ``average="binary"`` (positive class = 1); multiclass tasks
    use macro averaging, flagged via the ``averaging`` key. ``roc_auc`` is only
    reported for binary tasks when ``y_proba`` is supplied.

    Parameters
    ----------
    y_true : sequence of int
        Ground-truth labels.
    y_pred : sequence of int
        Predicted labels.
    y_proba : numpy.ndarray, optional
        Predicted probabilities. For binary ROC-AUC, either a 1-D array of
        positive-class probabilities or a 2-D ``(n, 2)`` array is accepted.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``, ``averaging``,
        ``roc_auc`` (may be ``None``), and ``confusion_matrix`` (nested list).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    averaging = "binary" if n_classes <= 2 else "macro"

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=averaging, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=averaging, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average=averaging, zero_division=0)),
        "averaging": averaging,
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None and n_classes == 2:
        proba = np.asarray(y_proba)
        if proba.ndim == 2:
            proba = proba[:, 1]
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        except ValueError:  # e.g. only one class present in y_true
            metrics["roc_auc"] = None

    return metrics


def compute_regression_metrics(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> Dict[str, float]:
    """Compute standard regression metrics.

    Parameters
    ----------
    y_true : sequence of float
        Ground-truth target values.
    y_pred : sequence of float
        Predicted target values.

    Returns
    -------
    dict
        Keys ``r2``, ``mae`` and ``rmse``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def cross_validate_model(
    pipeline: Any,
    X: Any,
    y: Any,
    cv: int = 5,
    scoring: str = "accuracy",
) -> Dict[str, float]:
    """Cross-validate a pipeline and return the mean and std of the score.

    Parameters
    ----------
    pipeline : estimator
        An unfitted scikit-learn estimator/pipeline.
    X : array-like
        Feature matrix.
    y : array-like
        Targets.
    cv : int, default=5
        Number of folds.
    scoring : str, default="accuracy"
        A scikit-learn scoring name (e.g. ``"accuracy"`` or ``"r2"``).

    Returns
    -------
    dict
        Keys ``cv_mean``, ``cv_std`` and ``scoring``.
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
    return {
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "scoring": scoring,
    }


def _blank_row(model: str, task: str) -> Dict[str, Any]:
    """Return a metrics row with every metric field blank."""
    row = {col: "" for col in METRICS_COLUMNS}
    row["model"] = model
    row["task"] = task
    return row


def build_metrics_row(
    model: str,
    task: str,
    n_train: int,
    n_test: int,
    classification: Optional[Dict[str, Any]] = None,
    regression: Optional[Dict[str, float]] = None,
    cv: Optional[Dict[str, float]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Assemble a single ``results/metrics.csv`` row from metric dicts.

    Parameters
    ----------
    model : str
        Model key (e.g. ``"svm"`` or ``"cardio_vo2max"``).
    task : str
        ``"classification"`` or ``"regression"``.
    n_train, n_test : int
        Train/test row counts.
    classification : dict, optional
        Output of :func:`compute_classification_metrics`.
    regression : dict, optional
        Output of :func:`compute_regression_metrics`.
    cv : dict, optional
        Output of :func:`cross_validate_model`.
    notes : str, default=""
        Free-text notes column.

    Returns
    -------
    dict
        A row keyed by :data:`METRICS_COLUMNS`.
    """
    row = _blank_row(model, task)
    row["n_train"] = n_train
    row["n_test"] = n_test
    row["notes"] = notes

    if classification is not None:
        for key in ("accuracy", "precision", "recall", "f1"):
            row[key] = round(classification[key], 4)
        if classification.get("roc_auc") is not None:
            row["roc_auc"] = round(classification["roc_auc"], 4)
    if regression is not None:
        for key in ("r2", "mae", "rmse"):
            row[key] = round(regression[key], 4)
    if cv is not None:
        row["cv_mean"] = round(cv["cv_mean"], 4)
        row["cv_std"] = round(cv["cv_std"], 4)
    return row


def update_metrics_csv(rows: List[Dict[str, Any]], results_dir: Path) -> Path:
    """Upsert metric rows into ``<results_dir>/metrics.csv`` keyed by ``model``.

    Existing rows for the same ``model`` are replaced; other rows are preserved.

    Parameters
    ----------
    rows : list of dict
        Rows produced by :func:`build_metrics_row`.
    results_dir : pathlib.Path
        Directory in which ``metrics.csv`` lives (created if missing).

    Returns
    -------
    pathlib.Path
        Path to the written CSV.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "metrics.csv"

    if path.exists():
        existing = pd.read_csv(path)
    else:
        existing = pd.DataFrame(columns=METRICS_COLUMNS)

    new_models = {row["model"] for row in rows}
    existing = existing[~existing["model"].isin(new_models)] if not existing.empty else existing
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.reindex(columns=METRICS_COLUMNS)
    combined.to_csv(path, index=False)
    return path


def _print_metrics_table(rows: List[Dict[str, Any]]) -> None:
    """Log a compact, aligned metrics table."""
    display_cols = [
        "model",
        "task",
        "accuracy",
        "f1",
        "roc_auc",
        "r2",
        "rmse",
        "cv_mean",
        "cv_std",
    ]
    df = pd.DataFrame(rows).reindex(columns=display_cols)
    logger.info("Evaluation summary:\n%s", df.to_string(index=False))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: load (or train) models, evaluate, and write metrics.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument vector (defaults to ``sys.argv``).
    """
    from . import train as train_module  # local import avoids an import cycle

    parser = argparse.ArgumentParser(description="Evaluate TelemetryHealthCare models.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--model",
        default="all",
        choices=[*train_module.MODEL_KEYS, "all"],
        help="Which model(s) to evaluate.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    config = train_module.load_config(args.config)
    results_dir = train_module.resolve_dir(config, "results_dir")
    models_dir = train_module.resolve_dir(config, "models_dir")
    cv_folds = int(config.get("cv_folds", 5))

    model_keys = train_module.MODEL_KEYS if args.model == "all" else [args.model]

    rows: List[Dict[str, Any]] = []
    for model_key in model_keys:
        logger.info("Evaluating '%s'...", model_key)
        units = train_module.build_training_units(model_key, config)
        for unit in units:
            model_path = models_dir / f"{unit.name}.pkl"
            if model_path.exists():
                import joblib

                estimator = joblib.load(model_path)
                logger.info("  Loaded saved model: %s", model_path)
            else:
                logger.warning(
                    "  No saved model at %s; training a fresh one for evaluation.",
                    model_path,
                )
                estimator = unit.estimator
                estimator.fit(unit.X_train, unit.y_train)

            scoring = "accuracy" if unit.task == "classification" else "r2"
            cv = cross_validate_model(
                unit.estimator, unit.X_train, unit.y_train, cv=cv_folds, scoring=scoring
            )

            y_pred = estimator.predict(unit.X_test)
            if unit.task == "classification":
                proba = (
                    estimator.predict_proba(unit.X_test)
                    if hasattr(estimator, "predict_proba")
                    else None
                )
                clf = compute_classification_metrics(unit.y_test, y_pred, proba)
                row = build_metrics_row(
                    unit.name,
                    unit.task,
                    len(unit.X_train),
                    len(unit.X_test),
                    classification=clf,
                    cv=cv,
                    notes=unit.notes,
                )
            else:
                reg = compute_regression_metrics(unit.y_test, y_pred)
                row = build_metrics_row(
                    unit.name,
                    unit.task,
                    len(unit.X_train),
                    len(unit.X_test),
                    regression=reg,
                    cv=cv,
                    notes=unit.notes,
                )
            rows.append(row)

    _print_metrics_table(rows)
    path = update_metrics_csv(rows, results_dir)
    logger.info("Wrote metrics to %s", path)


if __name__ == "__main__":
    main()
