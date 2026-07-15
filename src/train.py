"""Model factories, registry, and the training CLI.

Every model here is trained on **synthetic** data (see :mod:`src.data_processing`).
The factory functions return *unfitted* scikit-learn pipelines whose
hyperparameters match the original standalone training scripts.

Run from the repository root::

    python -m src.train --config configs/default.yaml --model svm
    python -m src.train --config configs/default.yaml --model all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import joblib
import yaml
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from . import RANDOM_SEED
from . import data_processing as dp
from . import evaluate as ev
from .feature_engineering import FEATURE_SETS, add_health_risk_derived_features

logger = logging.getLogger(__name__)

__all__ = [
    "build_rhythm_model",
    "build_health_risk_model",
    "build_hrv_model",
    "build_cardio_models",
    "MODEL_REGISTRY",
    "MODEL_KEYS",
    "load_config",
    "resolve_dir",
    "build_training_units",
    "train_model",
    "main",
]

#: Repository root (the parent of the ``src`` package directory).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Cardiovascular regression targets and the model key each is saved under.
CARDIO_TARGETS: Tuple[str, ...] = ("fitness_level", "vo2max", "cardiovascular_age")


# --------------------------------------------------------------------------- #
# Model factory functions (unfitted pipelines matching the original scripts).
# --------------------------------------------------------------------------- #
def build_rhythm_model(random_state: int = RANDOM_SEED) -> Pipeline:
    """Build the SVM soft-voting ensemble for binary rhythm classification.

    RobustScaler + soft-voting ensemble of SVC(RBF), LogisticRegression and
    RandomForest, matching ``train_svm_model.py``.

    Parameters
    ----------
    random_state : int, default=42
        Seed applied to every stochastic estimator.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline.
    """
    ensemble = VotingClassifier(
        estimators=[
            ("svm", SVC(kernel="rbf", C=10, gamma=0.1, probability=True, random_state=random_state)),
            ("lr", LogisticRegression(max_iter=1000, random_state=random_state)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ],
        voting="soft",
    )
    return Pipeline([("scaler", RobustScaler()), ("classifier", ensemble)])


def build_health_risk_model(random_state: int = RANDOM_SEED) -> Pipeline:
    """Build the gradient-boosting pipeline for binary health-risk classification.

    StandardScaler + GradientBoostingClassifier, matching ``train_gbm_model.py``.

    Parameters
    ----------
    random_state : int, default=42
        Seed for the estimator.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline.
    """
    gbm = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("gbm", gbm)])


def build_hrv_model(random_state: int = RANDOM_SEED) -> Pipeline:
    """Build the MLP pipeline for 4-class HRV pattern classification.

    StandardScaler + MLPClassifier(hidden=(64, 32, 16)), matching
    ``train_hrv_nn_model.py``.

    Parameters
    ----------
    random_state : int, default=42
        Seed for the estimator.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])


def build_cardio_models(random_state: int = RANDOM_SEED) -> Dict[str, Pipeline]:
    """Build the three cardiovascular-fitness regression pipelines.

    Matches ``train_cardiovascular_fitness_model.py``:

    * ``fitness_level`` — RobustScaler + VotingRegressor(RF, GBR, XGB).
    * ``vo2max`` — StandardScaler + MLPRegressor(hidden=(100, 50, 25)).
    * ``cardiovascular_age`` — RobustScaler + XGBRegressor.

    ``xgboost`` is imported lazily so that importing :mod:`src.train` never
    requires xgboost/libomp to be present.

    Parameters
    ----------
    random_state : int, default=42
        Seed applied to every estimator.

    Returns
    -------
    dict of str to sklearn.pipeline.Pipeline
        Mapping of target name to an unfitted pipeline.
    """
    import xgboost as xgb
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestRegressor,
        VotingRegressor,
    )
    from sklearn.neural_network import MLPRegressor

    fitness_ensemble = VotingRegressor(
        [
            ("rf", RandomForestRegressor(n_estimators=200, max_depth=15, random_state=random_state)),
            ("gb", GradientBoostingRegressor(n_estimators=150, max_depth=10, random_state=random_state)),
            ("xgb", xgb.XGBRegressor(n_estimators=200, max_depth=12, learning_rate=0.1, random_state=random_state)),
        ]
    )
    fitness_pipe = Pipeline([("scaler", RobustScaler()), ("model", fitness_ensemble)])

    vo2max_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=random_state,
                ),
            ),
        ]
    )

    cv_age_pipe = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "model",
                xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=10,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "fitness_level": fitness_pipe,
        "vo2max": vo2max_pipe,
        "cardiovascular_age": cv_age_pipe,
    }


class RegistryEntry(NamedTuple):
    """One row of :data:`MODEL_REGISTRY`.

    Attributes
    ----------
    generator : callable
        ``(n_samples, random_state) -> DataFrame`` synthetic data generator.
    feature_key : str
        Key into :data:`src.feature_engineering.FEATURE_SETS`.
    target : str or tuple of str
        Target column name(s). A tuple denotes a multi-target (cardio) entry.
    builder : callable
        Factory returning an unfitted pipeline (or, for cardio, a dict of them).
    task : str
        ``"classification"`` or ``"regression"``.
    """

    generator: Callable[..., Any]
    feature_key: str
    target: Union[str, Tuple[str, ...]]
    builder: Callable[..., Any]
    task: str


#: Registry mapping model key -> data generator, feature set, target, builder, task.
MODEL_REGISTRY: Dict[str, RegistryEntry] = {
    "svm": RegistryEntry(dp.generate_rhythm_data, "svm", "label", build_rhythm_model, "classification"),
    "gbm": RegistryEntry(dp.generate_health_risk_data, "gbm", "risk_level", build_health_risk_model, "classification"),
    "nn": RegistryEntry(dp.generate_hrv_pattern_data, "nn", "condition", build_hrv_model, "classification"),
    "cardio": RegistryEntry(dp.generate_cardio_fitness_data, "cardio", CARDIO_TARGETS, build_cardio_models, "regression"),
}

#: Ordered list of valid model keys for the CLIs.
MODEL_KEYS: List[str] = list(MODEL_REGISTRY.keys())


class TrainingUnit(NamedTuple):
    """A single trainable/evaluable unit (one estimator + its data split)."""

    name: str
    estimator: Any
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    task: str
    notes: str


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the YAML file (relative paths are resolved against the CWD, and,
        failing that, against the repository root).

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the config file cannot be found.
    """
    candidate = Path(path)
    if not candidate.is_file():
        candidate = REPO_ROOT / path
    if not candidate.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with candidate.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_dir(config: Dict[str, Any], key: str) -> Path:
    """Resolve an output directory from the config, relative to the repo root.

    Parameters
    ----------
    config : dict
        Parsed configuration.
    key : str
        One of ``"models_dir"``, ``"results_dir"``, ``"assets_dir"``.

    Returns
    -------
    pathlib.Path
        Absolute path (created if missing). Absolute config values are honoured
        as-is; relative values are joined to the repository root.
    """
    default = {"models_dir": "models", "results_dir": "results", "assets_dir": "assets"}[key]
    value = Path(config.get(key, default))
    path = value if value.is_absolute() else REPO_ROOT / value
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_config(config: Dict[str, Any], model_key: str) -> Tuple[int, float]:
    """Return ``(n_samples, test_size)`` for a model key from the config."""
    models_cfg = config.get("models", {})
    entry = models_cfg.get(model_key, {})
    default_samples = {"svm": 5000, "gbm": 10000, "nn": 4000, "cardio": 2000}[model_key]
    n_samples = int(entry.get("n_samples", default_samples))
    test_size = float(entry.get("test_size", config.get("test_size", 0.2)))
    return n_samples, test_size


def _notes_for(task: str, target: str, n_classes: int) -> str:
    """Build the free-text notes string for a metrics row."""
    if task == "regression":
        return f"synthetic data; regression target={target}"
    if n_classes > 2:
        return "synthetic data; multiclass macro-averaged"
    return "synthetic data; binary"


def build_training_units(model_key: str, config: Dict[str, Any]) -> List[TrainingUnit]:
    """Build the training units for a model key (data + unfitted estimator).

    Classification keys yield a single unit; ``"cardio"`` yields one unit per
    regression target. Data generation and splitting use the config's seed and
    per-model sample sizes, so calling this from :mod:`src.train` and
    :mod:`src.evaluate` produces identical splits.

    Parameters
    ----------
    model_key : str
        One of :data:`MODEL_KEYS`.
    config : dict
        Parsed configuration.

    Returns
    -------
    list of TrainingUnit
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key '{model_key}'. Valid keys: {MODEL_KEYS}")

    entry = MODEL_REGISTRY[model_key]
    seed = int(config.get("random_seed", RANDOM_SEED))
    n_samples, test_size = _model_config(config, model_key)
    feature_cols = FEATURE_SETS[entry.feature_key]

    df = entry.generator(n_samples=n_samples, random_state=seed)
    # Add derived features on demand (kept out of the raw generators).
    if any(col not in df.columns for col in feature_cols) and model_key == "gbm":
        df = add_health_risk_derived_features(df)

    units: List[TrainingUnit] = []

    if model_key == "cardio":
        pipelines = entry.builder(random_state=seed)
        for target in entry.target:
            X_train, X_test, y_train, y_test = dp.train_test_split_Xy(
                df[feature_cols + [target]], target, test_size=test_size,
                random_state=seed, stratify=False,
            )
            units.append(
                TrainingUnit(
                    name=f"cardio_{target}",
                    estimator=pipelines[target],
                    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                    task="regression",
                    notes=_notes_for("regression", target, 0),
                )
            )
        return units

    target = entry.target  # type: ignore[assignment]
    X_train, X_test, y_train, y_test = dp.train_test_split_Xy(
        df[feature_cols + [target]], target, test_size=test_size,
        random_state=seed, stratify=True,
    )
    n_classes = int(df[target].nunique())
    units.append(
        TrainingUnit(
            name=model_key,
            estimator=entry.builder(random_state=seed),
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            task=entry.task,
            notes=_notes_for(entry.task, str(target), n_classes),
        )
    )
    return units


def train_model(
    model_key: str, config: Dict[str, Any]
) -> Tuple[Any, Any]:
    """Train a model (or the cardio model group) and return fitted estimator(s).

    Parameters
    ----------
    model_key : str
        One of :data:`MODEL_KEYS`.
    config : dict
        Parsed configuration.

    Returns
    -------
    tuple
        For single-model keys: ``(fitted_pipeline, (X_train, X_test, y_train,
        y_test))``. For ``"cardio"``: ``(dict[name -> fitted_pipeline],
        dict[name -> (X_train, X_test, y_train, y_test)])``.
    """
    units = build_training_units(model_key, config)
    for unit in units:
        unit.estimator.fit(unit.X_train, unit.y_train)

    if model_key == "cardio":
        fitted = {u.name: u.estimator for u in units}
        splits = {u.name: (u.X_train, u.X_test, u.y_train, u.y_test) for u in units}
        return fitted, splits

    unit = units[0]
    return unit.estimator, (unit.X_train, unit.X_test, unit.y_train, unit.y_test)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: train the selected model(s), save them, write metrics.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument vector (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Train TelemetryHealthCare models on synthetic data.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--model", default="all", choices=[*MODEL_KEYS, "all"], help="Which model(s) to train."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    config = load_config(args.config)
    models_dir = resolve_dir(config, "models_dir")
    results_dir = resolve_dir(config, "results_dir")

    model_keys = MODEL_KEYS if args.model == "all" else [args.model]
    rows: List[Dict[str, Any]] = []

    for model_key in model_keys:
        logger.info("Training '%s'...", model_key)
        units = build_training_units(model_key, config)
        for unit in units:
            logger.info("  Fitting %s (n_train=%d)...", unit.name, len(unit.X_train))
            unit.estimator.fit(unit.X_train, unit.y_train)

            model_path = models_dir / f"{unit.name}.pkl"
            joblib.dump(unit.estimator, model_path)
            logger.info("  Saved model to %s", model_path)

            y_pred = unit.estimator.predict(unit.X_test)
            if unit.task == "classification":
                proba = (
                    unit.estimator.predict_proba(unit.X_test)
                    if hasattr(unit.estimator, "predict_proba")
                    else None
                )
                clf = ev.compute_classification_metrics(unit.y_test, y_pred, proba)
                rows.append(
                    ev.build_metrics_row(
                        unit.name, unit.task, len(unit.X_train), len(unit.X_test),
                        classification=clf, notes=unit.notes,
                    )
                )
            else:
                reg = ev.compute_regression_metrics(unit.y_test, y_pred)
                rows.append(
                    ev.build_metrics_row(
                        unit.name, unit.task, len(unit.X_train), len(unit.X_test),
                        regression=reg, notes=unit.notes,
                    )
                )

    path = ev.update_metrics_csv(rows, results_dir)
    logger.info("Training complete. Metrics written to %s", path)


if __name__ == "__main__":
    main()
