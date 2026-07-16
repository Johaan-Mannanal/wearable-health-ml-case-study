#!/usr/bin/env python3
"""Generate the figures in assets/ from the verified synthetic-data results.

Run from the repo root:  python scripts/generate_assets.py
All charts use zero-based axes (no exaggeration) and are regenerable.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src import RANDOM_SEED
from src.data_processing import train_test_split_Xy
from src.feature_engineering import add_health_risk_derived_features, FEATURE_SETS
from src.train import (
    build_rhythm_model,
    build_health_risk_model,
    build_hrv_model,
)
from src.data_processing import (
    generate_rhythm_data,
    generate_health_risk_data,
    generate_hrv_pattern_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("assets")

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "assets"
ASSETS.mkdir(exist_ok=True)


def _fit_predict(df, target, feature_cols, builder):
    X_tr, X_te, y_tr, y_te = train_test_split_Xy(
        df, target=target, test_size=0.2, random_state=RANDOM_SEED, stratify=True
    )
    X_tr, X_te = X_tr[feature_cols], X_te[feature_cols]
    model = builder()
    model.fit(X_tr, y_tr)
    return y_te, model.predict(X_te)


def confusion_png(y_true, y_pred, labels, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ASSETS / fname, dpi=140)
    plt.close(fig)
    log.info("wrote %s", fname)


def model_comparison_png():
    df = pd.read_csv(REPO / "results" / "metrics.csv")
    clf = df[df["task"] == "classification"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    metrics = ["accuracy", "f1", "cv_mean"]
    x = np.arange(len(clf))
    width = 0.25
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * width, clf[m], width, label=m)
    ax.set_xticks(x, clf["model"].str.upper())
    ax.set_ylim(0, 1.0)  # honest zero-based axis
    ax.set_ylabel("score")
    ax.set_title("Classifier comparison (synthetic data)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "model_comparison.png", dpi=140)
    plt.close(fig)
    log.info("wrote model_comparison.png")


def class_distribution_png():
    rhythm = generate_rhythm_data(random_state=RANDOM_SEED)
    hrv = generate_hrv_pattern_data(random_state=RANDOM_SEED)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    rc = rhythm["label"].value_counts().sort_index()
    axes[0].bar(["Normal", "Irregular"], rc.values, color=["#4c78a8", "#e45756"])
    axes[0].set_title("Rhythm classes")
    axes[0].set_ylim(0, max(rc.values) * 1.15)
    hrv_names = ["normal", "afib", "brady", "tachy"]
    hc = hrv["condition"].value_counts().sort_index()
    axes[1].bar([hrv_names[i] for i in hc.index], hc.values, color="#4c78a8")
    axes[1].set_title("HRV classes")
    axes[1].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Synthetic class distributions")
    fig.tight_layout()
    fig.savefig(ASSETS / "class_distribution.png", dpi=140)
    plt.close(fig)
    log.info("wrote class_distribution.png")


def workflow_png():
    steps = [
        "Synthetic\ndata gen",
        "Feature\nengineering",
        "Train/test\nsplit",
        "Pipeline\n(scale+model)",
        "Held-out eval\n+ 5-fold CV",
        "Charts &\nmetrics.csv",
    ]
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.axis("off")
    n = len(steps)
    for i, s in enumerate(steps):
        x = i / n
        ax.add_patch(plt.Rectangle((x + 0.01, 0.3), 0.12, 0.4,
                                   facecolor="#dfe7f5", edgecolor="#4c78a8"))
        ax.text(x + 0.07, 0.5, s, ha="center", va="center", fontsize=9)
        if i < n - 1:
            ax.annotate("", xy=(x + 0.155, 0.5), xytext=(x + 0.13, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ML pipeline workflow (synthetic data)")
    fig.tight_layout()
    fig.savefig(ASSETS / "workflow.png", dpi=140)
    plt.close(fig)
    log.info("wrote workflow.png")


def main() -> None:
    # Confusion matrices
    y, p = _fit_predict(generate_rhythm_data(random_state=RANDOM_SEED), "label",
                        FEATURE_SETS["svm"], build_rhythm_model)
    confusion_png(y, p, ["Normal", "Irregular"], "SVM ensemble — rhythm", "confusion_matrix_svm.png")

    hr = add_health_risk_derived_features(generate_health_risk_data(random_state=RANDOM_SEED))
    y, p = _fit_predict(hr, "risk_level", FEATURE_SETS["gbm"], build_health_risk_model)
    confusion_png(y, p, ["Low", "High"], "Gradient Boosting — risk", "confusion_matrix_gbm.png")

    y, p = _fit_predict(generate_hrv_pattern_data(random_state=RANDOM_SEED), "condition",
                        FEATURE_SETS["nn"], build_hrv_model)
    confusion_png(y, p, ["normal", "afib", "brady", "tachy"], "MLP — HRV", "confusion_matrix_nn.png")

    model_comparison_png()
    class_distribution_png()
    workflow_png()
    log.info("all assets written to %s", ASSETS)


if __name__ == "__main__":
    main()
