"""Honest matplotlib plotting helpers.

Bar charts start at zero and axes are labelled plainly — no visual tricks that
exaggerate differences. None of these functions run at import time; they only act
when called, writing a figure to the requested path.

The non-interactive ``Agg`` backend is selected so the helpers work headlessly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import matplotlib

matplotlib.use("Agg")  # headless backend; safe to set once at import.

import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = [
    "plot_model_comparison",
    "plot_confusion_matrix",
    "plot_class_distribution",
]

_PathLike = Union[str, Path]


def plot_model_comparison(metrics_df: pd.DataFrame, out_path: _PathLike) -> Path:
    """Plot a grouped bar chart comparing models on their available metrics.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Metrics table (as written to ``results/metrics.csv``). Must contain a
        ``model`` column; any of ``accuracy``, ``f1``, ``roc_auc``, ``r2`` that
        are present and non-empty are plotted.
    out_path : str or pathlib.Path
        Destination image path (parent directories are created).

    Returns
    -------
    pathlib.Path
        The path the figure was written to.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_metrics = ["accuracy", "f1", "roc_auc", "r2"]
    present = [
        m
        for m in candidate_metrics
        if m in metrics_df.columns and metrics_df[m].notna().any()
    ]
    models = metrics_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.8 / max(len(present), 1)

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
    for i, metric in enumerate(present):
        values = pd.to_numeric(metrics_df[metric], errors="coerce").to_numpy()
        ax.bar(x + i * width, np.nan_to_num(values), width, label=metric)

    ax.set_xticks(x + width * (len(present) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)  # honest, zero-based axis for [0, 1] scores
    ax.set_title("Model comparison (synthetic data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    labels: Sequence[str],
    title: str,
    out_path: _PathLike,
) -> Path:
    """Plot a confusion matrix as an annotated heatmap.

    Parameters
    ----------
    cm : sequence of sequence of int
        Square confusion matrix (rows = true, columns = predicted).
    labels : sequence of str
        Class labels, in index order.
    title : str
        Figure title.
    out_path : str or pathlib.Path
        Destination image path (parent directories are created).

    Returns
    -------
    pathlib.Path
        The path the figure was written to.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.5 + len(labels), 1.5 + len(labels)))
    im = ax.imshow(cm_arr, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = cm_arr.max() / 2 if cm_arr.size else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, str(cm_arr[i, j]), ha="center", va="center",
                color="white" if cm_arr[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_class_distribution(
    y: Sequence[int],
    labels: Sequence[str],
    title: str,
    out_path: _PathLike,
) -> Path:
    """Plot the class distribution of a label vector as a zero-based bar chart.

    Parameters
    ----------
    y : sequence of int
        Integer class labels.
    labels : sequence of str
        Class names, in index order.
    title : str
        Figure title.
    out_path : str or pathlib.Path
        Destination image path (parent directories are created).

    Returns
    -------
    pathlib.Path
        The path the figure was written to.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_arr = np.asarray(y)
    counts = [int(np.sum(y_arr == i)) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.2), 5))
    ax.bar(np.arange(len(labels)), counts, color="#4C72B0")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.1 if counts else 1)  # zero-based
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
