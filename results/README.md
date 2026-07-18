# Results

This directory holds the **verified** evaluation outputs. Every number here was produced by
re-running the training/evaluation code in this repo on **synthetic data** (fixed seed `42`),
not copied from earlier documentation.

## Files
- `metrics.csv`: one row per model with the metrics below, machine-readable.

## How these were produced
```bash
python -m src.train    --config configs/default.yaml --model all
python -m src.evaluate --config configs/default.yaml --model all   # writes metrics.csv
```

## Columns in `metrics.csv`
| Column | Meaning |
|--------|---------|
| `model` | model key (svm / gbm / nn / cardio_*) |
| `task` | classification or regression, and the label space |
| `accuracy`, `precision`, `recall`, `f1` | classification metrics (macro-averaged for the 4-class model) |
| `roc_auc` | area under ROC (binary tasks only) |
| `r2`, `mae`, `rmse` | regression metrics (cardiovascular targets only) |
| `cv_mean`, `cv_std` | 5-fold cross-validation mean ± std |
| `n_train`, `n_test` | split sizes |
| `notes` | caveats (e.g., synthetic data, class balance) |

## Read these numbers correctly
- The data is **synthetic and separable by construction**, so high scores are expected and do
  **not** indicate real-world performance. See [`../RESEARCH_NOTES.md`](../RESEARCH_NOTES.md).
- Small differences from the older saved metadata are due to newer library versions; the code,
  not the old metadata, is the source of truth.
- For health-adjacent tasks, look past accuracy: the **confusion matrix**, **recall on the
  minority/"irregular" class**, and **false-negative counts** matter more. These are plotted in
  [`../assets/`](../assets/).
