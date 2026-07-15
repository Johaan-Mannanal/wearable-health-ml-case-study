# Notebooks

> ⚠️ Synthetic-data research project — not a medical device. See the root
> [README.md](../README.md) and [MODEL_CARD.md](../MODEL_CARD.md).

These notebooks are **exploratory**. The maintained, tested, reproducible pipeline lives in
[`src/`](../src) and is what the README's verified results come from
(`python -m src.train` / `python -m src.evaluate`). Treat the notebooks as a lab notebook,
not as the source of truth.

| Notebook | Purpose | Status |
|----------|---------|--------|
| `Support_Vector_Machine.ipynb` | Baseline SVM exploration | exploratory |
| `Support_Vector_Machine_Improved.ipynb` | Ensemble SVM iteration | exploratory (JSON repaired) |
| `Gradient_Boosting_Machine.ipynb` | GBM baseline | exploratory |
| `Enhanced_Gradient_Boosting_Machine.ipynb` | GBM with engineered features | exploratory |
| `Convolutional_Neural_Network.ipynb` | CNN experiment (TensorFlow) | exploratory — the shipped NN is an MLP in `src/`, not a CNN |
| `HRV_CNN_Analysis.ipynb` | HRV sequence analysis | exploratory (TensorFlow) |

## Notes for reviewers
- All data used in the notebooks is **synthetic** (generated in-notebook or via `src/`).
- Some notebooks require TensorFlow (`pip install "tensorflow>=2.13"`); the core `src/`
  pipeline does **not**.
- Execution counts were cleared/were never fully run top-to-bottom; if you run them, do so from
  the repo root so `from src import ...` resolves.

## Known remaining cleanup (TODO)
- [ ] Add richer Markdown narrative and trim large raw outputs.
- [ ] Port notebook logic to call `src/` functions instead of duplicating data-generation code.
- [ ] Verify each notebook runs top-to-bottom in a clean kernel and commit with clean outputs.
