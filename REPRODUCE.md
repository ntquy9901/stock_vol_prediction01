# Reproducing the results

This repository is self-contained on `master`: the data, the data pipeline, the model/training code,
and evaluation are all present. Reviewers can either (A) quickly re-evaluate from provided prediction
dumps, or (B) retrain the models end-to-end.

> Status: the code paths below are smoke-verified to run for **training and evaluation** on `master`.
> A full 3-seed re-run on the cleaned data and the corresponding refresh of the paper's numbers is a
> separate pending step; the commands here are the reproduction entry points.

## 1. Environment

- Python 3.10, PyTorch (CUDA build) + pandas/numpy/scikit-learn. The project uses a GPU virtualenv at
  `.venv_gpu_encode/` (torch 2.6 + cu124). To recreate: `pip install -r requirements.txt` (a CUDA
  GPU is recommended for training; evaluation of dumps is CPU-only).
- Optional: `statsmodels` (only for one Diebold-Mariano cross-check unit test; not needed to run the
  reproduction itself).
- All commands are run from the repository root.

## 2. Data (already included)

Processed Parkinson-variance series are in `data/processed/<TICKER>_processed.csv` (33 VN30 tickers,
columns `date,parkinson_volatility`, through 2026-08-14). To regenerate from raw:

```bash
# raw OHLCV in data/raw/prices/<TICKER>_ohlcv.csv  (VN30). Optional cleaning + (re)process:
python -m src.data.verify_raw_prices data/raw/prices          # column audit (schema/OHLC/volume=0)
python -m src.data.clean_ohlc        data/raw/prices          # positive-aware OHLC repair (idempotent)
python -m src.common.process_parkinson_pipeline               # raw -> data/processed (Parkinson variance)
```

The same three scripts work for any universe by pointing at its folder (e.g. VN100:
`... data/raw/prices/vn100` and `--raw data/raw/prices/vn100 --out data/processed/vn100`).

## 3. Path A — quick re-evaluation from prediction dumps (CPU, seconds)

Held-out test prediction dumps are provided under `results/volatility_ablation_h{h}_seed{s}_<TS>/`.
Re-run the Diebold-Mariano comparison (HLN small-sample correction, HAC lag h-1) over the 3 seeds:

```bash
PY=.venv_gpu_encode/Scripts/python.exe   # or any python with numpy
$PY baselines/2026-08-15_volatility/code/dm_report.py <TS> <h> 42,123,2026
# e.g. ... dm_report.py 2026-08-15_085544_loo 5 42,123,2026
```

This prints the DM table (FULL vs HAR / minus_graph / minus_gate / minus_news / LSTM_only) on QLIKE,
squared-error (MSE/RMSE/R2 family) and absolute-error (MAE) — the significance tables reported in the
paper. Per-rung metrics are in each run's `ladder_metrics.json`.

## 4. Path B — full retraining (GPU)

Leave-one-out ablation (FULL and each minus-one variant) + the price-only LSTM anchor, per seed
(paper uses seeds 42, 123, 2026), across horizons 1/5/10/22:

```bash
PY=.venv_gpu_encode/Scripts/python.exe
CODE=baselines/2026-08-15_volatility/code
for S in 42 123 2026; do
  TS=run_seed${S}
  # one call trains every rung: FULL, minus_graph, minus_gate, minus_news, lstm_only (+ HAR baseline)
  $PY $CODE/run_ablation.py    $TS cuda $S 12 1 5 10 22
done
# then evaluate (Path A) with the matching TS across the 3 seeds:
$PY $CODE/dm_report.py run_seed42 5 42   # single seed, or seed-ensemble by passing a shared TS scheme
```

Notes:
- `run_ablation.py <TS> <device> <seed> <epochs> <horizons...>` — trains each rung and writes
  `results/volatility_ablation_h{h}_seed{seed}_<TS>/` (per-rung test dumps + `ladder_metrics.json`).
- Convergence is early-stopped (patience 3, min 6) under the epoch cap; the paper used a 12-epoch cap.
- `run_retrain_trainval.py` provides the train+val-merged variant (fixed epochs, no early stop).

## 5. Model (what is trained)

`model.py::VolatilityModel` = three parallel branches on the pooled masked graph snapshots: a price
LSTM, a multi-head GAT over a directed volume→volatility (vol→PK) lead-lag edge, and a gated news
branch (PhoBERT features); concatenated → head → softplus positivity floor. HAR is a pooled linear
baseline. See `baselines/2026-08-15_volatility/design/` and the explainer docs under
`docs/paper/explainers/`.

## 6. Layout

- `baselines/2026-08-15_volatility/code/` — model, training, evaluation (entry points above).
- `baselines/2026-08-{08,11,14}_*/code/` — reused basis builders (pooled manifest, edges, features)
  that the volatility code imports.
- `src/common/`, `src/data/` — data pipeline (Parkinson processing, OHLC cleaning, verification).
- `tests/` — data-quality tests (enforced on every push via `scripts/git_hooks/pre-push`).
- `data/`, `results/`, `docs/paper/` — data, run outputs/dumps, paper drafts + explainers.
