# VN30 Volatility Forecasting — Code Submission

Self-contained code to reproduce the paper on multi-horizon Parkinson-variance
volatility forecasting for VN30 stocks. This folder preserves the repository's
relative directory layout so the cross-module import bootstraps
(`Path(__file__).resolve().parents[N] / "baselines" / ...`) resolve entirely
within this tree — no external repo checkout is needed to run the code.

## What the model is

- **Target:** multi-horizon (h in {1, 5, 10, 22}) Parkinson **variance** (σ², not σ)
  of VN30 stocks, forecast from daily OHLCV.
- **Delivered model (FULL):** a 3-branch network
  1. a **price LSTM** over 5 per-ticker node features,
  2. a **GAT** (self-written multi-head graph attention) over a directed
     volume-shock -> next-day Parkinson (`vol -> PK`) edge, and
  3. a **gated PhoBERT news** branch with a per-ticker learned gate.
- **Baseline:** pooled **HAR** linear regression (daily/weekly/monthly).
- **Ablation:** leave-one-out — build FULL, then retrain each variant with exactly
  one component removed (`minus_graph`, `minus_gate`, `minus_news`, `lstm_only`)
  so every effect is measured on the same footing.

## Folder layout

```
deliverables_20260817/
  baselines/
    2026-08-15_volatility/          delivered model + tests (main entry points)
    2026-08-14_pooled_news_edanode_gnn/   combo_ladder.py (combo basis builder)
    2026-08-11_eda_gnn_baseline/    EDA node features (features.py) + vol->PK edges (edges.py)
    2026-08-08_pooled_news_gnn_ablation_baseline/  pooled data/scaling/train + Diebold-Mariano
  src/common/                       shared utils (evaluation, Parkinson pipeline, scalers, split)
  paper/                            paper markdown (final + GNNHAR P1/P2/P3 addendum)
  reports/                          key result reports (P1/P2/P3, glasso edge vs vol->PK, contribution HTML)
  requirements.txt
  REPRODUCE.md                      end-to-end reproduction notes (from repo root)
  README.md                         this file
```

## Key entry-point scripts

All under `baselines/2026-08-15_volatility/code/`. Folder names contain `-`, so run
with `python <path>/<script>.py` (not `python -m`); each script bootstraps `sys.path`.

- `run_ablation.py` — leave-one-out ablation across horizons (FULL vs minus_graph /
  minus_gate / minus_news / lstm_only, plus HAR). Primary experiment.
- `run_volatility.py` — build the volatility basis and train/evaluate the FULL model.
- `dm_report.py` — Diebold-Mariano pairwise significance tests.
- `regime_report.py` — regime-split (calm vs turbulent) evaluation.
- `harx_report.py` — fair HAR-X baseline report.
- `mad_report.py` — mean-absolute-deviation edge / diagnostic report.
- `edges_glasso.py` / `run_glasso_edge.py` — graphical-LASSO edge construction and
  its comparison against the directed `vol -> PK` edge.

## Data required (NOT included — supplied separately by reviewers)

Data is excluded from this submission due to size. To run the code, place:

- Processed VN30 price/volatility data at `data/processed/*_processed.csv`
  (one file per ticker; Parkinson-variance + HAR features, produced by
  `src/common/process_parkinson_pipeline.py`).
- The PhoBERT news panel at `data/features/dual_group_news_panel.parquet`.

Paths are resolved relative to the deliverables root (the `parents[N]` bootstrap),
so create `data/processed/` and `data/features/` inside this folder, or point the
scripts at your data location. See `REPRODUCE.md` for the full pipeline.

## Setup

```bash
pip install -r requirements.txt   # install a torch build matching your CUDA/CPU
python baselines/2026-08-15_volatility/code/run_ablation.py <TIMESTAMP>
pytest baselines/2026-08-15_volatility/test/ -v
```
