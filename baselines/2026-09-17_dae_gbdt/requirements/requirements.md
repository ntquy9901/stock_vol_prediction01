# Requirements — GBME+DAE (Hướng 2: Denoising-Autoencoder representation → GBDT)

## Objective
Falsify (or confirm) whether an unsupervised **denoising autoencoder (DAE)** bottleneck embedding of the
champion own-history feature set — the Kaggle-Grandmaster / Jahrer "DAE-representation → GBDT" paradigm —
adds out-of-sample forecasting value over the delivered gamma-GBM champion **GBME (GBM + earnings)** on HOSE
daily Parkinson-variance forecasting.

## Input
- HOSE enriched panel `data/processed_enriched/hose/*.csv` (405 tickers), loaded via `full_matrix.load`.
- Real crawled VN earnings dates `results/gamma_gbm/hose_earnings_combined.parquet` (402 tickers), merged via
  `run_dae._load_earn` exactly like `full_compare`.
- Champion feature set: **OWN-8** (`full_matrix.OWN` minus `rq`, single-sourced from
  `baselines/2026-09-13_paper_models/code/config.py::own_set`) + **EARN-4** earnings-proximity features.

## Method (what is built)
1. **DAE:** small PyTorch autoencoder over the standardised OWN-8+EARN features, trained to reconstruct the
   clean input from a **swap-noise**-corrupted copy (each cell replaced with prob `SWAP_RATE` by another
   row's value in the same column). Reconstruction MSE loss, batched on GPU. Bottleneck `Z ∈ R^K` extracted.
2. **Frozen-basis embedding:** ONE DAE trained on the burn-in window (first eligible fold's causal train
   rows), FROZEN, embeds every panel row → a single shared latent basis (avoids the per-fold neural-basis
   drift that detonated the sibling `2026-09-14_gbm_gnn_embed` long-horizon QLIKE).
3. **Hybrid GBM:** feed `[OWN-8 | EARN | Z]` into the champion
   `HistGradientBoostingRegressor(loss="gamma", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
   l2_regularization=1.0)` (reuse `full_matrix.gbm`), seed-ensembled, predictions clipped to `[FL, PRED_CAP]`.
4. **Comparison:** GBME (own+earn, no DAE) vs GBME+DAE (own+earn+Z) over the shared VN walk-forward
   (`vn_gbm_graph_stage1.FOLDS`, `TRAIN_START`, embargo `int(1.6·h)+5`), all 4 horizons, seed-ensembled,
   date-clustered Diebold-Mariano, per-fold QLIKE + regime-spike robustness.

## Output
- `results/gamma_gbm/dae_hose_h{1,5,10,22}.json`, one per horizon, atomic per-fold checkpoint. Each carries
  pooled 5-metric train/val/test for both models, `fit_diagnostics`, `learning_curves` (per-epoch DAE
  reconstruction loss), DM + gain + verdict, per-fold QLIKE and spike robustness.

## Success criteria (pre-registered kill criterion)
- **GO** iff GBME+DAE beats GBME at BOTH h1 and h5: QLIKE gain > `GAIN_MIN` (0.0) AND date-clustered DM
  p < `DM_ALPHA` (0.05), and the sign survives spike-window exclusion.
- **NO-GO** otherwise (the expected outcome, per the sibling GNN-embed falsification: a learned neural
  embedding fed to the same 8-feature own-history gamma-GBM is noise).

## Go / No-Go
This is a **falsification** baseline; the value is a clean, gate-clean negative result. NO-GO is a valid,
publishable finding (it strengthens the paper's "own-history HAR/GBM is hard to beat on QLIKE" thesis). A
GO would be a genuine positive worth escalating.

## Non-goals
- No SP500 run required (HOSE is the target; the driver supports `sp500` for parity/testing only).
- No hyper-parameter search on the DAE (a single principled configuration; this is a falsification, not a
  tuning study).
