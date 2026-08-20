# Extraction log (ENFORCE §1.6) — what was distilled into this submission folder

Each module here is a minimal, self-contained distillation of tested code from the main repo. Source
provenance:

| submission file | distilled from (main repo) |
|---|---|
| `metrics.py` | `baselines/2026-08-15_volatility/code/dm_report.py` (`_qlike`, floor 1e-8) + `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/diebold_mariano.py` (HLN DM) |
| `data_utils.py` | `scripts/sp500_crossmarket/run_sp500_crossmarket.py` (`build_ticker_features`, `make_windows`, per-ticker scaling), generalized to 80/10/10 + configurable lookback/data_root |
| `edges.py` | `baselines/2026-08-15_volatility/code/edges_glasso.py` (`glasso_partial_corr`, `precision_to_partial_corr`, Top-K), reduced to a plain train-panel API |
| `model.py` | `baselines/2026-08-15_volatility/code/model.py` (VolatilityModel) + `gat.py` (GATLayer), reduced to 3-feature HAR-LSTM-GAT + use_graph toggle (no news/gate/5-feature) |
| `baselines.py` | `baselines/classical_baselines/code/` (GARCH via `arch`) + OLS HAR |
| `snapshots.py` | new (common-date fixed-N snapshot builder + global-date 80/10/10 + universe selection) |
| `train.py`, `evaluate.py`, `run_all.py` | new glue (pooled MSE loop, test metrics + DM, orchestration) |
| `data/vn30/*` | copied from `data/processed/*_processed.csv` (VN30, 33 tickers) |
| `data/vn100/*` | copied from `data/processed/vn100_vnstock/*_processed.csv` (VN100, 104 tickers, vnstock) |
| `data/sp500/*` | NOT shipped (Yahoo-derived); regenerate via REPRODUCE.md |

Dependencies added to the GPU venv during the build: `arch` 8.0.0 (GARCH), `statsmodels`, `patsy`.
All new/adapted glue is covered by `tests/` (34 tests). Design deviation from the spec: the graph model
uses common-date snapshots with a GLOBAL-date 80/10/10 split (a GAT needs per-date snapshots) — see the
results report's Caveats.
