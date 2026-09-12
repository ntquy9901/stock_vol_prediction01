# Design — Complex-Network Topology baseline (2026-09-12)

Full plan: `docs/reports/2026-09-12_complex_network_gbm_design.md` (advisor-approved, source of truth). This
note records the code structure and the three SDD gates.

## Modules (`code/`)

| File | Role |
|---|---|
| `config.py` | Single source of truth for tunable constants (WIN/STEP/L/THR/ALPHA/α-grid/RF_KW/horizons/TOPO) — avoids the config-hardcode gate. |
| `config.py` (cont.) | Adds `WIN_ROBUST=132`, `RAW_VOL_DIR` for the raw OHLCV per market. |
| `volume_io.py` | `load_log_volume(market, tickers)` — REAL `ln(volume)` from raw OHLCV (Refinement 1, replaces `volume_zscore_22`); non-positive volume → NaN. |
| `topology.py` | `global_feats(C)` (6 metrics; eigenvector centrality dropped 2026-09-13 — ~0 MI with target + perfectly collinear, see feature screen); `build_topo_windows(frames, market, alpha, win)` → `(F, n_by_window)` (one 6-vec + common-ticker count N per causal window, Refinement 2); `build_topo_daily` (windows reindexed + ffilled for Exp B). Volume matrix uses `load_log_volume`. |
| `market_index.py` | `load_index` (VNINDEX/GSPC close+volume+log return), `future_targets` (strictly-future L-day idx_vol/idx_ret/idx_lnvol + target_end). |
| `run_index.py` | Experiment A: `build_sample` → `walk_forward` (causal expanding, target_end embargo, train-only scaler, per-fold feature importance — RF `feature_importances_` / standardized LR coef, Refinement 4) → `evaluate_sample`/`robustness`; LinearRegression + RandomForest; R²/RMSE + N-per-window stats. |
| `run_gbm.py` | Experiment B: broadcast daily topology onto stock panels (per-ticker ffill) → gamma-GBM (GBM vs GBM+topo) → pooled QLIKE + date-clustered DM + overfit evidence. |
| `diag_gbm.py` | Experiment-B diagnostic ("why topo does not help"): per-horizon final-fold QLIKE permutation importance (16 features), GBM-vs-GBM+topo prediction/error correlation, and train-vs-test QLIKE. |
| `build_topo_diag_html.py` | Renders `diag_gbm` + the head-to-head QLIKE/DM to a standalone HTML (`docs/reports/2026-09-12_complex_network_<market>_why_topo_hurts.html`). |

## Data flow

Exp A: stock frames → `build_topo_windows(alpha, win)` (7-vec per window, strictly-past) → join
`future_targets` (index over `[d0, d0+L)`) → sample S → causal walk-forward (train where `target_end ≤ d0_j`)
→ OOS R²/RMSE per model×target + robustness (α-grid, win=132).

Exp B: `build_topo_daily` → merge onto every ticker by date → `FM.panel` per h → expanding folds
(`S1.FOLDS`, embargo `int(h·1.6)+5`d) → seed-ensembled gamma-GBM → per-obs QLIKE → date-clustered DM.

## SDD gates

- **Simplicity Gate:** no new abstractions; reuses `FM.gbm`/`FM.panel`/`FM.load`, `M.per_obs_qlike`,
  `ST.date_clustered_dm`, and copies the paper's `global_feats`/window logic verbatim. Constants centralised
  in one config. PASS.
- **Anti-Abstraction Gate:** uses sklearn (`LinearRegression`, `RandomForestRegressor`, `StandardScaler`) and
  networkx directly; no wrappers. PASS.
- **Performance/Batching Gate:** topology is recomputed once per `STEP` (=22 trading days), NOT per day —
  ~90 windows for HOSE. Betweenness centrality cost is bounded by ≤~300-620 nodes × ~90 windows
  (seconds-to-minutes, one-off feature build), then cached as a small per-window/per-day table. Exp B reuses
  the project's seed-ensembled gamma-GBM (vectorised sklearn HistGradientBoosting). No batch=1 hot loop over
  samples on the main thread beyond the unavoidable per-window graph construction. PASS.
