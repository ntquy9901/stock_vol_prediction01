# Code review guide (for an external AI reviewer) — 2026-08-28

**Repo (public):** https://github.com/ntquy9901/stock_vol_prediction01 — review the files IN the repo (not a zip).
**Scope exclusion:** anything under `archive/` is retired and OUT of scope. Ignore it.

## 1. What the code does (one paragraph)
Multi-horizon daily volatility forecasting on Vietnamese (VN30/VN100 primary; HOSE/HNX out-of-sample) and
S&P 500 equities. Target = daily **Parkinson variance** `sigma^2 = ln(H/L)^2/(4 ln2)`. Three models share the
same five node features `[daily-Parkinson, HAR-weekly, HAR-monthly, market-Parkinson, 20d volume z-score]`:
a linear **HAR-X** (pooled OLS), a **LSTM**, and an **LSTM+GAT** (directed volume->Parkinson Top-5 weighted
2-hop graph). Benchmark: **GARCH(1,1)**. Evaluation: MSE/RMSE/MAE/QLIKE/R^2 on a **masked union-of-dates panel**,
5 seeds, chronological 80/10/10 split with purge, significance by **date-clustered Diebold-Mariano** (HLN).

## 2. Critical-path code (review in this order)
| Priority | File | What it is / what to check |
|---|---|---|
| **P1** | `baselines/2026-08-21_har_anchored_residual/code/run_masked_rich.py` | DEFINES the paper model `MaskedRichNet` + `WeightedGATLayer`, fits HAR-X, `train_masked_rich()`, `run()`. **Check:** per-seed reporting `seed_metric_stats()` (mean of per-seed metrics, NOT metric of the seed-averaged ensemble); `_ens()` used only for the DM forecast; `run(out_subdir=...)` writes to a separate tree; relative floor `1e-2*t_mean` applied to every model; no leakage. |
| **P1** | `baselines/2026-08-21_har_anchored_residual/code/masked_rich.py` | `build_masked_rich()` (5-feature masked panel, per-node train-only scalers, directed vol->PK edge + corr edge estimated on TRAIN rows only), `WeightedGATLayer`, `_volume_zscore_wide` (overnight/coverage). **Check:** train-only invariance (edges/scalers use only train slice), volume z-score fail-loud guard (`>2` missing files raises). |
| **P1** | `submission/soict_lstm_gat/baselines.py` | `har_fit/har_predict` (OLS); **`garch_forecast`** with **persistence cap 0.999 via variance targeting** (`_cap_params`, `_capped_forecast_path`) so near-unit-root (IGARCH) fits do not diverge; pseudo-returns are random-signed (symmetric GARCH variance is ~sign-invariant). **Check:** the cap logic + the analytic multi-step path. |
| **P2** | `baselines/2026-08-21_har_anchored_residual/code/stats.py` | `date_clustered_dm` (HLN small-sample correction, HAC lag h-1). **Check:** clustering by trading date; shape guard. |
| **P2** | `submission/soict_lstm_gat/metrics.py` | mse/rmse/mae/**qlike**(y,p,floor)/r2 + `per_obs_qlike`. **Check:** QLIKE floor applied identically across compared models. |
| **P2** | `scripts/garch_masked/compute_garch_masked.py` | Adds GARCH to the masked panel; validation-aligned (skips the val block); HAR-X **basis guard** (recomputed HAR-X QLIKE must match stored). |
| **P2** | `scripts/garch_masked/run_oos_suite.py` | OOS driver (HOSE/HNX/S&P 500), screened universe, resumable. |
| **P2** | `scripts/garch_masked/floor_sensitivity.py` | `screen_files` (liquidity+history: >=250 rows, <=50% H==L days) + common-floor rescoring. **Note:** the `== 0.0` zero-variance test matters (see review guide caveats). |
| **P3** | `scripts/eda/test_diagnostics.py` | Per-ticker error-analysis EDA (why QLIKE varies across stocks: low-range/floored days, volume coverage, SSE concentration) -> HTML. |
| **P3** | `scripts/eda/volatility_estimators.py` | 5 daily variance estimators from OHLCV (Parkinson/GK/RS/close2close/rs_overnight/**yz_daily**/yz_rma20); overnight winsorize +-0.20 + `prev_close>0` guard (unadjusted-split artifact). |
| **P3** | `scripts/eda/estimator_forecast_ablation.py` | HAR-X forecast under each estimator target; `screened_tickers` pins the universe. |
| **P3** | `scripts/eda/run_yz_robustness.py` | Full-pipeline Parkinson-vs-Yang-Zhang robustness (writes to `results/masked_rich_yz/`). |

## 3. What to verify (correctness priorities)
1. **No leakage:** every scaler, both graph edges, and the HAR-X/GARCH fits use TRAIN rows only. Regression test:
   `baselines/2026-08-21_har_anchored_residual/code/test_masked_rich.py::test_train_only_invariance_no_leakage`
   (perturb test-region inputs -> train edges + target scaler unchanged).
2. **Per-seed vs ensemble:** reported learned-model metrics are the MEAN of seed-level metrics
   (`metrics_per_seed`), not the metric of the seed-averaged prediction. The DM is on the 5-seed ensemble
   forecast (labelled as such). `config.batch_size` is recorded in each result.json for reproducibility.
3. **GARCH not diverging:** `garch_forecast` caps persistence; a near-unit-root fit converges to a finite
   variance. Verify `_capped_forecast_path` matches the direct GARCH recursion for a stationary fit.
4. **QLIKE / floor:** the per-node relative floor `1e-2*t_mean` and the QLIKE floor `1e-8` are applied
   identically to HAR-X, GARCH, LSTM, LSTM+GAT.
5. **DM:** date-clustered, HLN-corrected; the paper explains the naive per-obs test overstates significance by
   ~sqrt(#stocks).
6. **No silent degradation:** volume z-score fails loud above a bounded allowlist; the EDA reports per-ticker
   volume coverage. Volatility overnight is winsorized because VN raw prices are NOT split-adjusted (S&P 500
   already is).

## 4. Data, tests, reproduce
- **Interpreter:** `.venv_gpu_encode/Scripts/python.exe` (torch+cuda+arch); base `python3` (3.14, no cuda torch)
  runs the numpy/HAR-X/EDA parts. `PYTHONIOENCODING=utf-8`.
- **Results the paper reads:** `results/masked_rich_floor1e2/<ds>_h<h>/result.json` (5 panels x 4 horizons);
  robustness: `results/masked_rich_yz/<target>/<ds>_h<h>/result.json`.
- **Tests:** `python -m pytest baselines/2026-08-21_har_anchored_residual/code/test_masked_rich.py
  submission/soict_lstm_gat/tests/ scripts/garch_masked/test_*.py scripts/eda/test_*.py -q`.
- **Pre-push quality gate:** `scripts/git_hooks/pre-push` (TDD gate + pytest + ruff + Pandera + baseline tests).

## 5. Context reports (decisions + known issues, read for background)
- `docs/reports/2026-08-23_0830_garch_persistence_cap_report.md` — why GARCH was capped (IGARCH divergence, not a bug).
- `docs/reports/2026-08-23_1400_perseed_reproducibility_report.md` — per-seed fix (F1) + pinned config.
- `docs/reports/2026-08-23_1500_test_eda_diagnostics_report.md` — per-ticker error analysis (why QLIKE varies).
- `docs/reports/2026-08-23_1600_volatility_estimator_research.md` — Parkinson vs GK/RS/Yang-Zhang (deep research + ablation).
- `docs/reports/2026-08-28_1000_yz_proxy_robustness.md` — full-pipeline volatility-proxy robustness.
- `docs/reports/2026-08-28_ALL_metrics_review.md` — consolidated all-metrics numbers (main + DM + robustness + vol-types).
- Papers: `docs/paper/soict_harlstmgat{,_extended,_crossmarket}.tex`.

## 6. Known caveats (do not re-flag as new bugs)
- Learned metrics are **5-seed means (+std)**, DM is on the **5-seed ensemble** forecast (both stated).
- GARCH pseudo-returns are random-signed (variance is ~sign-invariant); GARCH is a dominated benchmark by design.
- VN raw OHLCV is **not split-adjusted** (overnight winsorized); S&P 500 already is (verified vs Yahoo adj close).
- The graph (LSTM+GAT vs LSTM) QLIKE effect is small and within per-seed dispersion / proxy-dependent -- stated honestly.
- Volatility-proxy robustness is full-pipeline at h1 (all horizons in progress); vol-type ablation is HAR-X-only.
