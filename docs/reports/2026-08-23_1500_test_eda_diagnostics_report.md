# Summary of update — reusable test-set error-analysis EDA + review findings (2026-08-23)

## What this is
A reusable diagnostic EDA that answers **why the pooled test metrics vary across stocks**. It breaks the
masked test set down per ticker, computes data characteristics and per-model errors, correlates them, and
renders a self-contained HTML report. Yes — this is error-analysis / diagnostic EDA (post-hoc, tied to the
reported metrics).

## Files
| Path | Purpose |
|---|---|
| `scripts/eda/test_diagnostics.py` | Reusable CLI: per-ticker characteristics + HAR-X/GARCH per-ticker errors + Spearman correlations + HTML render |
| `scripts/eda/test_test_diagnostics.py` | 4 smoke/unit tests (synthetic panel) — all pass |
| `docs/reports/eda/2026-08-23_h1_test_diagnostics.html` | Generated report, horizon h1, 5 panels (self-contained, plots embedded) |

**Re-run:** `python scripts/eda/test_diagnostics.py --panels vn30 vn100 hose hnx sp500 --horizon 1 --out <file>.html`
(use `.venv_gpu_encode/Scripts/python.exe` so GARCH uses the `arch` fit; base Python falls back to the
mean forecast for GARCH). Any horizon/panel subset works.

## Metrics computed (per ticker)
- **Data characteristics:** n_test days, mean/median Parkinson variance, vol-of-vol (std/mean), **low_range_frac**
  (fraction of days at/under the per-node floor), lag-1 autocorrelation, skew, **vol_missing_frac** (fraction of
  raw volume rows that are NaN/zero — data-coverage).
- **Per-model error (HAR-X, GARCH):** MSE, MAE, QLIKE, R², clip_rate (fraction floored), and **share of the
  pooled SSE** (which stocks dominate the pooled MSE).
- **Spearman correlation** of per-ticker QLIKE/MSE against each characteristic.
- **Visuals:** cross-panel summary; per-panel worst-ticker tables; scatter QLIKE-vs-{low_range_frac, vol_of_vol};
  log-log MSE-vs-variance; Lorenz curve of SSE concentration.

## Findings (h1)
Pooled HAR-X QLIKE tracks liquidity: SP500 0.359 < VN100 0.512 ≈ VN30 0.516 < HOSE 1.430 < HNX 1.977.

**QLIKE is driven by low-range / floored days** (Spearman QLIKE vs low_range_frac): HOSE **+0.92**, HNX **+0.95**,
VN100 +0.47, VN30 +0.34, SP500 **−0.02**. On the illiquid exchanges, stocks whose Parkinson variance sits at the
floor on most days (H≈L, no intraday range) drive QLIKE up — the worst tickers (HOSE L10 QLIKE 8.56 with 87% of
days at floor; HNX S55 8.68 with 92%) are near-untraded names. SP500 has essentially no floored days, so its
QLIKE is low and uniform across stocks.

**Volume coverage matters (confirms review finding F4):** HOSE QLIKE vs vol_missing_frac Spearman **+0.61** (HNX
+0.49) — tickers with more missing/zero raw volume have higher error, so the silently-neutralized volume feature
is a real data-quality signal on the illiquid panels.

**MSE is concentrated in a few high-variance stocks:** top-5 tickers carry 33% of the pooled SSE on VN30, 17% on
SP500, 15% on HNX. MSE correlates with the variance level, QLIKE does not.

**Where to improve:** (1) a scale-aware / liquidity-aware positivity floor (or a stricter liquidity screen) would
remove the low-range-day QLIKE inflation that dominates HOSE/HNX; (2) fill or mask missing-volume tickers (F4);
(3) the pooled MSE is a few-stock story, so report per-ticker dispersion (already added via per-seed std).

## Review findings (Codex rerun-04) addressed here
- **F3 (leakage tests):** added `test_train_only_invariance_no_leakage` — perturbs the test-region inputs and
  asserts the train graph adjacencies (`adj_vol2pk`, `adj_corr`) and the per-node train target scaler are
  unchanged (train-only), while the test targets do change. Passes.
- **F4 (silent volume→0):** the EDA now reports `vol_missing_frac` per ticker (missing/zero raw volume) and shows
  it correlates with error — surfacing the coverage that was previously silent. (Missing FILES already fail-loud
  above 2; this adds cell-level visibility.)
- **F6 (stale spec docs):** added a SUPERSEDED banner to the baseline's `requirements/requirements.md` and
  `design/design.md` (they describe the earlier HAR-anchored ladder, not the delivered masked-rich study).
- **F1 / F2 / F5:** already resolved in the prior per-seed / GARCH-cap work (metrics_per_seed + config present;
  GARCH sign-invariance disclosed; README/paper agree on 454 test dates). The review ran on the old package.
- **F7:** reviewer-environment interpreter path; the working interpreter is `.venv_gpu_encode/Scripts/python.exe`
  (documented in `deliverables_20260823/REPRODUCE.md`).

## Tests
`scripts/eda/test_test_diagnostics.py` 4/4; `test_masked_rich.py` 12/12 (incl. the new leakage test). No data or
model-output change — this is analysis-only.

## Code review
The EDA reuses the delivered `build_masked_rich` + deterministic HAR-X/GARCH predictors; no leakage (train-only
edges/scalers, verified by the new test). Per-ticker SSE shares sum to 1 (partition check in tests).
