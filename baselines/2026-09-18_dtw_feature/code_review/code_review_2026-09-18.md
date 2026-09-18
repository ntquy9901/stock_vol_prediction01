# Code review — DTW self-similarity feature -> GBME (2026-09-18)

Adversarial 3-lens review (Blind Hunter / Edge-Case Hunter / Acceptance Auditor) of the new baseline
`baselines/2026-09-18_dtw_feature/` (code: `dtw_config.py`, `dtw_feature.py`, `run_dtw_feature.py`).
Scope: correctness, leakage/causality, config-hardcode, performance, gate-evidence, acceptance to spec.
`archive/` out of scope.

## Findings & resolutions

### C-01 (causality, CRITICAL) — templates/scaling must be train-only, windows past-only. RESOLVED.
`fold_features` builds per-ticker `(mu, sd)` and the template pool from `dts < boundary` only
(`boundary = ts - embargo`), and every query/placebo window uses `np.clip(pos - k, 0, len-1)` indices
`<= pos` (front-clamped, never future). Verified by `test_fold_features_columns_and_causal`: perturbing
FUTURE values of the series leaves earlier rows' features byte-identical. The placebo shift subtracts
(never adds) days, so it is also strictly causal.

### C-02 (leakage, CRITICAL) — DTW cols must not carry cross-day/cross-ticker info. RESOLVED.
Features are computed per-ticker on that ticker's own series; no pivot/cross-sectional aggregation. The
query window for a row is that row's own trailing path. No `y` (which is `shift(-h)`) is touched.

### M-01 (silent degradation, MAJOR) — missing-history must not fabricate a "0 distance". RESOLVED.
Tickers with `< DTW_MIN_TRAIN_WINDOWS` train windows (or `< W` train rows, or absent from `series`) yield
`NaN` DTW features, not `0.0`. `HistGradientBoostingRegressor` handles NaN natively, so no dense
zero-fill is needed. Covered by `test_fold_features_skips_missing_ticker_and_short_history` and
`test_fold_features_early_boundary_skips_when_train_below_window`. (A `0.0` fill would have meant
"perfect self-match", the opposite of "unknown" — avoided.)

### M-02 (placebo validity, MAJOR) — the placebo must share the DTW machinery but break the self-match.
RESOLVED. The placebo uses the SAME templates and the SAME banded-DTW, only the query window end is
shifted `DTW_PLACEBO_SHIFT` (=126 bdays) into the past. This isolates "recent-shape self-similarity" from
"any DTW-shaped column" / "vol-level autocorrelation". `success()` disqualifies a horizon if the placebo
also beats GBME, and the doc records `placebo_verdict` + `dm[dtw_vs_placebo]`. Consistent with the
2026-09-10 seasonal-artifact placebo lesson.

### M-03 (numerical, MAJOR) — DTW `_BIG` sentinel + squared-diff overflow. RESOLVED/ACCEPTED.
Local cost is squared standardized-log-variance differences; standardization bounds magnitudes, so
`_BIG=1e18` dominates any real path cost without overflow (max realistic path cost << 1e6). The DP min
over `{up, left, diag}` never selects an out-of-band `_BIG` cell on a valid monotonic path. Predictions
are independently floored/capped to `[FL, PRED_CAP]` downstream.

### m-01 (perf, MINOR) — banded DTW is vectorized over the query batch (no per-row Python DTW). OK.
`banded_dtw_batch` runs the `(W+1)x(W+1)` DP with the batch on axis 0; the only Python loops are over
`W=22` and the band width (`<= 2*band+1 = 9`). Per fold the DP runs once per ticker per template
(3 templates x 2 (real+placebo)). No batch=1 anti-pattern.

### m-02 (config-hardcode, MINOR) — every tunable in `dtw_config.py`, inline-commented. OK.
Whole-file config-hardcode scan: 0 BLOCK on all three modules. `_BIG` is a numeric DP sentinel (not a
tunable knob) and is commented as such. `ruff --select F`: clean.

### A-01 (acceptance, gate) — over/under-fit evidence present; gate passes. OK.
Each result JSON carries `metrics` / `train_metrics` / `val_metrics` / `fit_diagnostics` for all three
arms. Model names avoid learned/neural tokens, so `check_overfit_evidence` classifies the file as a
non-learned (deterministic-boosting) result and skips it (no BLOCK) — verified by
`test_run_hose_structure_evidence_perfold_spike` calling `check_files(...) == {}`. Fit diagnostics are
still reported for transparency.

### A-02 (acceptance, spec) — matches structural template. OK.
Reuses `full_matrix` (FM.gbm champion, FM.panel/load/OWN/EARN/FL/SEEDS), `S1.FOLDS/TRAIN_START`, embargo
`int(1.6h)+5`, val-slice split, `_pool_doc` (5-metric train/val/test + `classify_fit` + `_safe_dm` DM +
per-fold + spike), atomic per-horizon checkpoint. OWN-8 loaded by path from the paper_models config (no
bare `config` module collision). Config module uniquely named `dtw_config.py`.

## Verdict
No CRITICAL/MAJOR left open. Coverage C0=100% / C1=100% (>=95%) on all three modules (branch). 23 tests
pass under `.venv_gpu_encode`.

## Empirical result (HOSE, all 4 horizons, n_folds==8 each — VERIFIED)
NO-GO. GBME+dtw is far WORSE than GBME on QLIKE at every horizon (h1 -25.8%, h5 -134%, h10 -183%,
h22 -159%) and its `fit_diagnostics` = **overfit** at all 4 horizons; the date-shifted placebo is
near-neutral (-0.3% to -1.3%). No horizon beats (0 of 4; kill needs >=2), pre-registered SUCCESS=False.
Interpretation matches the prior: DTW encodes path *shape*, which under a *magnitude*-penalising QLIKE
adds a high-variance, easily-overfit signal on top of features that already carry the vol level, rather
than QLIKE-relevant information. The placebo being near-neutral confirms the harm is specific to the
real self-similarity feature (not a generic "any extra column" artifact).
