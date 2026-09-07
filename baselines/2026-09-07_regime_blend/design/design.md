# Design — regime-conditional blend

## Data flow
1. `load_folds(market, h, deep)` reads the anchor=none cells, pivots to one row per (fold, ticker, date)
   with realized `y`, `harx`, `deep` forecasts, for val and test splits.
2. For each fold k: `fit_fold(val_k)` → θ (Nelder–Mead on val QLIKE) + fold-val standardization (μ, σ).
3. `apply_fold(test_k, params)` → blended forecast on test_k using the fold-val standardization.
4. Concatenate all test folds; `evaluate` computes the 5 metrics + DM for deep / harx / blend.

## Key decisions
- **Convex blend, not residual anchor.** `f = w·deep + (1−w)·harx`, `w ∈ (0,1)` via sigmoid — bounded,
  differentiable, `w→1` recovers deep, `w→0` recovers HAR-X. Avoids the multiplicative training anchor
  (`anchor=harx`) that was globally worse.
- **Causal gate features only.** Both `harx` and `deep` are made from data through t−h, so any function of
  them is causal. The target-day realized decile is used ONLY for post-hoc diagnosis, never in the gate.
- **Per-fold refit.** Weights for fold k come from fold k's own validation block; no other fold's data
  (which may be later in calendar time) touches fold k — removes the look-ahead the earlier pooled spike had.
- **Smooth logistic weight (2 features) instead of hard bins.** Robust to the thin per-fold storm-regime
  sample; three parameters fit per fold.
- **Shared floor.** `pipeline_config.QLIKE_FLOOR` for every QLIKE (deep, harx, blend) — identical basis
  (repeat of the H2 lesson: positivity floor must match across compared models).

## Gates (SDD)
- Simplicity: a 3-parameter post-hoc layer over frozen forecasts; no new network, no new data.
- Anti-abstraction: reuses `submission/soict_lstm_gat/metrics.py` (5 metrics + DM) and
  `baselines/2026-08-21_har_anchored_residual/code/stats.py` (date-clustered DM) directly.
- Performance/batching: fully vectorised numpy over the panel; the only loop is 7 folds × 4 horizons ×
  2 markets × 2 deep bases of Nelder–Mead on validation arrays (seconds); no per-item Python loop, no GPU.

## Files
- `code/regime_blend.py` — load / fit / apply / evaluate (pure, testable).
- `code/run_regime_blend.py` — driver over markets × horizons × deep bases; writes
  `results/qlike_anchor/regime_blend_result.json`, prints a verdict table.
- `test/test_regime_blend.py` — pytest: sigmoid bounds, blend recovers deep at w=1 / harx at w=0, fit lowers
  val QLIKE, causal (no target leakage), a real-data slice smoke run.

## Overfit evidence
Result JSON carries `val_metrics` + `test_metrics` (QLIKE) per cell and per-fold mean weight, so val→test
degradation is inspectable (the blend has only 3 fitted params/fold, low overfit surface).
