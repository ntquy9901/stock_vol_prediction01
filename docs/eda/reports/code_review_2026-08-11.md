# Adversarial Code Review - graph_eda (2026-08-11)

3-layer adversarial review (Blind Hunter + Edge Case Hunter + Acceptance Auditor) of the
`graph_eda/` leakage-safe GNN-justification EDA module. Focus: look-ahead leakage,
vectorised cross-correlation correctness, metric bugs, NaN handling, determinism.

## Verified safe (no action)
- Lead-lag alignment `Corr(X_i(t), X_j(t+k))` with `.iloc[:-k]` truncation drops exactly the
  shifted NaN tail; no row past the cutoff leaks. Spearman ranks after truncation. (V1)
- Vectorised correlation `xz.T @ yz / n` with ddof=0 standardisation equals exact Pearson r;
  no off-by-one. (V2)
- `chrono_split` masks disjoint + covering; train strictly precedes val precedes test; runner
  re-asserts `train.max() < test.min()`. (V3)
- `market_adjust_residuals` fits alpha/beta on TRAIN rows only. (V4)
- Target `shift(-horizon)` strictly future; neighbour identity + scalers + prediction floor all
  TRAIN-only; val sits between train and test so no train/test target overlap. (V5)
- All stochastic paths seeded; deterministic linear algebra. (V6)
- Rolling snapshots trailing-only; `assert_snapshot_no_lookahead` per snapshot. (V7)

## Findings and resolutions

| ID | Sev | Finding | Resolution |
|---|---|---|---|
| M1 | MAJOR | Gate 6/7 baselines used a per-baseline `valid` mask, so A/B/C were pooled over different rows/stocks -> not comparable. | `run_baselines` now builds ONE shared valid mask (own AND market AND neighbour features AND target present) applied to all baselines. Verified: pooled n identical (6369) across A/B/C; test `test_run_baselines_shared_mask_returns_per_stock` asserts equal n. |
| M2 | MAJOR | Decisive gate was a bare pooled-RMSE `<` (single seed/split, no significance) -> a ~1e-6 tie could flip the verdict. | Added per-stock RMSE table + win-rate + two-sided sign (binomial) test. Gates now require a >0.5% pooled margin AND a stock majority. Result: C beats A in 24% of stocks (p=0.005, significantly worse); C beats B in 42% (p=0.49, coin-flip). Verdict (NO GNN) unchanged and now significance-backed. |
| N1 | MINOR | Pooled DirAcc diffed across stock boundaries (~32 meaningless comparisons). | DirAcc now computed per stock then averaged (`_diracc` + `per_stock_diracc` arg). |
| N4 | MINOR | QLIKE clip-to-1e-12 blew up on zero-variance (high==low) target days. | `_metrics` excludes non-positive targets from QLIKE; predictions floored at train 1st-percentile. QLIKE for C now 0.418 (was 2.3e5). |
| N3 | MINOR | `cross_corr_matrix` divided by std with no zero guard (constant column -> inf). | Guard added: `std==0 -> NaN` (undefined correlation, not inf). |
| N2 | MINOR | Global complete-case row dropping in `cross_corr_matrix` shrinks n. | Accepted as documented; n is reported. Noted as a limitation. |
| N5 | MINOR | Pooled R^2 uses one cross-sectional mean (rewards level prediction). | Presentational only (gates use RMSE + sign test). Report flags it. |
| N6 | MINOR | Market median factor includes the target stock (self-weight ~1/33). | Negligible with 33 stocks; noted as a limitation. |

All MAJOR and metric-correctness findings fixed and covered by tests. No look-ahead leakage that
would falsely support the GNN case was found.

## Addendum - incremental.py (added after the review, self-reviewed to the same standard)

`graph_eda/incremental.py` (node-feature + edge-definition ranking over HAR) was added to satisfy
the sharpened intent (rank what carries cross-stock signal, recommend a concrete GNN config). It
reuses the reviewed `predictive._ridge_fit_predict` / `_diracc` and follows the same leakage-safe
pattern verified above: neighbour selection (PK-corr, market-adjusted residual-corr, directed
lead-lag, volume->PK) uses TRAIN-only matrices; every feature is dated <= t; the target is PK
variance at t+h; each candidate is fit on a shared valid mask so base (HAR) and base+addon use
identical rows (the M1 comparability fix applied here too); per-stock RMSE deltas drive a binomial
sign test. Full 63-test suite passes; diff-cover C0=100% on changed lines incl. this module.
