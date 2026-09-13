# Code Review — Principled HAR-family baseline (2026-09-13)

Adversarial review of `code/{estimators,build_panel,run_har,config}.py`, compared against the trusted
references `full_matrix.py` (FM), `vn_gbm_graph_stage1.py` (S1), `metrics.py`, `stats.py`, and the siblings
`2026-09-12_complex_network/code/{run_gbm.py,verify_reduced_features.py}`.

## Verdict: no CRITICAL / no MAJOR. Accept.

The baseline is causally sound and leakage-free; the walk-forward / pooled-QLIKE / date-clustered-DM protocol
matches the trusted sibling exactly (embargo `int(h*1.6)+5`, `tr<ts-embargo`, `te∈[ts,tend)`, gate
`len(te)==0 or len(tr)<min_rows`). DM direction correct (`gain_vs_own_pct>0 ⇒ principled lower QLIKE`), QLIKE
floor identical for both models (`FL`), fit-diagnostics logic identical to siblings. GK/RS/YZ/HAR columns are
retained by `FM.load`→`FM.panel` (dropna on `OWN+["y"]` only) and NaN warmup rows are handled natively by
HistGradientBoosting. Principled and own models train/score on identical rows (shared panel, feature-independent
fold gate) → fair DM. Leave-one-out correct; the `if r is None` pragma is justified (fold gate is
feature-independent, so a subset cannot fail when the full set scored).

## Findings

- **MINOR-1 (FIXED):** `semivariance` mapped a leading NaN daily-return (first bar per ticker) to `0.0` via
  `.where(r<0, 0.0)`, counting it as a real zero inside the `min_periods` window (understated the earliest
  1-3 semivariance values per ticker; benign, no leakage). **Fix applied:** `.where(r.notna())` so a NaN
  return is excluded from the rolling mean. Regression test added
  (`test_semivariance_nan_return_excluded_not_counted_as_zero`).
- **MINOR-2 (acknowledged, not changed):** the embargo constants `1.6`/`+5` (`run_har.py`) and the overfit
  multiplier `1.25` are inline. These are a **verbatim replication** of the canonical protocol in
  `run_gbm.py` / `verify_reduced_features.py` / `S1`; centralising them would require a cross-sibling
  `protocol_config` (out of scope). Left inline for exact protocol parity — deliberate, not a new violation.

## Tests / coverage
`test/`: 3 estimator tests (formula-exact split, min_periods NaN, causality, NaN-exclusion), 3 build_panel
tests (features present, semivariance causal, mapping), 2 run_har tests (smoke structure incl leave-one-out
per feature + fit diagnostics, empty-when-no-fold). C0 line = 100%, C1 branch = 100% on changed lines
(one defensive `# pragma: no cover` on the unreachable LOO guard). GK/RS/YZ formula-exactness is inherited
from the already-tested enriched pipeline (`2026-08-31_enriched_processed`), used from columns not recomputed.
