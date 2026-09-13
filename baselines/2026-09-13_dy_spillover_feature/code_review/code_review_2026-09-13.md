# Adversarial code review — DY spillover feature (2026-09-13)

Reviewer focus (per CLAUDE.md §5 / task): look-ahead & leakage in the VAR/GFEVD window, the sector-mean
aggregation, the causal ffill, and test rigor. Scope: `code/dy_spillover.py`, `code/run_dy.py`,
`code/config.py`, `test/test_dy_spillover.py`, `test/test_run_dy.py`.

## Verdict
**No HIGH findings. No exploitable look-ahead leakage.** Every path by which a feature at row-date `t`
could depend on volatility dated `> t` was traced and is clean:
- Trailing window `panel.iloc[max(0,pos-window+1):pos+1]` on a `.sort_index()` panel = dates `<= t`
  (contemporaneous `t` allowed by design).
- Anchor→all-dates `.reindex(dates).ffill()` is strictly forward (a date between anchors gets the *earlier*
  anchor; no backward propagation).
- GFEVD/VAR fit only on the in-window matrix; `sector_logvar_panel` is per-`(date,sector)` mean then log
  (purely contemporaneous); `merge_spillover` maps same-date values (no shift).
- The global precompute before the walk-forward is safe *because* the feature is causal.
- GFEVD math verified term-by-term vs Pesaran–Shin 1998 / Diebold–Yilmaz 2012 (numerator, `σ_jj`/`denom`
  normalisation, row-normalise, `S=100*(1-trace/K)`, `NET=TO−FROM` sign). Correct.

## Findings and dispositions

| # | Sev | Finding | Disposition |
|---|-----|---------|-------------|
| 1 | MEDIUM | Overfit verdict compares **last-fold in-sample** `train_qlike` vs **all-fold pooled** `test_qlike` (`run_dy.py::run_dy`, `_fold_qlike`) — non-comparable populations. | **Accepted as-is, documented.** This is byte-for-byte the canonical evidence format of the sibling `verify_index_vol_feature.py` / `run_gbm.py`; diverging would make the diagnostics inconsistent across the complex-network suite. Robust here regardless: train QLIKE (1.86–2.00) exceeds test QLIKE (1.57–1.91) by a wide margin, so the `ok` verdict is not marginal. |
| 2 | MEDIUM | `test_rolling_spillover_is_causal` could pass **vacuously** if the pre-cut region were all-NaN (the non-emptiness guard was on the full series). | **FIXED.** Guard tightened to `assert total[m].notna().any()` on the masked pre-cut region, so the causal equality check cannot be satisfied trivially. |
| 3 | LOW | Overfit-evidence pre-push gate may not treat `"GBM"`/`"GBM+spillover"` as "learned" (`overfit_check.looks_learned`), so the result JSON may skip enforcement. | **Documented, no change.** Identical to the sibling complex-network GBM outputs; the file is skipped by `_is_masked_rich_result` (no top-level `metrics`) and still carries full per-horizon `train/test/fit_diagnostics` evidence for human review. |
| 4 | LOW | Sector qualification (`sec_count`) uses full-sample constituent counts → survivorship look-ahead in *which* sectors qualify. | **Documented caveat, no leakage.** The per-day sector mean averages only stocks present that day (a not-yet-listed stock contributes no rows), so no future volatility enters any feature value. Which sectors form the VAR system is static metadata, not a forecast input. |
| 5 | LOW | `dropna(how="any")` yields non-contiguous VAR rows on thin panels. | **Comment added** in `spillover_from_window` noting it is a modeling approximation, not a leakage path. `gfevd` 0/0 NaN (uncaught by the VAR `except`) is benign — counted as a failed window via `np.isnan(s)` and ffilled. |
| — | INFO | Silent-degradation handling is correct: NaN for failed windows / non-VAR sectors is GBM-native missing data; the whole-panel-degenerate case fails loud (`run_dy` raises, covered by `test_run_dy_degenerate_feature_raises`). GFEVD sign/identity tests are non-vacuous. | No action. |

## Post-fix verification
`python -m pytest baselines/2026-09-13_dy_spillover_feature/test` → 14 passed; coverage C0 line = 100%,
C1 branch = 100% on `code/`; `ruff check --select F` clean; config-hardcode 0 BLOCK.
