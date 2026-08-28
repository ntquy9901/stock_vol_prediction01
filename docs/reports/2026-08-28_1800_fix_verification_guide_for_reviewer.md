# Fix-verification guide (for the external AI reviewer) — 2026-08-28

**Repo (public):** https://github.com/ntquy9901/stock_vol_prediction01
**Reviews the fixes for:** `code_review_report_2026-08-28.md` (4 High + 8 Medium/Low).
**Commits under review:** `f49bca2` (H-01..H-04, M-01/02/03, L-01) and `88ce8fb` (M-04..M-08, L-02, L-03).
**Scope exclusion:** anything under `archive/` is retired, out of scope.

This guide maps every finding to the exact code change (`file:line`), the regression test that pins it, and how
to verify. Two findings (H-01, H-02) were determined **not to be bugs** — the evidence is given so you can check
that judgment rather than take it on trust.

## 0. How to run every guard test at once

```
# base python3 (numpy-only parts): metrics/baselines/eda
python -m pytest submission/soict_lstm_gat/tests/test_metrics.py \
  submission/soict_lstm_gat/tests/test_baselines.py \
  scripts/eda/test_test_diagnostics.py scripts/eda/test_estimator_forecast_ablation.py \
  scripts/garch_masked/test_floor_sensitivity.py scripts/garch_masked/test_garch_masked.py -q

# torch venv (panel builder + OOS resume): masked_rich + run_oos_suite
.venv_gpu_encode/Scripts/python.exe -m pytest \
  baselines/2026-08-21_har_anchored_residual/code/test_masked_rich.py \
  scripts/garch_masked/test_run_oos_suite.py -q
```

Last local run: submission 58, garch_masked 15, eda 19, masked_rich 15 — all pass. Pre-push gate green on both
commits (data-quality 301, delivered-baseline 69).

## 1. What to check first (the two "not a bug" verdicts)

### H-01 — is the paper reporting the ensemble metric or the per-seed mean?
**Claim in the review:** the `metrics` field is the metric of the seed-AVERAGED ensemble, so any downstream table
that reads `result.json["metrics"]["LSTM"]` reports the wrong number.
**Verdict: not a paper bug.** The papers/reports read `metrics_per_seed`, not `metrics`. Verify:
```
python - <<'PY'
import json; r=json.load(open("results/masked_rich_floor1e2/vn30_h1/result.json"))
print("ensemble   :", r["metrics"]["LSTM"]["qlike"])          # ~0.6432
print("per-seed   :", r["metrics_per_seed"]["LSTM"]["qlike"]) # ~0.7037
PY
grep -n "LSTM" docs/reports/2026-08-28_ALL_metrics_review.md | head -1   # prints "0.7037 (0.054)" = per-seed
```
The published number `0.7037 (0.054)` is the per-seed mean±std, not the ensemble `0.6432`.
**Hardening applied:** schema comment at `run_masked_rich.py:280`; guard test
`test_masked_rich.py::test_ensemble_metric_differs_from_perseed_mean_schema_contract` pins that the two quantities
are distinct (ensemble mse 1.0 vs per-seed-mean mse 2.0 for the same two seeds), so a future edit cannot conflate
them. **To review:** confirm every learned number in the papers traces to `metrics_per_seed`, and that the
guard test fails if you swap the two.

### H-02 — is the GARCH multi-step offset (`n_va` count) misaligned?
**Claim:** the offset should be a calendar-date distance, not the validation observation count.
**Verdict: not a bug — observation-space is the self-consistent choice.** The per-node GARCH series is the node's
masked train-target OBSERVATIONS (`compute_garch_masked.py:_garch_pred`), so the frozen forecast recursion steps in
observation units; the node's own `n_va` validation observations precede its test observations in that same series,
so `fc_full[n_va:]` aligns each test target. A calendar-day offset would instead mis-match the irregularly-spaced,
observation-indexed path. GARCH is a dominated benchmark; step-alignment does not change its ranking. Rationale is
documented in the `_garch_pred` docstring (`compute_garch_masked.py:78-95`); alignment test:
`test_garch_masked.py::test_garch_pred_skips_validation_interval`. **To review:** confirm the series is built from
node observations (not a calendar grid), so observation-count is the matching offset.

## 2. Fixed findings — file:line + guard test

| ID | Bug | Fix (`file:line`) | Guard test |
|----|-----|-------------------|------------|
| **H-03** | EDA GARCH `sse_share` divided by the HAR-X pooled SSE → GARCH shares didn't sum to 1 | `scripts/eda/test_diagnostics.py:107` (`tot_sse_g`), `:128` (per-model total) | `test_test_diagnostics.py::test_per_ticker_frame_columns_and_shares` (asserts both `harx_`/`garch_sse_share` sum to 1) |
| **H-04** | order-dependent rolling/ewm estimators computed on raw as-read; unsorted/dup dates corrupt windows | `scripts/eda/estimator_forecast_ablation.py:76` (sort+dedup by date before estimators) | `test_estimator_forecast_ablation.py::test_write_estimator_processed_is_date_sorted_and_deduped` |
| **M-01** | `_fallback_forecast` returned NaN on a non-finite series (docstring promised finite/positive) | `submission/soict_lstm_gat/baselines.py:78-95` (finite-only mean, floor if empty) | `test_baselines.py::test_garch_forecast_finite_on_nonfinite_series`, `::..._all_nonfinite_returns_floor` |
| **M-02** | `mse/mae/r2/qlike` broadcast mismatched shapes `(n,)` vs `(n,1)` silently | `submission/soict_lstm_gat/metrics.py:26` (`_check_pair`), applied `:43,:54,:64,:90` | `test_metrics.py::test_point_metrics_reject_shape_mismatch` |
| **M-03** | QLIKE accepted NaN inputs and floor ≤0 / non-finite | `metrics.py:88` (floor validation) + `_check_pair` finite check | `test_metrics.py::test_qlike_rejects_invalid_floor`, `::test_qlike_rejects_non_finite_input` |
| **L-01** | R² returned NaN/inf when target constant (`ss_tot==0`) | `metrics.py:67-69` (1.0 exact / 0.0 else) | `test_metrics.py::test_r2_constant_target_exact_match_is_one`, `::..._nonexact_is_zero` |
| **L-02** | DM only checked `h≥1`; HLN/HAC undefined for `h≥n` | `metrics.py:135` (`if h >= n: raise`) | `test_metrics.py::test_dm_rejects_horizon_ge_n` |
| **M-04** | present ohlcv covering <50% of a ticker's dates silently zero-fills volume shock | `masked_rich.py:96-97,110-112` (`_MIN_VOL_COVERAGE` + `warnings.warn`) | `test_masked_rich.py::test_volume_zscore_warns_on_low_coverage` |
| **M-05** | two edges use different overlap thresholds (corr 100 vs vol2pk 30), undocumented | `masked_rich.py:41-43` (single-source constants), `run_masked_rich.py:303` (`edge_config` in result.json) | `test_masked_rich.py::test_run_out_subdir_writes_separate_results_tree` (asserts `edge_config`) |
| **M-06** | OOS resume skipped on the mere presence of `metrics["GARCH"]` (stale/other-universe not recomputed) | `run_oos_suite.py:48` (`_universe_fp`), `:54` (`_has_garch` schema+horizon+fingerprint), `:112` (call), `:92` (write fp) | `scripts/garch_masked/test_run_oos_suite.py` (7 cases) |
| **M-07** | `screen_files` swallowed all exceptions via `except Exception: continue`, only tested `v==0.0` | `floor_sensitivity.py:123` (reasons + `max_nan_frac` + `report`), `:139-152` (per-reason) | `test_floor_sensitivity.py::test_screen_files_reports_reasons_and_nan_and_missing_column` |
| **M-08** | GARCH mean-fallback written as an ordinary `GARCH` metric with no degradation record | `baselines.py:137,153` (`return_status`), `compute_garch_masked.py:122` (`garch_meta`), `:179` + `run_oos_suite.py:90-91` (write) | `test_baselines.py::test_garch_forecast_return_status_flags_fallback`, `test_garch_masked.py::test_garch_pred_collects_status_and_garch_meta_aggregates` |
| **L-03** | GARCH provenance not persisted | `compute_garch_masked.py:34` (`_ARCH_VERSION`), `garch_meta` writes seed/horizon/floor/arch-version | same `garch_meta` test |

## 3. Correctness properties to challenge (adversarial angles)

Please try to break these specifically:

1. **No result drift (round 2).** Every M-04..L-03 change was intended to be a **no-op on the delivered clean
   data** (metadata / warnings / validation only). Verify no committed `results/masked_rich_floor1e2/**/result.json`
   value changed in `88ce8fb`/`f49bca2` (they add `edge_config`/`garch_meta` keys only when a run is re-executed;
   no committed metric was edited). Command: `git show 88ce8fb --name-only --pretty=format: | grep result.json`
   (expect empty — verified: 0 result.json files in either commit; both touch only code/tests/reports).
2. **M-01 fallback.** Feed `garch_forecast` a series of all `inf` / all `nan` / length-1 — must return finite,
   `>= floor`, shape `(n_test,)`; with `return_status=True` must report `fallback=True` + a reason.
3. **M-02/M-03 validators.** Confirm the new `_check_pair` does not reject legitimate inputs the pipeline actually
   uses (floored finite predictions, equal shapes) — i.e. the delivered runs still pass (they do: 58 tests).
   Then confirm it rejects `(n,)` vs `(n,1)`, NaN, and floor ≤ 0.
4. **M-06 fingerprint.** Change one ticker in the screened set and confirm `_has_garch(...)` returns `False`
   (forces recompute); confirm an order-permuted file list yields the same fingerprint (no false recompute).
5. **M-07 universe invariance.** Confirm the kept set is identical to the previous `screen_files` on NaN-free data
   (the new NaN-fraction gate only excludes NaN-heavy files, absent in clean processed data).
6. **H-03 partition.** For a synthetic multi-ticker panel, assert `sum(garch_sse_share) == 1` (previously ≠ 1).

## 4. Unchanged invariants (still hold, do not re-flag)
- Train-only scalers + both graph edges (leakage): `test_masked_rich.py::test_train_only_invariance_no_leakage`.
- GARCH persistence cap 0.999 (IGARCH divergence) + shared QLIKE floor: `test_baselines.py` cap tests.
- Per-seed mean reporting + DM on the 5-seed ensemble forecast (H-01 above).
- VN raw not split-adjusted (overnight winsorized); S&P 500 already adjusted.

## 5. Full context
- Triage + rationale for every finding: `docs/reports/2026-08-28_1600_external_code_review_triage_report.md`.
- Original review-scope guide: `docs/reports/2026-08-28_code_review_guide.md`.
- Consolidated result numbers: `docs/reports/2026-08-28_ALL_metrics_review.md`.
