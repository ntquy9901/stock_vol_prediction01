# Fix-verification guide v2 (for the external AI reviewer) — 2026-08-29

**Repo (public):** https://github.com/ntquy9901/stock_vol_prediction01 — review the files IN the repo.
**Supersedes:** `2026-08-28_1800_fix_verification_guide_for_reviewer.md` (adds the residual round R-01..R-14, the
internal multi-agent review's 2 fixes, the coverage gate, and the lessons-regression suite).
**Head at write time:** `ce66f70`. **Scope exclusion:** `archive/` is retired and OUT of scope.

This maps every fix to `file:line` + its guard test + how to verify. Three findings across the rounds were judged
**not bugs**; the evidence is given so you can check that judgment rather than trust it.

## 0. Run everything

```
# numpy-only parts (base python): metrics, baselines lib, eda, garch, lessons
python -m pytest submission/soict_lstm_gat/tests/test_metrics.py \
  submission/soict_lstm_gat/tests/test_baselines.py scripts/eda/test_*.py \
  scripts/garch_masked/test_*.py tests/test_lessons_regression.py -q
# torch venv: panel builder + resume guard
.venv_gpu_encode/Scripts/python.exe -m pytest \
  baselines/2026-08-21_har_anchored_residual/code/test_masked_rich.py \
  scripts/garch_masked/test_run_oos_suite.py -q
```

Last local run: submission 60, eda 23, garch 15+, masked_rich 17, lessons 7 — all pass. The pre-push gate now
also enforces **diff-coverage C0 line=100% + C1 branch=95%** on changed lines (see §4).

## 1. Review-round history (what changed and why)

| Round | Commit | Scope |
|-------|--------|-------|
| Triage | `f49bca2` | H-01..H-04, M-01/02/03, L-01 |
| Hardening | `88ce8fb` | M-04..M-08, L-02, L-03 |
| Residual | `71f3c61` | R-01..R-14 |
| Multi-agent + gate | `ce66f70` | R-09 tolerance fix, M-04 upper-bound fix, coverage gate, lessons suite |

Full triage + rationale for every finding: `docs/reports/2026-08-28_1600_external_code_review_triage_report.md`.

## 2. The three "not a bug" verdicts — verify these first

**H-01 (metrics ensemble vs per-seed).** The papers read `metrics_per_seed`, not the ensemble `metrics`.
`python -c "import json;r=json.load(open('results/masked_rich_floor1e2/vn30_h1/result.json'));print(r['metrics']['LSTM']['qlike'], r['metrics_per_seed']['LSTM']['qlike'])"` → ensemble ~0.643 vs per-seed 0.7037; the report prints 0.7037.
Guard: `test_masked_rich.py::test_ensemble_metric_differs_from_perseed_mean_schema_contract`.
**Open item to check:** no *code* path consumes `metrics_per_seed` (`build_report.py` reads the ensemble `metrics`).
The consolidated `docs/reports/2026-08-28_ALL_metrics_review.md` was built from per-seed; please confirm the
generator that feeds the FINAL paper tables also uses per-seed (this is the one unresolved consistency risk).

**H-02 / R-01 (GARCH `n_va` offset).** Observation-space, not calendar — the GARCH series is the node's masked
train-target OBSERVATIONS, so `fc_full[n_va:]` aligns each test obs by observation count. Rationale in
`compute_garch_masked.py:_garch_pred` docstring; proof test `test_garch_masked.py::test_garch_alignment_with_missing_dates_and_purge` (sparse masks, parametrized h1/5/10/22). GARCH is a dominated benchmark.

**DM per-obs HAC (LOW, pre-existing).** `submission/soict_lstm_gat/metrics.py::diebold_mariano`'s Newey-West lag
crosses ticker seams on pooled data for h>1. The PAPER uses `stats.date_clustered_dm` (aggregates per date first),
not this per-obs DM, so paper p-values are unaffected. Noted for the submission's `evaluate.py` only.

## 3. Fixed findings — file:line + guard test

| ID | Fix (`file:line`) | Guard test |
|----|-------------------|------------|
| H-03 | `scripts/eda/test_diagnostics.py:107,128` (per-model `tot_sse_g`) | `test_test_diagnostics.py::test_per_ticker_frame_columns_and_shares` |
| H-04/R-07 | `scripts/eda/estimator_forecast_ablation.py:76` (stable sort+dedup by date) | `test_estimator_forecast_ablation.py::test_write_estimator_processed_is_date_sorted_and_deduped` |
| M-01/R-02 | `submission/soict_lstm_gat/baselines.py:78-96` (finite-only + overflow→floor) | `test_baselines.py::test_garch_forecast_fallback_overflow_safe`, `..._finite_on_nonfinite_series` |
| M-02/M-03/R-05/R-14 | `metrics.py:26-40` (`_check_pair`: shape+finite+non-empty), `:88` (qlike floor) | `test_metrics.py::test_point_metrics_reject_shape_mismatch`, `..._reject_empty_input`, `..._qlike_rejects_invalid_floor` |
| L-01 | `metrics.py:67-69` (R² constant-target policy) | `test_metrics.py::test_r2_constant_target_*` |
| L-02/R-13 | `metrics.py:131-140` (DM 1≤h<n, integer h) | `test_metrics.py::test_dm_rejects_horizon_ge_n`, `..._non_integer_horizon`, `..._below_one` |
| M-06 | `run_oos_suite.py:48,54` (`_universe_fp` + schema/horizon/fingerprint) | `scripts/garch_masked/test_run_oos_suite.py` (8 cases) |
| M-07/R-06 | `floor_sensitivity.py:123` (per-reason exclusions, NaN + non-numeric) | `test_floor_sensitivity.py::test_screen_files_*` |
| M-08/L-03/R-03 | `baselines.py:137` (`return_status`), `compute_garch_masked.py:123` (`garch_meta`, `seed` forwarded) | `test_baselines.py::test_garch_forecast_return_status_flags_fallback`, `test_garch_masked.py::test_garch_pred_collects_status_and_garch_meta_aggregates` |
| M-04 (+ upper bound) | `masked_rich.py:96-99,122-126` (`_EMPTY_VOL_COVERAGE`: present-but-empty ≡ missing → fail-loud cap) | `test_masked_rich.py::test_volume_zscore_warns_on_low_coverage`, `..._fails_loud_on_present_but_empty_files` |
| M-05/R-10 | `masked_rich.py:41-44` (edge constants), `run_masked_rich.py:303` (`edge_config`) | `test_masked_rich.py::test_run_out_subdir_writes_separate_results_tree` |
| **R-09** | `volatility_estimators.py:31,64-67` (OHLC geometry with `OHLC_RTOL=1e-5`) | `test_volatility_estimators.py::test_ohlc_geometry_violations_are_invalid`, `..._float32_noise_within_tolerance_stays_valid` |
| R-08 | `volatility_estimators.py:110-111` (`panel_summary` sort+dedup) | `test_volatility_estimators.py::test_panel_summary_robust_to_unsorted_duplicate_dates` |
| R-11 | `test_masked_rich.py` (leakage test also asserts `X_tr`/`t_std` invariance) | `test_train_only_invariance_no_leakage` |
| R-12 | `run_yz_robustness.py:_done` (schema-validates both learned models' finite QLIKE) | `test_run_yz_robustness.py::test_done_requires_metrics_per_seed` |

**Two fixes from the internal multi-agent review (the ones that changed behaviour):**
- **R-09** was originally an EXACT compare that dropped 22,348 gate-clean S&P 500 rows (float32 storage noise);
  now uses the raw-quality-gate tolerance `OHLC_RTOL=1e-5` → no-op on clean data, delivered Parkinson numbers
  reproduce. **Please re-verify** on `data/raw/prices/sp500` that base-valid≈ok-count (tolerance restored).
- **M-04 upper bound**: a present-but-empty ohlcv file (dates don't intersect the panel) previously only warned;
  it now counts toward the `>2` fail-loud cap (semantically missing), closing a no-silent-degradation hole.

## 4. Coverage gate (now enforced) — audit it

`scripts/git_hooks/pre-push` step 2 hard-blocks the push on **C0 line `diff-cover --fail-under=100`** and
**C1 branch `--branch-coverage --fail-under=95`** over the changed lines (`--include` = committed
`PUSH_BASE..HEAD` files, so uncommitted WIP doesn't pollute). Changed-scope tests are discovered as the sibling
`test_<module>.py` + a `tests/`/`test/` subdir and run under `.venv_gpu_encode` with `--cov-branch`. Entry-driver
`main()` functions carry `# pragma: no cover` (exercised by real OOS runs, not unit tests). Knobs
`QG_MIN_COVER`/`QG_MIN_BRANCH`. **Check:** the discovery is sibling-only (not a broad `test_*.py` glob, which had
false-blocked on a pre-existing broken `test_draw_masked_panel.py`); `--include` scoping is correct.

## 5. Lessons-regression suite (new) — audit it

`tests/test_lessons_regression.py` (7 tests, runs every push at gate step 4) codifies documented lessons as
invariants against the REAL functions: DirAcc = sign of CHANGES; temporal split chronological + NaT/duplicate
fail-loud; Parkinson target is VARIANCE (σ²) not σ; QLIKE shared positivity floor; date-clustered DM does not
overstate vs naive per-obs; VolatilityNormalizer transform-applied + invertible round-trip. **Check:** each test
imports and exercises the shipped function (no re-implementation), and the invariant genuinely fails if the past
bug is reintroduced.

## 6. Adversarial angles to try
1. **No result drift:** confirm `git show ce66f70 --name-only | grep result.json` is empty (only the R-09/M-04
   code + tests + gate changed; no committed metric edited).
2. **R-09 tolerance:** find a real S&P 500 bar with `high` a few ×1e-7 below `max(open,close)` and confirm it is
   `ok=True` (kept) now, `parkinson` finite.
3. **M-04 cap:** construct >2 present-but-empty ohlcv files and confirm `_volume_zscore_wide` raises.
4. **Coverage gate:** stage a source edit with no test and confirm the gate would block on C0<100.
5. **Leakage (unchanged invariant):** `test_train_only_invariance_no_leakage` — perturb test region, assert train
   edges/scalers/`X_tr` unchanged.

## 7. Known caveats (do not re-flag)
- Learned metrics are 5-seed means (`metrics_per_seed`); DM is on the 5-seed ensemble (both stated).
- GARCH is a dominated benchmark; pseudo-returns random-signed; observation-space offset is intentional.
- VN raw OHLCV not split-adjusted (overnight winsorized); S&P 500 already adjusted.
- The GAT graph effect is small / within per-seed dispersion — stated honestly.
