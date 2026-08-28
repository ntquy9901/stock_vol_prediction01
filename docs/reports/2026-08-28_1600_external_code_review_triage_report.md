# External code-review triage — response to `code_review_report_2026-08-28.md`

**Date:** 2026-08-28
**Source:** `C:\tools\codex_ml_senior_architect_agent\code_review_report_2026-08-28.md` (4-agent read-only review, 4 High + 8 Medium/Low)
**Method:** each finding re-verified against the live code before acceptance; fixes are TDD (failing test first).

## Disposition summary

| ID | Severity | Verified verdict | Action | Result impact |
|----|----------|------------------|--------|---------------|
| H-01 | High | Partially valid (schema clarity, NOT a paper bug) | Guard test + schema comment | None — paper already uses `metrics_per_seed` |
| H-02 | High | Not a numeric bug (observation-space alignment is self-consistent) | Docstring clarification | None |
| H-03 | High | **Valid bug** (EDA GARCH share used HAR-X denominator) | Fixed + regression test | EDA tool only, not paper |
| H-04 | High | Low risk (value already paired with own raw-row date) | Defensive sort/dedup guard + test | None on clean ETL data |
| M-01 | Medium | **Valid** (fallback returned NaN on non-finite series) | Fixed + 2 tests | None (path unreached on real data) |
| M-02 | Medium | **Valid** (no shape guard in point metrics) | Shared `_check_pair` + test | None |
| M-03 | Medium | **Valid** (QLIKE accepted bad floor / NaN) | Floor+finite validation + tests | None |
| L-01 | Low | **Valid** (R² NaN/inf on constant target) | Constant-target policy + tests | None |
| M-04 | Medium | Valid concern | Documented follow-up | None (ETL data weekday-complete) |
| M-05 | Medium | Valid concern | Documented follow-up | None |
| M-06 | Medium | Valid concern | Documented follow-up | None |
| M-07 | Medium | Valid concern | Documented follow-up | None |
| M-08 | Medium | Valid concern | Documented follow-up | None |
| L-02 | Low | Valid concern | Documented follow-up | None |
| L-03 | Low | Valid concern | Documented follow-up | None |

## High findings — detail

### H-01 — learned `metrics` field is the ensemble, not the per-seed mean
**Verified NOT a paper bug.** The published numbers come from `metrics_per_seed`, not `metrics`. Confirmed on
`results/masked_rich_floor1e2/vn30_h1/result.json`: stored ensemble QLIKE(LSTM)=`0.64324`, per-seed
mean=`0.70366`; the report `docs/reports/2026-08-28_ALL_metrics_review.md` prints **`0.7037 (0.054)`** — i.e. the
per-seed mean±std. Every learned number in the papers/reports is the per-seed mean.
**Action:** the `metrics` field (learned = seed-ensemble metric, used only for the DM forecast) and
`metrics_per_seed` (paper-reported mean of seed-level metrics) are two distinct quantities. Added a schema
comment in `run_masked_rich.py` and a guard test `test_ensemble_metric_differs_from_perseed_mean_schema_contract`
that pins ensemble-mse≠per-seed-mean-mse so a future edit cannot silently conflate them. No re-run needed.

### H-02 — GARCH OOS offset from `n_va` count, not target dates
**Verified defensible, not a numeric bug.** The per-node GARCH series is the node's masked **train-target
observations**, so the frozen forecast recursion steps in *observation* units, not calendar days. The node's own
validation observations are exactly the `n_va` steps preceding its test observations in that same series, so
`fc_full[n_va:]` is the observation-consistent alignment. A calendar-day offset (the reviewer's suggestion) would
instead mis-match the irregularly-spaced, observation-indexed path. Applying GARCH to a masked panel is inherently
an approximation; GARCH is a dominated benchmark and step-alignment does not change its ranking.
**Action:** expanded the `_garch_pred` docstring to record this rationale. No code change, no re-run.

### H-03 — EDA GARCH ticker shares used the HAR-X pooled denominator (**valid bug**)
`per_ticker_frame` computed `garch_sse_share = garch_sse_j / tot_sse_HARX`, so GARCH shares did not partition
(sum to 1). **Fixed:** compute `tot_sse_g` from the GARCH residuals and divide each model's share by its own
pooled total. Regression test asserts both `harx_sse_share` and `garch_sse_share` sum to 1. Scope: EDA diagnostic
HTML only — no paper number depends on it.

### H-04 — estimator ablation merged by row position (low risk)
The estimator value was already paired with its **own raw-row date** (same DataFrame row), so the (date, value)
pair was correct; `build_masked_rich` re-aligns by date downstream. The real exposure was that the order-dependent
rolling/ewm/diff estimators are computed on the raw file *as read* — an unsorted/duplicated raw file would corrupt
the windows. **Fixed defensively:** sort by date + drop duplicate dates *before* computing estimators (a no-op on
the clean ETL data). Test injects shuffled + duplicate dates and asserts the written CSV is sorted, unique, and its
per-date values match the estimators recomputed on the cleaned raw.

## Medium/Low fixed (input-validation hardening, delivered lib)

- **M-01** `_fallback_forecast` now uses only the finite entries of the series (this path is reached precisely
  when the series is degenerate/non-finite) and returns `floor` if nothing finite remains — restoring the
  docstring's finite/positive/≥floor guarantee. Tests: non-finite series → finite output; all-non-finite → floor.
- **M-02 / M-03** shared `_check_pair(y, p)` in `metrics.py` requires equal shape + finite values across
  `mse/mae/r2/qlike`; `per_obs_qlike` additionally rejects a non-finite/non-positive floor. Tests cover
  shape-mismatch, NaN input, and invalid floor.
- **L-01** `r2` constant-target policy: `ss_tot == 0` → `1.0` for an exact prediction else `0.0` (no NaN/inf).
  Tests cover both branches.

## Medium/Low documented as follow-ups (no result impact)

- **M-04** partial per-date volume coverage is zero-filled (guard only counts fully-missing files). The delivered
  panels come from weekday-complete ETL data, so partial coverage is minimal; a per-date coverage audit that
  reports (not silently zero-fills) is a hardening follow-up. Aligns with the CLAUDE.md no-silent-degradation rule.
- **M-05** directed vol→PK edge uses a fixed `_MIN_PAIRS=30` while the correlation edge uses `edge_min_overlap`;
  unify/config + record in metadata.
- **M-06** GARCH resume `_has_garch()` only checks the key exists; add a completion/version fingerprint of
  universe+config+horizon.
- **M-07** `floor_sensitivity.screen_files` swallows malformed files via `except Exception: continue` and only
  tests `v==0.0`; add per-reason exclusion logging + NaN-fraction audit.
- **M-08** GARCH fallback (arch missing / fit fail) is written as an ordinary `GARCH` metric with no
  fallback-count/reason in the artifact; add degradation metadata + aggregate fallback rate.
- **L-02** DM only checks `h≥1`; document/enforce `1≤h<n`.
- **L-03** persist GARCH seed/config/dependency-version provenance in the output artifact.

## Verification

- `submission/soict_lstm_gat`: `pytest tests/` → **56 passed** (`.venv_gpu_encode`).
- EDA: `pytest scripts/eda/test_test_diagnostics.py scripts/eda/test_estimator_forecast_ablation.py` → **9 passed**.
- `test_masked_rich.py` H-01/per-seed/out_subdir subset → **3 passed** (`.venv_gpu_encode`).
- ruff on changed files: E702/E501 are the pre-existing semicolon/line-length house style (matched per surgical
  rule); the `pytest` F401 in `test_baselines.py` pre-dates this change (0 uses in git HEAD). ruff is warn-only in
  the pre-push gate.

## Data-quality gate

`N/A (no data change)` — this change touches evaluation/metrics/EDA code and tests only; no `data/`,
feature, manifest, or training-data change.

## Code review

This report IS the triage of an external adversarial review. The valid findings were fixed with regression tests
first; the non-valid ones are documented above with the verification that refutes or de-scopes them.
