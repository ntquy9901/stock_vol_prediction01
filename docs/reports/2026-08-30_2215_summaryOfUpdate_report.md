# Summary of update — dirty-data audit + ETL-cleaning toolkit

Date: 2026-08-30. Scope: new audit toolkit + generated deliverables. CPU/pandas only (GPU committed to a
training job).

## What changed
New module `scripts/etl_audit/` (SDD spec + TDD code + tests):
- `requirements.md`, `design.md` — SDD spec written before code.
- `dirty_data_detectors.py` — pure per-(ticker,date) LOCATE detectors for the 9 dirty-data classes.
- `etl_cleaning.py` — NEW pure ETL cleaning functions (widen_range, clip_oc, swap_or_drop_high_low,
  reconstruct_nonpositive, backadjust_splits, cut_to_listing, drop_naninf, flag_zero_range, flag_zero_volume).
- `build_dirty_data_report.py` — driver: per-market HTML (executive summary + sortable per-ticker table +
  per-stock drill-down charts) + consolidated ETL spec markdown + raw-vs-processed clip measurement.
- `test_dirty_data_detectors.py`, `test_etl_cleaning.py`, `test_build_dirty_data_report.py` — unique
  basenames (no pytest collision).

Generated deliverables (`docs/reports/`):
- `2026-08-31_{hnx,hose,vn30,vn100,sp500}_dirty_data_etl.html` — self-contained (base64 charts, 0 external
  CDN refs), sortable per-ticker table, per-stock drill-down with dirty dates highlighted.
- `2026-08-31_etl_cleaning_spec.md` — per-class detection + cleaning + estimator impact + priority +
  cross-market prevalence + raw-vs-processed table.

## Headline findings (all trace to data)
- O/C-outside: 189/299 HNX tickers affected (matches the prior audit's 189/299 exactly); 3,441 HNX rows.
- Dominant TARGET-affecting issue is zero-range (H==L): HNX 486,273 raw rows (~45% of ticker-days), passing
  through to 482,724 zero processed Parkinson targets (the QLIKE floor driver).
- Raw-vs-processed: processed Parkinson IS upper-clipped at 0.1 (measured, not assumed) but affects very few
  rows — HNX 24, HOSE 1, vn30 1, sp500 172, vn100 0 (max 0.09457, never hit cap). Refines the prior
  "HOSE/HNX clipped, vn30/vn100 not" claim: the clip exists on 4/5 markets but is a rare-row effect; the
  real target impact is the zero-range pass-through.
- REAL (Parkinson-affecting): high<low, nonpositive, zero-range, split-jumps, NaN/inf. Cosmetic for
  Parkinson (only O/C estimators or leading/liquidity): open/close-outside, stale runs, leading backfill,
  zero-volume.

## Tests + coverage
- 54 new tests pass under the GPU venv (`.venv_gpu_encode`). C0 line = 100%, C1 branch = 100% on the three
  new source files (`--cov=scripts/etl_audit --cov-branch`).
- TDD: detector + cleaning tests written first, confirmed red (ModuleNotFoundError), then implemented to green.
- Each ETL rule has a formula/behaviour test (widen -> OHLC internally consistent; reconstruct -> positive
  OHLC; backadjust -> jump removed; swap -> valid geometry; cut-to-listing -> leading rows dropped).
- Real-data-sample smoke per market (skips cleanly if a market's data is absent; pragma no cover on the skip).

## Regression (no new breakage)
- Existing EDA suite `scripts/eda/` : 140 passed, 0 new failures (my code only reads those detectors,
  modifies nothing there).
- Cross-check test asserts my locate-detector counts equal `vnmarkets_eda.detect_ohlc_violations`.

## Extra deliverable (design-only)
- `docs/reports/2026-08-31_processed_schema_spec.md` — enriched processed-file schema DESIGN SPEC (no
  implementation this pass): why (compute-once/verifiable/mining), causal columns to add (per-day estimators
  + windowed YZ n=20 + HAR/market_pk/volume-z intermediates + audit flags), the CRITICAL leakage rule (no
  scalers/global-z/adjacency/future-window baked in — train-only per split), the volume-zscore window
  inconsistency (delivered `_20` on log1p(volume) vs monthly convention 22 -> recommend canonical
  `volume_zscore_22`, keep `_20` for back-compat, do NOT change delivered feature now), and a Pandera +
  cross-estimator + Evidently + schema-version validation plan. Grounded: `_VOL_WIN=20`
  (masked_rich.py:34), `MONTHLY_WIN=22` (data_utils.py:23).

## Code review — 3-lens adversarial (Blind Hunter + Edge Case Hunter + Acceptance Auditor)
Ran 3 parallel review agents. Triage + actions (all critical/major fixed):
- MAJOR (Blind + Acceptance, convergent): `split_jumps` was misclassified as REAL Parkinson-target-affecting.
  Parkinson ln(H/L)^2 (and GK/RS within-day ratios) are SCALE-INVARIANT, so an unadjusted split does not move
  the Parkinson target; only YZ's overnight term is affected on the boundary day. FIXED: reclassified as
  cosmetic-for-Parkinson in `ETL_RULE`, spec prose, requirements.md; estimator label -> "YZ overnight-boundary
  only". Deliverables regenerated. Test added.
- MAJOR (Edge Case): missing-volume column read as zero-volume -> `leading_backfill`/`cut_to_listing`
  over-flagged/over-cut (silent degradation). FIXED: absent volume -> NaN (neutral), non-trading rests on
  zero-range only. Tests added (detector + cleaner). Note: all 5 delivered markets HAVE volume, so
  deliverables were not affected; fix is for correctness/robustness.
- MINOR: stale_runs counted RUNS not DAYS (unit mixing in % rows) -> FIXED to stale DAYS + unit note in spec.
- MINOR: O/C-outside drill-down examples not sorted by magnitude -> FIXED (sort by violation desc).
- MINOR: `reconstruct_nonpositive` emitted All-NaN RuntimeWarning on all-nonpositive bars -> FIXED (suppress).
- MINOR: `clip_evidence` KeyError if a frame lacks `date` -> FIXED (guard returns has_processed False). Test.
- Noted/accepted: detectors assume date-sorted/unique input (the driver sorts/dedups via `_load_raw`, matching
  the sibling `vnmarkets_eda` precondition); drill-down highlight count capped at 25 (illustrative; true count
  in the sortable table); strict `>50%` threshold misses an exact 2:1 halving (matches documented spec).

## Performance
- Read-only per-file streaming, one market at a time (RAM bounded; per-market frames released). Vectorised
  numpy per frame. No model / no GPU. `--limit` for smoke. No batch=1 hot loop applies (no training).

## DoD checklist
- [x] SDD spec before code  [x] TDD red->green  [x] C0=100/C1=100 changed lines (60 tests)  [x] ruff -F clean
- [x] deliverables generated from real data, self-contained  [x] regression 0 new failures (140 EDA tests)
- [x] 3-lens code review, critical/major fixed  [x] enriched-schema design spec added  [x] pushed via gate
