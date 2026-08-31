# Baseline A3 — design / plan

Date: 2026-08-31. CLAUDE.md §5 Plan. Passes the 3 gates below.

## Data flow
```
raw OHLCV csv (per ticker)
  -> parse date, sort, drop_duplicates(date, keep=last)                     [order-safe]
  -> clean_ohlcv()  (ETL, priority order, reuses scripts/etl_audit)         [per-row labels + rejections]
       1 drop_naninf            (drop) reason=naninf
       2 reconstruct_nonpositive(fix)  cleaning=reconstruct_nonpositive; (drop) reason=nonpositive_unrecoverable
       3 swap_or_drop_high_low  (fix)  cleaning=swap_high_low;            (drop) reason=high_lt_low_unrecoverable
       4 cut_to_listing         (drop) reason=leading_backfill
       5 widen_range            (fix)  cleaning=widen_range   (open/close-outside; GK/RS/YZ only)
       6 backadjust_splits      (fix)  cleaning=backadjust_split (level rescale; Parkinson scale-invariant)
       7 flag_zero_range        (flag) zero_range_flag
       8 flag_zero_volume       (flag) zero_volume_flag
  -> estimators_from_ohlcv()  (reused; parkinson/garman_klass/rogers_satchell/yang_zhang[n=20])
  -> per-row causal columns (log_range, daily_return, har_daily/weekly/monthly, volume_zscore_20/22)
  -> [market pass] market_pk = cross-sectional MEAN of parkinson_variance over valid tickers per date
  -> write data/processed_enriched/<market>/<ticker>.csv + <ticker>_rejections.csv + _schema_version.json
```

## Key design decisions
- **NEW output location** `data/processed_enriched/` — leaves `data/processed/*_processed.csv` byte-identical,
  so the delivered-baseline tests (which the pre-push gate step-5 runs) cannot break. VN30 + VN100 enriched
  are committed (tracked markets); hose/hnx/sp500 enriched are local-only (gitignored, matching the existing
  sp500 data policy).
- **Reuse, don't re-derive.** Estimators come from `scripts/eda/volatility_estimators.estimators_from_ohlcv`
  (tested vs published formula incl. `test_windowed_yang_zhang_matches_reference_formula`). ETL cleaners come
  from `scripts/etl_audit/etl_cleaning.py` (each already unit-tested). Column mapping:
  parkinson→`parkinson_variance`, garman_klass→`garman_klass_variance`, rogers_satchell→
  `rogers_satchell_variance`, yang_zhang→`yang_zhang_n20`.
- **Windows from config, never hardcoded** — `HAR_WEEKLY_WINDOW=5`, `HAR_MONTHLY_WINDOW=22`,
  `VOLUME_ZSCORE_WINDOW=22` imported from `pipeline_config`; `_VZ20=20` backward-compat variant is a named
  module constant with a documented reason (delivered results trained on 20), not a bare `.rolling(20)`.
- **cleaning_applied labeling** = observe each cleaner's effect by diffing OHLC before/after that cleaner
  (aligned on the unique `date`), appending the cleaner name; drops recorded in the rejection manifest with a
  reason. Deterministic and directly observed (not re-derived). `dirty_flag` = the RAW bar tripped ≥1 of the
  6 REAL/structural detectors (high<low, open/close-outside, nonpositive, zero-range, split-jump, NaN/inf),
  computed independently on the raw bar (independent of cleaning_applied).
- **market_pk = cross-sectional MEAN of `parkinson_variance`** per the schema spec §2c and the task's explicit
  column definition + test (e). NOTE: this DIFFERS from the delivered model's node feature, which uses
  `nanmedian(sqrt(pk))` (`masked_rich.py:177`) computed at train time over the SCREENED universe. The enriched
  `market_pk` is a per-row causal cache for EDA/verification; the delivered model still computes its own
  train-time market factor and is unaffected (it does not read this file).
- **Clean-bar regression vs existing processed**: enriched `parkinson_variance` == existing to 1e-12 on all
  non-dirty bars EXCEPT the bars the old pipeline upper-CAPPED at 0.1 (VN30: exactly 1 such bar). The 0.1 cap
  is a downstream QLIKE/modeling floor, NOT part of the causal estimator, so the enriched file preserves the
  true uncapped variance. The regression test excludes capped bars and asserts < 1e-12 elsewhere.
- **Estimator NaN policy**: estimators are NaN on non-`ok` bars (invalid geometry) and on documented leading
  windows (`yang_zhang_n20` first n-1). Pandera treats estimator columns as nullable with a ≥0-&-finite check
  on the non-null values; `parkinson_variance` column must be PRESENT.
- **No OHLC columns** in the enriched output (the task column list omits O/H/L/C), so the level-changing
  cleaners (widen_range / backadjust_split) only affect `daily_return` and `yang_zhang_n20` overnight terms;
  Parkinson/GK/RS are scale-invariant and unaffected by back-adjust.

## Performance / batching gate
Data-processing on CPU/pandas (not GPU). Per-ticker work is fully VECTORIZED (numpy/pandas rolling); there is
NO per-row Python loop in the hot path — the only Python loop is the per-TICKER outer loop, which is the
natural unit of independent work. `build_market` takes a `map_fn` (default `map`) so the per-ticker stage is
parallelizable across tickers with a process/thread pool (the CLI exposes `--jobs`); market_pk is a single
vectorized cross-sectional `nanmean` over the assembled wide panel.

## Simplicity gate
No new abstraction beyond one library module (`enrich.py`), a thin CLI (`cli.py`), and an HTML report
(`report.py`). No config/flexibility beyond the required `--markets/--out/--jobs`. Reuses existing tested code.

## Anti-abstraction gate
Uses `pandas`/`numpy` directly and imports the existing `etl_cleaning` / `volatility_estimators` /
`pipeline_config` modules straight — no wrappers.

## File list
- `code/__init__.py`
- `code/enrich.py`     — detectors, `clean_ohlcv`, causal columns, `build_ticker`, `build_market`.
- `code/cli.py`        — `run(...)` (tested) + `main()` (entry driver, `# pragma: no cover`).
- `code/report.py`     — `build_html_report(...)` (tested).
- `code/tests/*`       — formula, look-ahead, cleaning, market_pk, leading-NaN, real-data smoke, cli/report.
- `test/test_structure_smoke.py` — §3.F structural smoke.
- `scripts/quality_gate/data_schemas.py` (+ `test_data_schemas.py`) — ENRICHED_SCHEMA + check.
```
