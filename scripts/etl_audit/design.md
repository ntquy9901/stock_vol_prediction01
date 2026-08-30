# Dirty-data audit + ETL-cleaning — design (SDD: Plan)

## Modules (scripts/etl_audit/, CPU/pandas only)
- `dirty_data_detectors.py` — pure, per-`(ticker,date)` LOCATE detectors on one raw OHLCV frame. Returns the
  specific offending dates + magnitudes (the existing `vnmarkets_eda` detectors return counts only). No torch.
- `etl_cleaning.py` — NEW pure ETL cleaning functions, each `(df) -> (cleaned_df, info)`; independently tested
  against the published formula/behaviour. No torch.
- `build_dirty_data_report.py` — driver: load raw+processed per market, run detectors, build per-ticker table
  + drill-down charts, render per-market HTML + consolidated spec markdown. matplotlib IO = `# pragma: no cover`.

## Gates (per CLAUDE.md §5)
- Simplicity: reuse `volatility_estimators.PRICE`, `floor_sensitivity.screen_files`; no new abstraction layer.
- Anti-abstraction: plain pandas; detectors are free functions, not a class hierarchy.
- Performance: audit is read-only per-file streaming (RAM bounded, one market at a time). No batch=1 hot loop
  on GPU (no model). Vectorised numpy per frame. `--limit` for smoke.

## Detector contracts (all operate on a date-sorted, deduped frame; tolerance _OHLC_RTOL=1e-5 matching gate)
- `high_lt_low(df) -> list[str]` dates where high < low.
- `open_close_outside(df) -> list[(date, rel_violation)]` where high<max(o,c)*(1-rtol) or low>min(o,c)*(1+rtol);
  rel_violation = max relative overshoot (magnitude for the "median relative violation" stat).
- `nonpositive(df) -> list[str]` any of O/H/L/C <= 0 (finite).
- `zero_range(df) -> list[str]` finite positive high==low.
- `split_jumps(df, thresh=0.5) -> list[(date, simple_ret)]`.
- `stale_runs(df, min_run=5) -> list[(start,end,length)]`.
- `naninf(df) -> list[str]` rows with non-finite O/H/L/C or volume.
- `zero_volume(df) -> list[str]` finite volume == 0.
- `leading_backfill(df) -> dict{n_leading, first_trade_date}` leading run of constant close AND
  (zero volume OR zero range) = pre-listing backfill.
- `detect_all(df) -> dict` counts + capped example lists for each class.
- `per_ticker_summary(ticker, df) -> dict` one flat row for the sortable table.

## ETL cleaning contracts (pure; return cleaned frame + info)
1. `widen_range(df)` H=max(H,O,C), L=min(L,O,C) -> O/C internally consistent; Parkinson H/L change only where
   O/C exceeded the recorded range. RECOMMENDED for O/C-outside. Info: n_widened.
2. `clip_oc(df)` clip O,C into [L,H] (alternative; loses the trade info).
3. `swap_or_drop_high_low(df)` if high<low: swap when swap yields valid geometry (transposition), else drop.
4. `reconstruct_nonpositive(df)` per row with any nonpositive OHLC: H=max(positive OHLC), L=min(positive OHLC),
   clamp O/C into [L,H]; if <2 positive values -> drop row. Yields positive OHLC (CLAUDE.md rule).
5. `backadjust_splits(df, thresh=0.5)` detect |simple ret|>thresh; factor=close_t/close_{t-1}; multiply ALL
   prior-day OHLC by factor so the level jump is removed (|log-ret| at the day < thresh afterward).
6. `cut_to_listing(df)` drop the leading backfill run (via `leading_backfill`).
7. `drop_naninf(df)` drop rows with non-finite OHLCV.
8. `flag_zero_range(df)` / `flag_zero_volume(df)` KEEP rows, add boolean flag column (liquidity screen /
   vol floor is the right handling, not deletion).

## RAW-vs-PROCESSED comparison
For each market: recompute raw Parkinson from raw OHLCV, align to processed `parkinson_volatility`, and MEASURE
(no assumption): fraction of processed rows exactly at the 0.1 upper value while raw Parkinson exceeded 0.1
(evidence of an ETL upper-clip), and processed max. Report which issues the current ETL already removes
(nonpositive/NaN dropped, upper-clip) vs which pass through (zero-range -> floored target).

## Testing (TDD, failing-first)
- `test_dirty_data_detectors.py` — synthetic frames with each planted defect + a cross-check that detector
  counts equal `vnmarkets_eda.detect_ohlc_violations` on the same frame + a real-data-sample smoke per market
  (skips cleanly, `# pragma: no cover` on the skip guard, when a market's data is absent).
- `test_etl_cleaning.py` — one formula/behaviour test per cleaning function (widen -> internal consistency;
  reconstruct -> positive OHLC; backadjust -> jump removed; swap -> valid geometry; cut-to-listing -> dropped;
  flags -> rows kept + marked).
- `test_build_dirty_data_report.py` — smoke: tiny synthetic + `--limit` real slice -> HTML string has the
  required sections; self-contained (no `http`/CDN `src`); spec md has every class. Skips if data absent.

## Coverage
matplotlib chart->base64 and pure-HTML string builders that are only exercisable via a full render are
`# pragma: no cover` (IO), consistent with `vnmarkets_eda`. All detector + cleaning + aggregation logic is
covered to C0=100 / C1>=95 on changed lines.
