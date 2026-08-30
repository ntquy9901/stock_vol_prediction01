# Dirty-data audit + ETL-cleaning spec — requirements (SDD: Specify)

Date: 2026-08-30. CPU/pandas only (GPU committed to a training job). Objective wording; no fabricated numbers.

## Goal
Audit ALL raw AND processed data across 5 markets (VN: HNX, HOSE, VN30, VN100; US: SP500), pinpoint the exact
`(ticker, date)` rows affected by each dirty-data class, prescribe a concrete ETL cleaning rule per class, and
render self-contained HTML per market (executive summary + sortable per-ticker table + per-stock drill-down
charts) plus one consolidated ETL-cleaning spec markdown.

## Inputs (READ-ONLY)
- Raw OHLCV: `volatility_estimators.PRICE[panel]` (`*_ohlcv.csv`: date,open,high,low,close,volume).
- Processed Parkinson-VARIANCE target (`parkinson_volatility` = sigma^2, NOT sigma):
  - hnx/hose/sp500 -> `data/processed/{hnx,hose,sp500}`
  - vn30/vn100 -> `submission/soict_lstm_gat/data/{vn30,vn100}` (the dirs the delivered runners read;
    equivalently reconstructable from raw via `estimator_forecast_ablation._write_estimator_processed`).
- Screened universe (hose/hnx): `floor_sensitivity.screen_files`.

## Dirty-data classes (detect + quantify per market + list affected (ticker,date))
1. high < low (impossible geometry)
2. open/close outside [low,high]  (headline VN issue)
3. nonpositive OHLC (<=0)
4. zero-range (high==low)  -> Parkinson=0 -> floored target
5. extreme single-day jumps (|log-return|>50%)  -> candidate unadjusted split/dividend
6. backfill-before-listing / leading constant series
7. stale/repeated prices (N consecutive identical closes)
8. NaN/inf in OHLCV
9. zero-volume days

## Output — acceptance criteria (Validate)
- Per-market HTML `docs/reports/2026-08-31_{hnx,hose,vn30,vn100,sp500}_dirty_data_etl.html`, each with:
  (i) executive dirty-data summary table (class x count x % x recommended ETL),
  (ii) SORTABLE per-ticker table (one row/stock: per-class counts + flags),
  (iii) per-stock drill-down plots for the worst offenders per class (top ~8), OHLC line/candle with the
       dirty bars/dates highlighted. Base64-embedded PNG, NO external CDN.
- Consolidated `docs/reports/2026-08-31_etl_cleaning_spec.md`: per class = detection rule + cleaning rule +
  which estimators affected (Parkinson/GK/RS/YZ) + prioritised recommendation (apply vs cosmetic) +
  cross-market prevalence table.
- Every count traces to the data (no fabrication). Distinguish TARGET-affecting (Parkinson) vs cosmetic.

## Go / No-go
- GO when: all 5 HTMLs + spec.md generated from real data; detectors + cleaning funcs unit-tested (TDD);
  pre-push gate green (C0=100/C1>=95 on changed lines, ruff -F clean, data-quality gate); 3-lens code
  review critical/major resolved; existing EDA tests + touched-dir tests show 0 new failures.
- NO-GO: any fabricated number, any cleaning function without a formula-level test, any silent skip.

## Estimator impact map (which classes affect which estimator)
- Parkinson = ln(H/L)^2/(4ln2): uses ONLY high/low. Affected by: high<low, nonpositive H/L, zero-range,
  NaN/inf. NOT affected by open/close-outside, stale close, zero-vol.
- Parkinson (and the GK/RS within-day ratios) are SCALE-INVARIANT: an unadjusted split rescales H and L by
  the same factor, so ln(H/L) is unchanged -> a split does NOT move the Parkinson target on any day. Only
  Yang-Zhang's overnight term ln(O_t/C_{t-1}) crosses the split boundary (affected on the boundary day only).
  => split jumps are COSMETIC for the delivered Parkinson target (corrected per code review 2026-08-30).
- Garman-Klass / Rogers-Satchell / Yang-Zhang: use open+close too -> ALSO affected by open/close-outside.
- => open/close-outside is COSMETIC for the delivered Parkinson target but REAL for O/C estimators.
