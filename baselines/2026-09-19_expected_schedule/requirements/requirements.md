# Requirements — Expected-schedule earnings (leakage-safe), 2026-09-19

## Goal
Replace the realized (actual, ex-post crawled) earnings dates with a strictly leakage-safe
**expected schedule** as the MAIN earnings feature for the paper's XGB+E / XGB+E+LG models, on both
markets, and produce NEW result JSONs without overwriting the 2026-09-18 actual-date baseline.

## Why
Using the realized date assumes the exact date of a FUTURE release was known at the forecast origin
`t` — a point-in-time leak (the date is "past" only relative to today's crawl, not relative to `t`).
The expected schedule predicts each release from the firm's OWN past reporting cadence
(previous release + median of PRIOR inter-release gaps), so every value is knowable at `t` → 100%
leakage-free.

## Input
- Actual dates: `results/gamma_gbm/sp500_earnings.parquet` (yfinance) and, for HOSE, the SSC disclosure
  archive `data/raw/vn_earnings/hose_disclosures.csv` (has `quarter` / `scope` labels).
- Processed panels `data/processed_enriched/{sp500_clean,hose}/*.csv` (own-history features), via
  `full_matrix`.

## Output
- `results/gamma_gbm/expected_schedule/leaf_graph_paper_{sp500,hose}_full_h{1,5,10,22}.json`
  (XGB+E base + XGB+E+LG, with metrics + DM + spike-robustness + over/under-fit evidence + alpha).
- The XGB (no-earn) column is earnings-independent → reuses the committed
  `leaf_graph_paper_{market}_noearn_h*.json` unchanged (documented, not re-run).

## Method
- HOSE: reduce the multi-filing disclosure stream to one clean QUARTERLY event per
  (ticker, fiscal-year, quarter) = earliest disclosure across scopes (parent/consolidated/combined are
  ~same-day), then apply the expected schedule. SP500 is already quarterly.
- Reuse the 2026-09-18 leaf-graph runner (`run_leaf_graph_paper.run`) unchanged by injecting an
  expected-schedule `_load_earn` at runtime (the sibling baseline's file is NOT modified).

## Success criteria (go/no-go)
- Runner produces gate-valid JSONs (train/val/test fit evidence, DM, spike for HOSE) for both markets ×
  4 horizons, `use_earn=True`.
- Expected-vs-actual quality check passes (median discrepancy small on the clean quarterly / SP500
  schedule); recorded before the switch.
- Pure helpers unit-tested (predict_schedule / expected_schedule / hose_quarterly_dates /
  expected_vs_actual / summarize), C0/C1 gate met.
- The 2026-09-18 actual-date results and paper remain untouched (new baseline + new paper version).

## Expected result (from the pre-switch quality check)
SP500 earnings gain drops from actual +8.7–11.2% to expected +2.8–3.3% (still DM-significant); HOSE
earnings remains inert. This is the leakage-safe headline the new paper reports.
