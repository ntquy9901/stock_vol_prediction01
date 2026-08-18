# Quality Gate Report

- Timestamp: 2026-08-08_153025
- Project root: stock_vol_prediction01
- Overall: FAIL

## Checks

| Check | Status | Detail |
| --- | --- | --- |
| LINT | FAIL | Found 820 errors. |
| TESTS | FAIL | 4 skipped, 1 error in 14.83s |
| SCHEMA | PASS | 34/34 artifacts valid |
| DRIFT | SKIPPED | --fast (drift skipped) |

## Notes

- HARD checks (LINT, TESTS, SCHEMA) determine the exit code. DRIFT is informational and SKIPPED checks do not fail the gate.
- Data validated: C:\luanvan\stock_vol_prediction01\data\processed (per-ticker CSVs) and dual_group_news_panel.parquet.
