# Data-pipeline run — vn100

- Generated: 2026-09-01 04:49
- Mode: full (incremental=False)

## Phase status

| phase | status | detail |
| --- | --- | --- |
| P1 raw-quality tests | PASS | 301 passed in 2.24s |
| P2 dirty-data audit | PASS | audited 104 tickers -> vn100_audit.json |
| P3 ETL clean | PASS | dirty_bars=11921, dropped=48 |
| P4 enrich (causal) | PASS | rows_out=357802, tickers=104 |
| P5 data-quality gate | FAIL | schema=PASS, enriched=FAIL, drift=INFO |
| P6 freeze/version | PASS | schema_version=enriched-1.1, n_tickers=104 |

## Build summary

- tickers: 104
- rows_out: 357802
- dirty bars: 11921
- dropped: 48

## Dirty-data audit (per-class ticker-day counts)

| class | count |
| --- | --- |
| high_lt_low | 0 |
| open_close_outside | 0 |
| nonpositive | 0 |
| zero_range | 11968 |
| split_jumps | 2 |
| stale_runs | 6825 |
| naninf | 0 |
| zero_volume | 4197 |
| leading_backfill | 48 |
