# Data-pipeline run — hose

- Generated: 2026-09-13 17:24
- Mode: full (incremental=False)

## Phase status

| phase | status | detail |
| --- | --- | --- |
| P1 raw-quality tests | PASS | 301 passed in 2.43s |
| P2 dirty-data audit | PASS | audited 405 tickers -> hose_audit.json |
| P3 ETL clean | PASS | dirty_bars=212022, dropped=515 |
| P4 enrich (causal) | PASS | rows_out=1387666, tickers=405 |
| P5 data-quality gate | FAIL | schema=PASS, enriched=FAIL, drift=INFO |
| P6 freeze/version | PASS | schema_version=enriched-1.1, n_tickers=405 |

## Build summary

- tickers: 405
- rows_out: 1387666
- dirty bars: 212022
- dropped: 515

## Dirty-data audit (per-class ticker-day counts)

| class | count |
| --- | --- |
| high_lt_low | 3 |
| open_close_outside | 3541 |
| nonpositive | 84 |
| zero_range | 211378 |
| split_jumps | 44 |
| stale_runs | 81466 |
| naninf | 23 |
| zero_volume | 74120 |
| leading_backfill | 334 |
