# Data-pipeline run — hnx

- Generated: 2026-09-13 17:36
- Mode: full (incremental=False)

## Phase status

| phase | status | detail |
| --- | --- | --- |
| P1 raw-quality tests | PASS | 301 passed in 2.47s |
| P2 dirty-data audit | PASS | audited 299 tickers -> hnx_audit.json |
| P3 ETL clean | PASS | dirty_bars=486306, dropped=730 |
| P4 enrich (causal) | PASS | rows_out=1068720, tickers=299 |
| P5 data-quality gate | PASS | schema=PASS, enriched=PASS, drift=INFO |
| P6 freeze/version | PASS | schema_version=enriched-1.1, n_tickers=299 |

## Build summary

- tickers: 299
- rows_out: 1068720
- dirty bars: 486306
- dropped: 730

## Dirty-data audit (per-class ticker-day counts)

| class | count |
| --- | --- |
| high_lt_low | 3 |
| open_close_outside | 3440 |
| nonpositive | 174 |
| zero_range | 486272 |
| split_jumps | 53 |
| stale_runs | 297528 |
| naninf | 2 |
| zero_volume | 287238 |
| leading_backfill | 423 |
