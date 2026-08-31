# Data-pipeline run — vn30

- Generated: 2026-08-31 18:53
- Mode: full (incremental=False)

## Phase status

| phase | status | detail |
| --- | --- | --- |
| P1 raw-quality tests | PASS | 301 passed in 2.18s |
| P2 dirty-data audit | PASS | audited 4 tickers -> vn30_audit.json |
| P3 ETL clean | PASS | dirty_bars=164, dropped=0 |
| P4 enrich (causal) | PASS | rows_out=14440, tickers=4 |
| P5 data-quality gate | PASS | schema=PASS, enriched=PASS, drift=INFO |
| P6 freeze/version | PASS | schema_version=enriched-1.0, n_tickers=4 |

## Build summary

- tickers: 4
- rows_out: 14440
- dirty bars: 164
- dropped: 0

## Dirty-data audit (per-class ticker-day counts)

| class | count |
| --- | --- |
| high_lt_low | 0 |
| open_close_outside | 0 |
| nonpositive | 0 |
| zero_range | 164 |
| split_jumps | 0 |
| stale_runs | 132 |
| naninf | 0 |
| zero_volume | 14 |
| leading_backfill | 0 |
