# A4 data-pipeline runbook — design note

Single-command orchestrator for the P1->P6 crawl-to-ready flow, one market per run.

    python scripts/data_pipeline/run_pipeline.py --market {vn30|vn100|hose|hnx|sp500} [--incremental] [--dry-run]

## Phases (each is a callable returning a status dict; the CLI runs them in order)

| phase | what it does | reused module |
|---|---|---|
| P1 raw-quality tests | run `tests/test_raw_prices_quality.py` + `tests/test_processed_data_quality.py`; gate structural corruption | pytest (subprocess) |
| P2 dirty-data audit | per-ticker `detect_all` detectors -> per-class ticker-day counts; write `results/data_pipeline_audit/<market>_audit.json` | `scripts/etl_audit/dirty_data_detectors.py` |
| P3 ETL clean | apply the spec's priority-ordered cleaners (drop-naninf -> reconstruct -> swap-high/low -> cut-listing -> widen -> backadjust -> flag) | `enrich.clean_ohlcv` (baseline `2026-08-31_enriched_processed`) |
| P4 enrich | write causal columns to `data/processed_enriched/<market>/` (estimators, HAR, market_pk, volume_zscore) | `enrich.build_market` / `build_ticker` |
| P5 data-quality gate | Pandera `check_schema` + Evidently `check_drift` + `validate_enriched` on the market output | `scripts/quality_gate/` |
| P6 freeze/version | refresh `_schema_version.json` + write `_provenance.json` (schema version, git sha, mode, last_build_date); backward-compat additive | `enrich.SCHEMA_VERSION` |

P3 and P4 share the one `enrich.build_market` call (the enriched writer cleans then enriches); the phase
dict reports them separately.

## `--incremental` (daily append)

Recompute only dates newer than the last build. Correctness rests on a causal lookback:

    INCREMENTAL_LOOKBACK = max(HAR_MONTHLY_WINDOW, VOLUME_ZSCORE_WINDOW, YZ_N)   # = 22 (all from config)

The tail is recomputed from `first_new_index - INCREMENTAL_LOOKBACK` so every trailing rolling window on a
kept (new) row sees a fully-populated window identical to a full rebuild — the recomputed tail equals the
full-rebuild tail (no look-ahead; verified by test). `market_pk` for the new dates is the same-day
cross-sectional mean over the new-date slice (causal). New rows are appended; frozen history is untouched.
Assumes daily append has no historical split / listing-cut inside the new window (documented; scale-invariant
Parkinson/GK/RS are unaffected by level rescales regardless).

- First-ever build (no `_schema_version.json`): falls back to a full build.
- No new dates: no-op (0 rows appended, nothing written).

## `--dry-run`

Reports what each phase WOULD do and writes nothing (no data, no audit artifact, no report).

## Config / reuse

All windows/thresholds import from `submission/soict_lstm_gat/pipeline_config.py` and
`scripts/eda/volatility_estimators.py` (`_YZ_N`). No estimator/cleaner is reimplemented — the existing,
independently-tested modules are called.
