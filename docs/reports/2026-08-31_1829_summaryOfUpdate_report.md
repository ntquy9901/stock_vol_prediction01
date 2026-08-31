# Summary of update — Baseline A3: ETL-clean + ENRICH processed data (2026-08-31)

## What changed
New baseline `baselines/2026-08-31_enriched_processed/` that ETL-cleans raw OHLCV and writes cached,
verifiable, CAUSAL enriched columns per ticker to a NEW versioned location
`data/processed_enriched/<market>/<ticker>.csv`. The existing `data/processed/*_processed.csv` are left
byte-identical (delivered-baseline tests unaffected). Reuses the tested building blocks
(`scripts/etl_audit/etl_cleaning.py`, `scripts/eda/volatility_estimators.py`, `pipeline_config.py`);
no formula re-derived.

## Files (path -> purpose)
- `baselines/2026-08-31_enriched_processed/requirements/requirements.md` — spec, acceptance criteria, go/no-go.
- `.../design/design.md` — data flow, decisions, 3 SDD gates.
- `.../code/enrich.py` — detectors, `clean_ohlcv` (ETL priority order), causal columns, `build_ticker`,
  `build_market` (cross-sectional `market_pk`), `regression_vs_processed`.
- `.../code/cli.py` — `run()` + `main()` (entry driver), `--markets/--jobs/--limit/--html`.
- `.../code/report.py` — self-contained HTML build report.
- `.../code/tests/*` + `.../test/test_structure_smoke.py` — formula / look-ahead / cleaning / market_pk /
  leading-NaN / real-data smoke / cli+report / structure.
- `scripts/quality_gate/data_schemas.py` (+ adjacent `test_data_schemas.py`) — `ENRICHED_SCHEMA` +
  `validate_enriched()` for the new columns (schema-spec §5).
- `docs/reports/2026-08-31_enriched_processed_build.html` — per-market build report.
- `data/processed_enriched/{vn30,vn100}/**` — committed enriched output (tracked markets);
  `hose/hnx/sp500` are local-only (gitignored).

## Enriched columns (all causal / backward-looking)
`date, parkinson_variance, garman_klass_variance, rogers_satchell_variance, yang_zhang_n20, log_range,
daily_return, har_daily, har_weekly, har_monthly, market_pk, volume_zscore_22, volume_zscore_20,
dirty_flag, cleaning_applied, zero_range_flag, zero_volume_flag` + per-ticker `<ticker>_rejections.csv`
and a `_schema_version.json` sidecar (schema_version=`enriched-1.0`).

## Build results (5 markets, ~7.3M rows, 100s @ jobs=6)
| market | tickers | rows_out | dirty_bars | dropped |
|---|---|---|---|---|
| vn30 | 33 | 106,634 | 1,844 | 14 |
| vn100 | 104 | 357,802 | 11,921 | 48 |
| hose | 405 | 1,387,666 | 212,022 | 515 |
| hnx | 299 | 1,068,721 | 486,307 | 730 |
| sp500 | 503 | 4,379,290 | 50,280 | 101 |

Dirty-bar counts track the ETL-spec prevalence table (zero-range dominates on HNX/HOSE).

## Clean-bar regression (acceptance criterion 1)
VN30 enriched `parkinson_variance` vs delivered `data/processed`: **worst non-capped diff = 1e-16**
(< 1e-12) over 104,789 compared bars; exactly **1** capped bar (the delivered ETL's 0.1 upper-clip)
excluded and preserved uncapped in the enriched file. Committed markets have 0 dirty divergences on
clean bars.

## Leading-NaN coverage (acceptance criterion 4)
`har_weekly`=4, `har_monthly`=21, `volume_zscore_20`=19, `volume_zscore_22`=21 (window−1);
`yang_zhang_n20`=20 (n−1 plus the row-0 overnight-return NaN in the YZ overnight variance window — the
spec's "19" ignored the first-row overnight NaN; 20 is correct behavior).

## Tests + coverage
- 39 baseline tests + 2 structure-smoke pass. Adjacent `test_data_schemas.py`: 10 tests pass.
- Coverage on changed code (GPU venv, `--cov-branch`): **C0 line 100% / C1 branch 100%** for
  `enrich.py`, `cli.py`, `report.py`; env-dependent skip guards + import bootstraps marked
  `# pragma: no cover`.
- Test kinds present: test-vs-published-formula (independent recompute for parkinson/GK/RS/YZ),
  no-look-ahead per causal column, ETL cleaning (dirty flagged+fixed, drops manifested),
  market_pk cross-sectional-mean, leading-NaN, real-data-sample smoke per market.

## Data-quality gate (Pandera + Evidently)
- Pandera `validate_enriched`: **1343/1344 pass**. The 1 failure = `hnx/BBS.csv` weekend-dated bar
  (Sat 2006-04-01), a RAW-ingestion defect in a local-only market; committed VN30/VN100 = 0 weekend bars.
  Fix belongs upstream (P1 raw quality), out of scope for this enrichment pass.
- Evidently drift: `drift.html` generated on the enriched VN30 numeric columns (12 features), train/test
  70/30 split -> `results/quality_gate/enriched_*/drift.html`.

## Code review (adversarial 3-lens) — result + actions
No CRITICAL/MAJOR. 4 MINOR: M1 (empty-ticker header-only CSV) FIXED (skip 0-row frames, keep manifest),
M2 (rows_in undercount) FIXED (honest `_n_raw`), M4 (regression date-format silent-zero) FIXED (datetime
normalize before merge), M3 (diagnostic `cap=0.1`) DOCUMENTED (diagnostic-only, never written to data).
Full record: `baselines/2026-08-31_enriched_processed/code_review/code_review_2026-08-31.md`.

## Performance
Data-processing on CPU/pandas. Per-ticker work fully vectorized (no per-row Python hot loop);
per-ticker stage parallelized across tickers via `ProcessPoolExecutor` (`--jobs`); `market_pk` a single
vectorized cross-sectional `nanmean`. 5-market build = ~100s @ jobs=6.

## Leakage
Only causal per-row columns baked. No train/val/test-boundary statistic, scaler, adjacency, or
future/centered-window/shift(-k) target. `market_pk` is a same-day cross-sectional mean. No-look-ahead
test asserts each causal column at t is unchanged when rows > t are perturbed.

## Risks / follow-ups
- Committed enriched CSVs are ~128 MB (vn30 30 MB + vn100 98 MB); heavier than the 2-column delivered
  processed files (17 columns). Intended per task (tracked source markets).
- HNX BBS weekend-date raw defect to fix in raw ingestion (P1).
- Switching the delivered model feature from `volume_zscore_20` to `_22` remains a separate,
  approval-gated rerun (out of scope; both columns cached for that comparison).

## DoD checklist
- [x] SDD artifacts (requirements/design/code/code_review/test).
- [x] Tests + C0=100%/C1=100% on changed lines; ruff F clean; config-hardcode no BLOCK.
- [x] Adversarial code review run + findings resolved.
- [x] Data-quality gate (Pandera + Evidently) run + evidence captured.
- [x] Delivered-baseline `data/processed` untouched (0 new failures expected in gate step-5).
- [x] Summary report (this file).
