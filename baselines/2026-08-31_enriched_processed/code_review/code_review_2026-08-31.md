# Code review — baseline A3 (enriched processed) — 2026-08-31

Method: adversarial 3-lens review (Lens 1 correctness & leakage, Lens 2 edge cases, Lens 3
performance & config-hardcode) over `code/enrich.py`, `code/cli.py`, `code/report.py`, and the
enriched section of `scripts/quality_gate/data_schemas.py`. Reviewer read the three reused modules
(`etl_cleaning`, `volatility_estimators`, `pipeline_config`) to verify formula/window imports and
label alignment, and ran repros for the edge cases. `archive/` out of scope.

## Verdict
No CRITICAL and no MAJOR defect. Hard rules verified to hold:
- **Leakage:** every output column is backward-looking (trailing `.rolling`, `prev_c`=t−1) or same-day
  cross-sectional (`market_pk`). No scaler / whole-series stat / adjacency / future/centered window / split
  boundary. Compliant with schema-spec §3.
- **Formula reuse:** estimators imported (`estimators_from_ohlcv`) with correct name→column mapping;
  `yang_zhang_n20` = the windowed true YZ. HAR/volume windows from `pipeline_config` (no hardcoded rolling);
  `_VZ20=20` is the documented back-compat exception.
- **Label/reject alignment:** `cleaning_applied` + rejection manifest align on the `date` column through
  drops/reorders (verified on a mixed dirty frame).
- **Performance:** per-ticker work fully vectorized; per-ticker stage parallelized via `ProcessPoolExecutor`
  (`cli._map_fn`); `compute_market_pk` is a single vectorized `DataFrame.mean(axis=1)`.

## Findings (all MINOR) and disposition
| # | finding | disposition |
|---|---|---|
| M1 | An all-dropped ticker wrote a header-only CSV that the enriched Pandera schema then rejected for a spurious `WRONG_DATATYPE` reason. | **FIXED** — `build_market` now excludes 0-row frames from the panel + written data, but still writes their rejection manifest (audit trail). Test `test_build_market_skips_all_dropped_ticker_but_keeps_its_manifest`. |
| M2 | `rows_in` undercounted raw rows (parse/dedup drops omitted) — report figure only. | **FIXED** — `build_ticker` returns the true pre-parse `_n_raw`; `build_market` sums it. |
| M3 | `regression_vs_processed(cap=0.1)` diagnostic threshold not sourced from `pipeline_config`. | **DOCUMENTED** — diagnostic-only (VN30 clean-bar sanity, never written to data); `0.1` is the delivered ETL's known Parkinson upper-clip. Comment added; not a produced-data constant, so no `pipeline_config` entry needed. |
| M4 | Regression diagnostic could silently return `n_compared=0` on a date-string-format mismatch (false "perfect agreement"). | **FIXED** — both date columns normalized to datetime before merge; real-vn30 test asserts `n_compared>1000`. |

## Concerns checked and refuted (not bugs)
- Empty-after-clean does NOT crash (numpy `(1,)`↔`(0,)` broadcast is clean).
- `_volume_zscore` returns NaN (not neutral zeros) when volume absent → CLAUDE.md fail-loud/no-silent-degradation compliant.
- backadjust labels only the split-boundary day (scale-invariant rescale; prior-row ratio columns unchanged) — documented tradeoff.
- `log_range` outside the estimator `ok`-mask is safe (post-clean bars satisfy the geometry the estimator tests).

## Separate raw-data finding (out of scope for this pass)
`data/processed_enriched/hnx/BBS.csv` carries 1 weekend-dated bar (Sat 2006-04-01) inherited from the HNX
raw file — a raw-ingestion (P1) defect in a LOCAL-ONLY market. Committed markets (VN30, VN100) have 0
weekend bars. The enriched Pandera check correctly flags it; fix belongs upstream in raw ingestion, not here.

## Gate status at review time
- Baseline tests: 39 passed. Coverage on changed code: C0 line 100% / C1 branch 100% (`enrich`, `cli`,
  `report`); env-dependent skip guards + import bootstraps marked `# pragma: no cover`.
- ruff `--select F`: clean. config-hardcode scan: no BLOCK.
- Pandera `validate_enriched`: 1343/1344 pass (the 1 fail = the HNX BBS weekend-date raw defect above).
- Evidently drift: `drift.html` generated on the enriched VN30 columns (12 numeric features).
