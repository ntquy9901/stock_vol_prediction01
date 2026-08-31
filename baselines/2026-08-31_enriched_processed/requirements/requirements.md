# Baseline A3 — ETL-clean + ENRICH the processed data (requirements / spec)

Date: 2026-08-31. Follows CLAUDE.md §5 SDD. Constitution = CLAUDE.md.

## Goal
Produce CLEANED + ENRICHED per-ticker processed files so downstream models read cached, verifiable,
CAUSAL columns instead of recomputing estimators/features on the fly. Reuse the already-tested building
blocks (`scripts/etl_audit/etl_cleaning.py`, `scripts/eda/volatility_estimators.py`,
`submission/soict_lstm_gat/pipeline_config.py`); do NOT re-derive any formula.

## Inputs
- Raw OHLCV CSVs `date,open,high,low,close,volume` per ticker, 5 markets:
  - vn30  → `data/raw/prices/*_ohlcv.csv` (33)
  - vn100 → `data/raw/prices/vn100_vnstock/*_ohlcv.csv` (104)
  - hose  → `data/raw/prices/hose_vnstock/*_ohlcv.csv` (406)
  - hnx   → `data/raw/prices/hnx_vnstock/*_ohlcv.csv` (300)
  - sp500 → `data/raw/prices/sp500/*_ohlcv.csv` (503)

## Output
- `data/processed_enriched/<market>/<ticker>.csv` (NEW versioned location; existing
  `data/processed/*_processed.csv` is left untouched so delivered-baseline tests stay green).
- Per-ticker rejection manifest `data/processed_enriched/<market>/<ticker>_rejections.csv`
  (`date,reason`) for DROPPED rows (no silent deletion).
- `data/processed_enriched/<market>/_schema_version.json` sidecar (schema_version + build meta).
- HTML build report `docs/reports/2026-08-31_enriched_processed_build.html`.

## Enriched columns (all on POST-ETL-CLEANED OHLC; all CAUSAL / backward-looking)
`date` (key: monotonic, unique, weekday), `parkinson_variance`, `garman_klass_variance`,
`rogers_satchell_variance`, `yang_zhang_n20`, `log_range=ln(H/L)`, `daily_return=ln(C_t/C_{t-1})`,
`har_daily` (= parkinson_variance at t), `har_weekly` (rolling mean 5, min_periods 5),
`har_monthly` (rolling mean 22, min_periods 22), `market_pk` (cross-sectional MEAN of
parkinson_variance over the panel's VALID tickers at day t — same-day, no future),
`volume_zscore_22` (trailing 22d z-score of log1p(volume), CANONICAL), `volume_zscore_20`
(trailing 20d, backward-compat), `dirty_flag` (bool), `cleaning_applied` (str),
`zero_range_flag` (bool), `zero_volume_flag` (bool).

## Success criteria (acceptance)
1. Clean-bar regression: for every non-dirty bar whose existing `data/processed` value is NOT at the
   0.1 modeling cap, enriched `parkinson_variance` equals the pre-existing value to < 1e-12
   (VERIFIED on VN30: worst non-capped diff = 1e-16; exactly 1 capped bar, documented). Genuinely-dirty
   bars (nonpositive / high<low / NaN) may differ and are flagged in `dirty_flag`/`cleaning_applied` +
   rejection manifest.
2. Every estimator column: test-vs-published-formula (independent recompute, not reusing implementation).
3. Every causal column: no-look-ahead test (perturb rows > t, value at t unchanged).
4. Leading-NaN coverage: `har_monthly` first 21 NaN, `yang_zhang_n20` first 19 NaN,
   `volume_zscore_22` first 21 NaN, `volume_zscore_20` first 19 NaN.
5. `market_pk` equals the recomputed cross-sectional mean on a sampled date.
6. Pandera schema extended for the enriched columns + adjacent test green.
7. Data-quality gate (Pandera + Evidently) run on the enriched output; evidence captured.
8. Delivered-baseline tests + repo data-quality tests: 0 new failures.
9. Pre-push quality gate passes with NO QG_SKIP.

## Leakage rule (schema spec §3) — HARD constraint
Only causal per-row columns. Do NOT bake: train-only scalers, whole-series z-scores, graph adjacency,
or any future/centered-window/shift(-k) target. Row t depends solely on dates ≤ t and NOT on any
train/val/test boundary.

## Go / No-go
- GO to enrich a market only after its raw slice reads and the estimator recompute matches on clean bars.
- NO-GO (leave for a follow-up) if a market's enriched parkinson_variance diverges from the delivered
  value on a NON-dirty, non-capped bar — that signals an estimator/cleaning mismatch, not enrichment.

## Out of scope (explicitly deferred)
- Changing the delivered model feature from `volume_zscore_20` to `_22` (separate rerun + approval).
- Rewriting `data/processed/*_processed.csv` in place.
- Any train-time artifact (scalers, adjacency, targets).
