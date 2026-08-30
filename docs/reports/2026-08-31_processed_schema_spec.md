# Enriched processed-file schema — DESIGN SPEC (design only; no implementation this pass)

Date: 2026-08-30. Companion to `2026-08-31_etl_cleaning_spec.md`. This document specifies a richer
processed-CSV schema so downstream code can read cached, verifiable, causal columns instead of recomputing
them on the fly. It is DESIGN ONLY: no processed files are rewritten and no delivered feature is changed in
this pass (that is a separate later baseline requiring a full model rerun). Estimators here are defined on the
POST-ETL CLEANED OHLC produced by the rules in `2026-08-31_etl_cleaning_spec.md`.

## 1. Why enrich (motivation)
- **Compute-once / read-later.** `estimator_forecast_ablation._write_estimator_processed` currently
  recomputes an estimator target from raw OHLCV into a temp dir on every run (order-sorts, dedups, recomputes
  rolling estimators per ticker). The delivered processed files carry essentially one column
  (`parkinson_volatility`). Baking the causal columns once removes repeated recomputation and temp-file churn.
- **Verifiability.** A reviewer can open one processed file and see the estimator inputs (`log_range`,
  `daily_return`), the estimators, and the audit flags side by side, and independently recompute any value.
- **Data-mining readiness.** HAR terms, market factor, and volume shock are the standard EDA/feature set;
  caching them (causal only) makes cross-ticker mining and ablations reproducible from a single artifact.

## 2. Columns to add (all computed on POST-ETL CLEANED OHLC; all CAUSAL / backward-looking only)
`date` (monotonic, unique, weekday) is the key. `parkinson_volatility` (= sigma^2 VARIANCE) is KEPT under its
existing name for backward compatibility (see §5). Sigma-vs-variance: all estimator columns below are
VARIANCES (sigma^2), non-negative.

### 2a. Per-day estimators (published formula; each requires an independent test-vs-formula before use)
| column | formula (source) | inputs | notes |
|---|---|---|---|
| `parkinson` | ln(H/L)^2 / (4 ln2)  (Parkinson 1980) | H, L | == existing `parkinson_volatility`; scale-invariant |
| `garman_klass` | 0.5 ln(H/L)^2 − (2 ln2 − 1) ln(C/O)^2  (Garman–Klass 1980) | O,H,L,C | uses open/close |
| `rogers_satchell` | ln(H/C)ln(H/O) + ln(L/C)ln(L/O)  (Rogers–Satchell 1991), clipped ≥0 | O,H,L,C | drift-independent |

These already exist and are tested in `scripts/eda/volatility_estimators.py`
(`estimators_from_ohlcv`) — the enriched writer must REUSE that implementation, not re-derive it, and its
`tests/` (e.g. `test_windowed_yang_zhang_matches_reference_formula`) must gate the cached column values.

### 2b. Windowed Yang–Zhang (NOT per-day — document the window explicitly)
| column | definition | window |
|---|---|---|
| `yang_zhang_n20` | σ²_overnight + k·σ²_open-close + (1−k)·σ²_RS, mean-subtracted n-day SAMPLE variances of ln(O_t/C_{t−1}) and ln(C_t/O_t) + rolling-mean RS; k=0.34/(1.34+(n+1)/(n−1)) (Yang–Zhang 2000) | **n = 20 trading days** |

The column name carries the window (`_n20`) so it is never mistaken for a per-day value. Undefined (NaN) for
the first n−1 rows. Do NOT ship a per-day "yz_daily" proxy under a Yang–Zhang name (CLAUDE.md: named
estimators must use the published formula; the per-day proxy is a different quantity).

### 2c. Intermediates for verification + mining (causal)
| column | definition | causal? |
|---|---|---|
| `log_range` | ln(H/L) on cleaned H,L | yes (same-day) |
| `daily_return` | ln(C_t / C_{t−1}) | yes (uses only past close) |
| `har_daily` | `parkinson` at day t (the daily HAR term) | yes |
| `har_weekly` | backward rolling mean of `parkinson`, window 5 (min_periods 5) | yes (trailing) |
| `har_monthly` | backward rolling mean of `parkinson`, window 22 (min_periods 22) | yes (trailing) |
| `market_pk` | cross-sectional mean of `parkinson` across the panel's VALID tickers at day t | yes (same-day, no future) |
| `volume_zscore_22` | trailing 22-day z-score of `log1p(volume)` on the ticker's own series | yes (trailing) — CANONICAL, see §4 |
| `volume_zscore_20` | trailing 20-day z-score of `log1p(volume)` | yes (trailing) — BACKWARD-COMPAT, see §4 |

`market_pk` is per-day and same-day (a contemporaneous cross-section), which is how the delivered runner uses
it; it is NOT forward-looking. It must be computed over the SAME screened universe used at training so the
cached value matches what the model consumes.

### 2d. Audit flags (from the dirty-data detectors / ETL cleaners in this pass)
| column | type | meaning |
|---|---|---|
| `dirty_flag` | bool | the RAW bar tripped ≥1 dirty-data detector (high<low, O/C-outside, nonpositive, zero-range, split-jump, NaN/inf) |
| `cleaning_applied` | string | which ETL rule was applied to this bar: one of `none, widen_range, clip_oc, swap_high_low, reconstruct_nonpositive, backadjust_split, dropped(reason), flag_zero_range, flag_zero_volume` |
| `zero_range_flag` | bool | cleaned high==low (limit/no-trade) — carried so the model can apply a liquidity/vol floor |
| `zero_volume_flag` | bool | volume==0 (illiquidity) |

`dropped` rows are removed from the file, but the DROP is recorded in a per-ticker rejection manifest
(`<ticker>_rejections.csv`: date, reason) so the audit trail is complete (no silent deletion — CLAUDE.md).

## 3. CRITICAL leakage rule (state explicitly so a future implementer does not leak)
ONLY causal, backward-looking, per-row computations may be baked into the processed file. The following MUST
NOT be pre-computed into the file and MUST remain at model-training time, fit on TRAIN ONLY per split:
- **Per-ticker / global scalers** (StandardScaler mean/std, min-max) — fit on the train fold only, applied to
  val/test. Baking a whole-series scaler leaks test statistics into training (the exact "fit-never-applied"
  and whole-series-normalisation traps in CLAUDE.md).
- **Global (whole-series) z-scores / standardisation** — any statistic computed over the FULL series
  (including future rows). Note `volume_zscore_*` here is a TRAILING window (causal) and is allowed; a
  full-series z-score is NOT.
- **Graph adjacency** (correlation / lead-lag edges) — computed on the train window per split at model time;
  it mixes tickers and time and is not a per-row causal quantity.
- **Any future-or-centered-window feature** — centered rolling, `shift(-k)` targets, look-ahead fills,
  forward-filled prices across a gap. Targets (`parkinson` at t+h) are formed at training time, not stored.
Rule of thumb for inclusion: a column is allowed ONLY if row t's value depends solely on data at dates ≤ t
AND does not depend on a train/val/test boundary.

## 4. Volume-zscore window inconsistency (user-flagged; real)
- **Fact (grounded in code):** the delivered feature is `volume_zscore_20` — a TRAILING 20-day z-score of
  `log1p(volume)` (`baselines/2026-08-21_har_anchored_residual/code/masked_rich.py:34` `_VOL_WIN = 20`,
  `_volume_zscore_wide`). The project's monthly convention is 22 trading days
  (`data_utils.py:23` `MONTHLY_WIN = 22`; `har_monthly = rolling(22)`). So the volume window (20) is
  inconsistent with the monthly HAR window (22).
- **Recommendation:** in the enriched schema compute `volume_zscore_22` (22-day, CONSISTENT with the monthly
  convention) as the CANONICAL volume column, and KEEP `volume_zscore_20` as a backward-compat column so the
  delivered results (trained on `_20`) still reproduce exactly.
- **Do NOT change the delivered feature now.** Switching the models from `_20` to `_22` is a SEPARATE decision
  requiring a full model rerun (all datasets/horizons/seeds). Expected effect is near-negligible (a 2-day
  difference in a 20-day trailing window), but it must be measured, not assumed, and only after approval.
- Keep the same `log1p(volume)` transform and the same causal trailing-window definition for both columns so
  the only difference is the window length.

## 5. Validation plan for the enriched file
- **Pandera schema** (extend `scripts/quality_gate/` `check_schema()`): `date` monotonic increasing + unique +
  weekday; OHLC (if OHLC is carried) valid geometry (high≥low, open/close∈[low,high], all positive); every
  estimator column ≥ 0 and finite (no NaN except the documented leading-window NaNs for `har_*`,
  `yang_zhang_n20`, `volume_zscore_*`); `parkinson_volatility` present under its existing name.
- **Cross-estimator sanity checks** (unit tests): on clean bars `garman_klass ≈` and `rogers_satchell ≈`
  `parkinson` in order of magnitude; `parkinson == log_range^2/(4 ln2)` exactly; `yang_zhang_n20` NaN for the
  first 19 rows then finite; `har_monthly` NaN for the first 21 rows; `market_pk` equals the recomputed
  cross-sectional mean on a sampled date.
- **Evidently drift** (extend `check_drift()`): train-vs-test drift on the new feature columns, `drift.html`
  artifact, same as the current gate.
- **Regenerate-on-raw-change + schema version + backward-compat:**
  - add a `schema_version` sidecar / header comment; bump on any column change.
  - the enriched writer is deterministic and re-runs whenever raw data changes (wired into the raw-ingestion
    quality flow already in CLAUDE.md); the raw-prices + processed-data quality tests must pass first.
  - `parkinson_volatility` column name is PRESERVED so every existing runner keeps reading the same key; new
    columns are additive. A reader that only knows `parkinson_volatility` is unaffected.
- **TDD when implemented:** each estimator column gets a test-vs-published-formula (independent recompute),
  each causal column gets a no-look-ahead test (value at t unchanged when rows >t are perturbed), and a
  real-data-sample smoke per market. This spec does not implement any of it in this pass.

## Consistency with the dirty-data audit
Estimators are computed on the CLEANED OHLC (post `widen_range` / `reconstruct_nonpositive` / `swap_high_low`
/ `backadjust_split` / `drop_naninf`), zero-range/zero-volume are FLAGGED not deleted, and split back-adjust
is applied to close-to-close / overnight terms only (it does not move the scale-invariant Parkinson/GK/RS
values — see `2026-08-31_etl_cleaning_spec.md` §Priority).
