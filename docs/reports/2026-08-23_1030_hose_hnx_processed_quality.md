# HOSE / HNX Processed Data Quality Audit — Parkinson Volatility

Generated: 2026-08-23
Scope: `data/processed/hose/*_processed.csv` (405 tickers) and
`data/processed/hnx/*_processed.csv` (299 tickers), schema
`date,parkinson_volatility`. Cross-checked against raw OHLCV in
`data/raw/prices/hose_vnstock/` and `data/raw/prices/hnx_vnstock/`
(schema `date,open,high,low,close,volume`). Read-only on `data/`.

Values are Parkinson **variance** (sigma^2), produced by the ETL as
`(ln(H/L)^2) / (4 ln 2)` then dropna then upper-clip at 0.1.

## Environment

- Interpreter used: repository `python` (Python **3.14.6**), which imports
  `pandera` **0.32.1** (matches CLAUDE.md). Probed alternatives: `py -3.11`
  (not installed), `.venv/Scripts/python.exe` (absent),
  `.venv_vnstock/Scripts/python.exe` and `.venv_gpu_encode/Scripts/python.exe`
  (no `pandera`). GPU not used; all work CPU-only.
- Pandera schema applied: `scripts/quality_gate/data_schemas.py`
  `_check_processed_file` (schema validation + high>=low where OHLCV present +
  parseable/strictly-increasing/unique dates + no all-NaN columns).
- Invariant checks replicate `tests/test_processed_data_quality.py` (which is
  hard-wired to the 33-ticker VN30 `data/processed` dir) programmatically
  against the hose/hnx dirs, without editing the test.
- Evidently drift: N/A for this audit. Evidently `check_drift()` is a
  train-vs-test feature-drift report on the modelling panel, not a per-file
  integrity check of raw processed series; no train/test split was constructed
  here.

## STEP 2 — Pandera schema (repo quality gate)

| Exchange | Files | PASS | FAIL | SKIP/other |
|---|---|---|---|---|
| HOSE | 405 | 405 | 0 | 0 |
| HNX  | 299 | 299 | 0 | 0 |

All 704 processed files satisfy the repo Pandera processed schema:
`parkinson_volatility` present, float, `>= 0`, non-null; `date` present, string,
parseable, strictly increasing, no duplicates; no all-NaN columns. No failures.

Raw-vs-processed ticker parity: the only raw file without a processed
counterpart in each exchange is `_manifest.csv` (metadata, not a ticker). Every
raw ticker has a processed series.

## STEP 3 — Invariant checks

### Value invariants (finite, non-negative, bounded)

- No NaN/inf, no negative values in any of the 704 files.
- No value exceeds the audit suspicion bound (> 0.1); the ETL upper-clip is 0.1,
  so any raw H/L range large enough to exceed 0.1 variance is already capped.
  Rows sitting exactly at the 0.1 ceiling: HOSE 2 rows (0.0001% of 1,381,390),
  HNX 24 rows (0.0023% of 1,065,741) — negligible.
- Global value distribution (variance):
  - HOSE: median 2.02e-04, mean 5.27e-04, max 0.100000.
  - HNX: median 3.77e-05, mean 6.93e-04, max 0.100000.
- No zero-variance (degenerate std==0) whole-series tickers in either exchange.

### Date invariants

- All dates parseable (`%Y-%m-%d`), unique, strictly monotonic increasing, and
  weekday-only across all 704 files. No weekend dates, no duplicates, no
  backward steps.

### Coverage

| Exchange | Tickers | rows min / median / max | Earliest first-date | Latest last-date |
|---|---|---|---|---|
| HOSE | 405 | 4 / 3892 / 5805 | 2003-05-20 | 2026-08-21 |
| HNX  | 299 | 298 / 3932 / 5109 | 2005-12-28 | 2026-08-21 |

Panel max date (both exchanges): **2026-08-21**.

### ETL math cross-check (processed variance vs raw OHLC)

For sampled tickers per exchange (first five alphabetically plus mid/last),
`(ln(high/low)^2)/(4 ln 2)`, upper-clipped at 0.1, was recomputed from raw OHLC
and compared to the processed column on the overlapping (positive-price) dates:

- HOSE sample (AAA, AAM, AAN, AAT, ABR, MBB, YEG): all `ok`, max abs diff
  ~1e-16 (floating-point epsilon), 62–4215 overlapping rows each.
- HNX sample (ADC, ALT, AMC, AME, AMV, NVB, X20): all `ok`, max abs diff
  ~1e-16, 2065–4136 overlapping rows each.

The ETL formula is reproduced exactly (differences at machine precision).

## Flagged tickers (modelling-universe concerns, not integrity failures)

### Short history (< 250 rows)

- HOSE (9): LPS (4), DMX (12), AAN (62), GEL (133), HPA (133), CRV (155),
  VCK (169), VPX (172), TCX (209) — all recent listings (first-date late
  2025/2026), still current (last-date 2026-08-21).
- HNX (0): none; HNX minimum is 298 rows.

### Stale last-date (ended > 30 days before panel max 2026-08-21)

- HOSE (3): LGC (last 2026-04-29, 114d), TTE (2026-07-01, 51d),
  BTT (2026-07-21, 31d).
- HNX (19): LCD (2025-02-21, 546d), VE4 (2025-11-04, 290d), GMA (2025-12-24,
  240d), HCT (2026-03-30, 144d), MED (2026-05-13, 100d), BED (2026-05-26, 87d),
  PEN (2026-06-02, 80d), PIA (2026-06-05, 77d), KMT (2026-06-15, 67d),
  NHC (2026-06-25, 57d), ECI (2026-06-26, 56d), HEV (2026-06-30, 52d),
  NBW (2026-07-02, 50d), ARM (2026-07-06, 46d), SDC (2026-07-09, 43d),
  CJC (2026-07-10, 42d), QHD/VNT (2026-07-13, 39d), PTD (2026-07-21, 31d).

Stale series indicate delisting/halt/thin trading; they end before the OOS test
window and would contribute no recent target.

### High zero-variance fraction (H==L days: limit moves / thin trading)

Parkinson variance is exactly 0 whenever raw high == low (no intraday range).
This is arithmetically correct but produces a degenerate target on those days.

- Overall zero-row fraction: HOSE 14.82%, HNX 45.30%.
- Per-ticker zero-fraction quartiles [25/50/75/90%]:
  HOSE [0.01, 0.06, 0.20, 0.41]; HNX [0.19, 0.45, 0.67, 0.83].
- Tickers with > 50% zero-variance rows: **HOSE 18/405**, **HNX 137/299**.
  Worst HNX: LCD 0.98, QST 0.96, PTD 0.95, TET 0.95, BED 0.93, THS 0.93,
  PTX 0.93. Worst HOSE: TTE 0.78, TDW 0.74, CLW 0.71, ADP 0.71, DTT 0.70.

For roughly half the HNX universe the target is zero on the majority of days
(illiquid names). Volatility forecasting and any directional-accuracy metric are
near-meaningless on such series.

## Verdict

**GO — with an exclusion filter.** All 704 processed files pass the repo Pandera
schema and every structural invariant (finite/non-negative/bounded values;
parseable, unique, strictly-increasing, weekday-only dates), and the Parkinson
ETL math is reproduced to machine precision against raw OHLC. The data is
integrity-clean and can be trusted for an out-of-sample run.

The flagged items are fitness-for-modelling issues, not corruption. Recommended
universe filter before the OOS run:

1. **Exclude short-history** (< 250 rows): HOSE LPS, DMX, AAN, GEL, HPA, CRV,
   VCK, VPX, TCX. Insufficient in-sample history for a stable model/scaler.
2. **Exclude stale last-date** (> 30d before 2026-08-21): HOSE LGC, TTE, BTT;
   HNX LCD, VE4, GMA, HCT, MED, BED, PEN, PIA, KMT, NHC, ECI, HEV, NBW, ARM,
   SDC, CJC, QHD, VNT, PTD. No recent target for the test window.
3. **Down-weight or exclude high zero-variance** (> 50% zero rows): HOSE 18
   tickers, HNX 137 tickers. If retained, report metrics separately — a target
   that is zero on 50–98% of days makes RMSE/DirAcc uninformative. HNX in
   particular loses ~46% of its universe under this threshold; consider a
   liquidity screen (e.g. zero-fraction <= 25–30%) rather than the full 299.

No ticker requires exclusion for data-integrity reasons.
