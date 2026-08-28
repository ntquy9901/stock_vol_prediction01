# HOSE / HNX daily OHLCV — raw review, ETL, and processed review

Generated: 2026-08-22 15:30

Scope: newly crawled daily OHLCV for two Vietnamese exchanges (gitignored raw
inputs). Three stages: (1) raw review, (2) ETL raw -> Parkinson-variance
processed, (3) processed review. Work is CPU/pandas only.

Inputs (read-only):
- `data/raw/prices/hose_vnstock/<TICKER>_ohlcv.csv` — 405 tickers + `_manifest.csv`
- `data/raw/prices/hnx_vnstock/<TICKER>_ohlcv.csv` — 299 tickers + `_manifest.csv`
- Schema `date,open,high,low,close,volume`, date `YYYY-MM-DD`.

Outputs (written this session):
- `data/processed/hose/<TICKER>_processed.csv` — 405 files, schema
  `date,parkinson_volatility` (matches `data/processed/*_processed.csv` vn30 format).
- `data/processed/hnx/<TICKER>_processed.csv` — 299 files, same schema.
- Wrapper: `scripts/etl_hose_hnx.py` (reuses the Parkinson formula from
  `src/common/parkinson_utils.calculate_parkinson_volatility`; does not re-derive math).

Crawl completeness: HOSE 405/405 (manifest present), HNX 299/299 (manifest
present). Both crawls complete at review time.

---

## Stage 1 — Raw review (diagnostics, read-only)

Every ticker CSV was scanned. Counts are tickers-affected and rows-affected.
Prices are in thousands-VND; date parsing strips any tz time part via
`split(' ')[0]`. Parkinson variance downstream uses only high/low.

### HOSE (405 tickers)

| Issue | Tickers | Rows | Example tickers (worst first) |
|---|---|---|---|
| Unparseable dates | 0 | 0 | — |
| Duplicate dates | 0 | 0 | — |
| Non-monotonic dates | 0 | 0 | — |
| Weekend dates | 0 | 0 | — |
| NaN in OHLCV | 3 | 23 | KHP, VSH, VCA |
| Inf in OHLCV | 0 | 0 | — |
| Non-positive price (any of O/H/L/C <=0 or NaN) | 23 | 107 | ADP, KHP, VSH, VCA, SGR, NT2 |
| high < low | 3 | 3 | AFX, SGN, VCA |
| open/close outside [low, high] | 233 | 3541 | ADP, ABT, ACL, VCA, GHC, ABR |
| Negative volume | 0 | 0 | — |
| Leading flat run (high==low prefix = backfill signature) | 214 | 6765 | VHM, SAM, REE, HAP, TMS, LAF |
| Zero-volume rows (scattered) | 348 | 74120 | DTT, ADP, TDW, CLW, TIX, COM |
| All-zero-volume series | 0 | 0 | — |

- History length (rows): min 4, median 3896, max 6345. First dates span
  2000-07-28 .. 2026-08-18; last dates: 378 end 2026-08-21, remainder earlier.
- Largest leading-backfill prefixes: VHM 710 rows (first real 2014-09-22),
  SAM 541, REE 540, HAP 533, TMS 532, LAF 481. 66 tickers have a leading flat
  run > 5 rows.

### HNX (299 tickers)

| Issue | Tickers | Rows | Example tickers (worst first) |
|---|---|---|---|
| Unparseable dates | 0 | 0 | — |
| Duplicate dates | 0 | 0 | — |
| Non-monotonic dates | 0 | 0 | — |
| Weekend dates | 1 | 1 | BBS (2006-04-01, Saturday) |
| NaN in OHLCV | 2 | 2 | PTD, VNR |
| Inf in OHLCV | 0 | 0 | — |
| Non-positive price (any of O/H/L/C <=0 or NaN) | 16 | 176 | NAP, NBW, DTG, BTW, MAS, GDW |
| high < low | 1 | 3 | PJC |
| open/close outside [low, high] | 189 | 3441 | PTX, DTK, SPC, L40, KSV, MAS |
| Negative volume | 0 | 0 | — |
| Leading flat run (backfill signature) | 70 | 3709 | DTK, L40, NAP, PPY, PTX, KSV |
| Zero-volume rows (scattered) | 281 | 287238 | QST, TET, BED, LCD, SDC, CJC |
| All-zero-volume series | 0 | 0 | — |

- History length (rows): min 298, median 3941, max 5110. First dates span
  2005-12-28 .. 2025-06-16; 206 tickers end 2026-08-21, remainder earlier.
- Largest leading-backfill prefixes: DTK 829 rows (first real 2020-04-14),
  L40 730, NAP 673, PPY 291, PTX 264, KSV 249. 26 tickers have leading flat
  run > 5 rows.

### Characterisation of the issue classes

- `open/close outside [low, high]` is the largest category by count on both
  exchanges. Inspection (e.g. HOSE ADP, HNX PTX) shows these are raw-source
  inconsistencies — rows where one of the four prices is 0 while others are
  positive, or uneven back-adjustment across O/H/L/C. **These rows do not affect
  Parkinson variance**, which uses only high/low, and high >= low holds on almost
  all of them. They are therefore left untouched in ETL (see Stage 2) except
  where high/low itself is invalid.
- `high < low` is rare (HOSE 3 rows, HNX 3 rows) and genuinely inverts the
  Parkinson inputs; corrected in ETL.
- Non-positive prices concentrate in old illiquid rows (e.g. a 0.00 high on a
  no-trade day) and in the KHP/VSH/VCA NaN-open rows.
- Leading flat runs are the synthetic-backfill / pre-listing signature (H==L →
  Parkinson variance 0); trimmed in ETL to each ticker's first real trading day.

---

## Stage 2 — ETL (raw -> processed Parkinson variance)

Pipeline: `scripts/etl_hose_hnx.py etl`, reusing
`src.common.parkinson_utils.calculate_parkinson_volatility`
(σ² = ln(H/L)²/(4 ln2)), then inf->NaN, dropna, clip upper 0.1 — identical
post-processing to the delivered vn30/vn100 pipeline. Output schema
`date,parkinson_volatility`.

Cleaning rules applied, in order (per CLAUDE.md; no invented data, no silent
degradation):
1. Drop rows with unparseable dates (none present).
2. Drop exact-duplicate dates, keep last (none present).
3. Sort ascending; enforce weekday-only (drop weekend rows).
4. Fix **only Parkinson-invalid rows** (high<=0, low<=0, NaN high/low, or
   high<low) by setting high = max, low = min over the **positive** prices of
   that row. Rows where high >= low > 0 are left untouched even if open/close
   fall outside [low, high] (matches vn30 processing; Parkinson uses only H/L).
   A row with no positive price at all is dropped (cannot be recovered).
5. Trim the leading contiguous flat run (high==low prefix) to the first real
   trading day.

### Corrections applied

| Correction | HOSE tickers | HOSE rows | HNX tickers | HNX rows |
|---|---|---|---|---|
| Dropped duplicate dates | 0 | 0 | 0 | 0 |
| Dropped weekend rows | 0 | 0 | 1 | 1 |
| Dropped rows with no positive price | 3 | 38 | 3 | 24 |
| OHLC high/low corrected (invalid rows) | 13 | 25 | 10 | 140 |
| Leading backfill trimmed | 215 | 6753 | 70 | 3685 |
| Clipped to ceiling 0.1 | 2 | 2 | 24 | 24 |
| Dropped NaN/inf Parkinson | 0 | 0 | 0 | 0 |

- Largest trims (rows, new first date): HOSE VHM 710 (2014-09-22), SAM 541
  (2003-05-21), REE 540 (2003-05-20); HNX NAP 673 (2019-04-02), DTK 485
  (2018-11-22), PPY 291 (2017-04-28).
- Largest OHLC corrections concentrate on a few illiquid tickers (HNX PTX,
  DTK, SPC; HOSE VCA, SGN). All corrections use max/min over that row's positive
  prices only.
- No ticker produced an empty output. Output row totals: HOSE 1,381,390 rows
  across 405 tickers; HNX 1,065,741 rows across 299 tickers.

---

## Stage 3 — Processed review

`tests/test_processed_data_quality.py` hard-wires `DATA_DIR = data/processed`
and asserts an exact 33-ticker (vn30) universe, so it cannot be pointed at the
new dirs without editing it. An equivalent check was run instead, asserting the
same invariants the test enforces (checks 1-3 + dates), plus the repo Pandera
schema `scripts/quality_gate/data_schemas._check_processed_file` (reused
directly, not reimplemented).

Invariants checked per file:
- columns exactly `date,parkinson_volatility`;
- parkinson_volatility finite, >= 0, <= 0.1 (clip ceiling);
- dates parseable, unique, strictly increasing (monotonic), weekday-only;
- Pandera schema (`PROCESSED_SCHEMA`): dtype, ge(0), monotonic, no dup dates,
  no all-NaN columns.

### Results

| Exchange | Tickers | Rows | Invariant checks | Pandera schema | Date coverage |
|---|---|---|---|---|---|
| HOSE | 405 | 1,381,390 | PASS (0 fails) | PASS (0 fails) | 2003-05-20 .. 2026-08-21 |
| HNX | 299 | 1,065,741 | PASS (0 fails) | PASS (0 fails) | 2005-12-28 .. 2026-08-21 |

The high >= low invariant is resolved upstream in ETL (the processed output
carries only `parkinson_volatility >= 0`, which the invariant and Pandera checks
confirm on every file). Evidently drift was not run: it compares a train-vs-test
feature split, which is not defined for a raw per-ticker Parkinson series at the
ETL stage (no feature/manifest change), so it is `N/A` for this data step.

---

## Items flagged for human decision

These are surfaced, not auto-resolved:

- **Very short histories (HOSE).** 9 tickers have < 250 processed rows, dominated
  by two that are too short for volatility modelling: `LPS` (4 rows), `DMX`
  (12 rows). Others: AAN (62), GEL (133), HPA (133), CRV (155), VCK (169),
  VPX (172), TCX (209). HNX has none < 250 (min 298, `TD6`). Recommendation:
  exclude LPS and DMX from any modelling universe; the rest are newly-listed but
  usable with caution.
- **Stale last dates (possible delisting/suspension).** HOSE: 27 tickers end
  before the 2026-08-21 mode; oldest LGC (2026-04-29), TTE (2026-07-01), BTT
  (2026-07-21). HNX: 93 tickers end before mode; oldest LCD (2025-02-21), VE4
  (2025-11-04), GMA (2025-12-24). These are candidates for delisted/suspended
  status and should be reviewed before inclusion in a live-universe backtest.
- **OHLC-invalid concentration.** A small set of illiquid tickers carries most
  of the corrected rows (HNX PTX, DTK, SPC; HOSE VCA). Their processed series are
  valid but thin; treat as low-liquidity.

---

## Reproduction

```
python scripts/etl_hose_hnx.py scan --raw data/raw/prices/hose_vnstock --out _tmp_etl/hose_scan.json
python scripts/etl_hose_hnx.py etl  --raw data/raw/prices/hose_vnstock --processed data/processed/hose --out _tmp_etl/hose_etl.json
python scripts/etl_hose_hnx.py scan --raw data/raw/prices/hnx_vnstock  --out _tmp_etl/hnx_scan.json
python scripts/etl_hose_hnx.py etl  --raw data/raw/prices/hnx_vnstock  --processed data/processed/hnx --out _tmp_etl/hnx_etl.json
```

Stage-3 verification reused `scripts/quality_gate/data_schemas._check_processed_file`
plus the invariant assertions above. No raw file was modified; no GPU was used.
