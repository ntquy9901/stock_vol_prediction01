# Raw Daily Price Data — Audit and Update Report

- Date: 2026-08-15
- Target directory: `C:\luanvan\stock_vol_prediction01\data\raw\prices`
- Reference "today": 2026-08-15 (Saturday; latest VN trading day available = Friday 2026-08-14)
- Scope note: `archive/` is out of scope and was not audited or written to.

---

## 1. Audit

### 1.1 Inventory

- 33 ticker files, format `<TICKER>_ohlcv.csv`, columns: `date,open,high,low,close,volume`.
- 1 provenance note: `LPB_SOURCE_NOTE.md`.
- No subdirectories under the target dir.

### 1.2 Per-ticker table

`n_rows` and dates read from each file. `behind` = calendar days from `last_date` to 2026-08-15.

| ticker | file | first_date | last_date | n_rows | behind (days) | notes |
|--------|------|-----------|-----------|-------:|-----:|-------|
| ACB | ACB_ohlcv.csv | 2006-11-21 | 2026-06-09 | 4868 | 67 | thousands VND, plain date |
| BCM | BCM_ohlcv.csv | 2018-02-21 | 2026-06-09 | 2065 | 67 | |
| BID | BID_ohlcv.csv | 2014-01-24 | 2026-06-09 | 3083 | 67 | |
| BVH | BVH_ohlcv.csv | 2009-06-25 | 2026-06-09 | 4232 | 67 | |
| CTG | CTG_ohlcv.csv | 2009-07-16 | 2026-06-09 | 4217 | 67 | |
| FPT | FPT_ohlcv.csv | 2006-12-13 | 2026-06-09 | 4854 | 67 | |
| GAS | GAS_ohlcv.csv | 2012-05-21 | 2026-06-09 | 3508 | 67 | |
| GVR | GVR_ohlcv.csv | 2018-03-21 | 2026-06-09 | 2046 | 67 | |
| HDB | HDB_ohlcv.csv | 2018-01-05 | 2026-06-09 | 2100 | 67 | |
| HPG | HPG_ohlcv.csv | 2007-11-15 | 2026-06-09 | 4625 | 67 | |
| LPB | LPB_ohlcv.csv | 2020-11-09 | 2026-08-10 | 1434 | 5 | SSI source, tz-aware, ~1.16x scale caveat (see note file) |
| MBB | MBB_ohlcv.csv | 2011-11-01 | 2026-06-09 | 3643 | 67 | |
| MSN | MSN_ohlcv.csv | 2009-11-05 | 2026-06-09 | 4138 | 67 | |
| MWG | MWG_ohlcv.csv | 2014-07-14 | 2026-06-09 | 2973 | 67 | |
| NVL | NVL_ohlcv.csv | 2016-12-28 | 2026-06-09 | 2356 | 67 | |
| PDR | PDR_ohlcv.csv | 2010-07-30 | 2026-06-09 | 3956 | 67 | |
| PLX | PLX_ohlcv.csv | 2017-04-21 | 2026-06-09 | 2281 | 67 | |
| POW | POW_ohlcv.csv | 2018-03-06 | 2026-06-09 | 2054 | 67 | |
| SAB | SAB_ohlcv.csv | 2016-12-06 | 2026-06-09 | 2372 | 67 | |
| SHB | SHB_ohlcv.csv | 2009-04-20 | 2026-06-09 | 4275 | 67 | |
| SSB | SSB_ohlcv.csv | 2021-03-24 | 2026-06-09 | 1299 | 67 | |
| SSI | SSI_ohlcv.csv | 2006-12-15 | 2026-06-09 | 4842 | 67 | |
| STB | STB_ohlcv.csv | 2006-10-27 | 2026-06-09 | 4887 | 67 | |
| TCB | TCB_ohlcv.csv | 2018-06-04 | 2026-06-09 | 2002 | 67 | |
| TPB | TPB_ohlcv.csv | 2018-04-19 | 2026-06-09 | 2031 | 67 | |
| VCB | VCB_ohlcv.csv | 2009-06-30 | 2026-06-09 | 4229 | 67 | |
| VHM | VHM_ohlcv.csv | 2011-11-10 | 2026-06-09 | 3426 | 67 | |
| VIB | VIB_ohlcv.csv | 2017-01-09 | 2026-06-09 | 2342 | 67 | |
| VIC | VIC_ohlcv.csv | 2007-09-19 | 2026-06-09 | 4666 | 67 | |
| VJC | VJC_ohlcv.csv | 2017-02-28 | 2026-06-09 | 2318 | 67 | |
| VNM | VNM_ohlcv.csv | 2006-10-27 | 2026-06-09 | 4887 | 67 | |
| VPB | VPB_ohlcv.csv | 2017-08-17 | 2026-06-19 | 2294 | 57 | full VND, tz-aware datetime |
| VRE | VRE_ohlcv.csv | 2017-11-06 | 2026-06-19 | 2237 | 57 | full VND, tz-aware datetime |

### 1.3 Anomalies / findings

- Dates: all files strictly monotonic increasing, no duplicate dates.
- Gaps: `>7`-day gaps flagged in most files scale with history length (e.g. ACB 21, SSI 21). These are the annual Tet holiday plus weekend clusters (Vietnam market closes ~9 consecutive days for Tet). Not data errors.
- Three distinct storage conventions coexist:
  - 31 tickers: price levels in thousands of VND, `date` as plain `YYYY-MM-DD`, last date 2026-06-09.
  - VPB, VRE: price levels in full VND (~1000x the others), `date` as tz-aware `YYYY-MM-DD 00:00:00+07:00`, last date 2026-06-19 (merged from `data/raw/vn30/` per git history).
  - LPB: SSI-sourced, tz-aware date, last date 2026-08-10; `LPB_SOURCE_NOTE.md` documents an ~1.16x price-level adjustment convention difference vs the other tickers.
- Cross-ticker level inconsistency (thousands vs full VND) is material for absolute-level joins but immaterial for the modeling target: Parkinson volatility uses `log(H/L)`, which is invariant to any per-day multiplicative scale.

### 1.4 Overall latest date and staleness

- Overall latest date present: 2026-08-10 (LPB only).
- Stale cohort: 31 tickers end 2026-06-09 (67 days behind); VPB, VRE end 2026-06-19 (57 days behind).
- LPB is current to within a few days (5 days behind); it was refreshed separately on 2026-08-10.
- Latest actual VN trading day as of the reference date: 2026-08-14 (2026-08-15 and 2026-08-16 are weekend).

---

## 2. Update mechanism

### 2.1 Crawler

- Script: `src/data/crawl_vietnam_stocks.py` (class `VietnamStockCrawler`).
- Source/API: Yahoo Finance via `yfinance`, symbols suffixed `.VN` (e.g. `ACB.VN`), `interval="1d"`, `auto_adjust` default (dividend/split back-adjusted).
- Output columns produced: `date,open,high,low,close,volume` (drops dividends/splits) — matches the audited files.
- Invocation: `python src/data/crawl_vietnam_stocks.py`. As written, `main()` writes to `data/raw/vn30/`, `data/raw/vn100/`, `data/raw/hnx/` (not directly `data/raw/prices/`). Git history shows `data/raw/vn30/` was later merged into `data/raw/prices/` ("Merge data/raw/vn30/ into prices/"), which explains VPB/VRE's full-VND / tz-aware convention.
- LPB was recovered separately from the SSI iBoard API (`iboard-api.ssi.com.vn/statistics/charts/history`, resolution=1D), documented in `data/raw/prices/LPB_SOURCE_NOTE.md`.

### 2.2 Environment feasibility

- Network: available. yfinance returns VN data through 2026-08-14 in this environment.
- System Python 3.14.6: `yfinance` 1.5.2, `requests` 2.34.2, `pandas` 3.0.3 installed. `vnstock` not installed.
- GPU venv `.venv_gpu_encode` (Python 3.10.11): `pandas` 2.3.3 only; `yfinance`, `requests`, `vnstock` not installed.
- Fetch was run with system Python (the venv lacks yfinance). No API key required for yfinance.
- Adjustment caveat: yfinance `auto_adjust` re-references the entire history to the latest dividend epoch on every fetch. A re-crawl today therefore returns historical levels shifted vs the stored files (e.g. ACB 2026-06-09 stored close 26.5 [thousands] vs yfinance 22751.3 VND; ratio ~1.165). The intraday H/L ratio matches exactly, so Parkinson volatility is unaffected; absolute levels are not directly appendable without rescaling.

---

## 3. Fetch performed (non-destructive, staging)

A fetch WAS performed. Originals in `data/raw/prices/` were not modified or deleted.

- Script: `/tmp/fetch_update.py` (ad hoc; uses `yfinance`, sequential per ticker, modest memory).
- Fetch window: `start=2026-05-01`, `end=2026-08-16`, `interval=1d`, for all 33 tickers currently in the raw dir.
- Result: all 33 tickers returned data; latest fetched trading day = 2026-08-14 for every ticker.

### 3.1 Output location

`data/raw/prices_update_2026-08-16/`
- `raw_fetch/<TICKER>_fetch.csv` — raw fetched slice (2026-05 onward), yfinance values divided by 1000, columns `date,open,high,low,close,volume`.
- `append_ready/<TICKER>_gap.csv` — rows strictly after each ticker's existing `last_date`, up to 2026-08-14, rescaled by the per-ticker overlap factor so levels are continuous with the existing file.
- `update_summary.csv`, `update_summary.json` — per-ticker: existing_last, fetched_last, n_overlap, scale_med, scale_cv, n_gap, status.

### 3.2 Scale-factor method

For each ticker the overlap ratio `existing_close / (fetched_close/1000)` was computed on shared dates (2026-05-01..existing_last). The median ratio rescales the gap rows into the existing convention. Ratio dispersion (`scale_cv`) near 0 confirms a single constant factor is valid (no dividend event inside the overlap window).

- 31 tickers: `scale_med` between 1.0 and ~2.08, `scale_cv` ~0 (clean). Non-1.0 factors (ACB 1.165, MBB 1.196, VHM 2.077, VJC 1.3, SAB 1.069, etc.) reflect dividend-adjustment drift since the June crawl.
- VPB, VRE: `scale_med` ~1000 / ~1035 — confirms these files are stored in full VND; the rescale maps the fetched gap back to full VND, preserving continuity.
- NVL: `scale_cv` 0.032 (small dispersion; a minor dividend/adjustment change within the overlap window) — gap rows still written but flagged as slightly less clean.
- LPB: only 4 gap rows (already current to 2026-08-10), `scale_med` ~1.0.

### 3.3 Before/after last_date

| cohort | before last_date | after (staged) last_date | gap trading rows staged |
|--------|------------------|--------------------------|-----:|
| 31 tickers | 2026-06-09 | 2026-08-14 | 48 |
| VPB, VRE | 2026-06-19 | 2026-08-14 | 40 |
| LPB | 2026-08-10 | 2026-08-14 | 4 |

Continuity spot-checks:
- ACB existing 2026-06-09 close 26.5 → staged 2026-06-10 open 26.5 (thousands scale preserved).
- VPB existing 2026-06-19 close 25900 → staged 2026-06-22 open 25900 (full-VND scale preserved).

### 3.4 Not applied to originals

The staged `append_ready` files are not concatenated into `data/raw/prices/`. Applying them is a deliberate follow-up decision because:
1. The originals mix scales/date formats; a downstream reader should confirm which convention it expects.
2. yfinance `auto_adjust` differs from the original adjustment epoch; the rescale corrects levels but assumes no dividend event inside each ticker's gap window (validated only via `scale_cv` on the pre-gap overlap, not inside the gap itself).
3. The project's modeling target is Parkinson volatility (scale-invariant), so an alternative is a full re-crawl + re-run of `process_data.py` rather than an append.

Recommended follow-up (not executed here): review `data/raw/prices_update_2026-08-16/update_summary.csv`, decide append vs full re-crawl, then re-run the Parkinson processing pipeline and the data-quality gate (`python scripts/quality_gate/run_quality_gate.py`).

---

## 4. Summary

- 33 tickers audited; clean, monotonic, no duplicates; gaps are Tet holidays only.
- Latest stored date 2026-08-10 (LPB); the other 32 are stale (2026-06-09, or 2026-06-19 for VPB/VRE) — 57–67 days behind.
- Crawler: `src/data/crawl_vietnam_stocks.py`, Yahoo Finance via yfinance; LPB via SSI iBoard API.
- Fetch feasible and performed with system Python; fresh data through 2026-08-14 for all 33 tickers, written non-destructively to `data/raw/prices_update_2026-08-16/` with per-ticker overlap rescaling. Originals untouched. Append into `data/raw/prices/` deferred as a follow-up decision.
