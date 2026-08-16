# VN100 vnstock OHLCV Crawl — Report

Date: 2026-08-16
Scope: crawl VN100 daily OHLCV from vnstock into staging, verify integrity, compare against the
existing Yahoo `vn100/` files, and recommend on adopting vnstock for VN100.

Staging output: `data/raw/prices/vn100_vnstock/` (Yahoo `data/raw/prices/vn100/` NOT modified).
Crawler: `src/data/crawl_vnstock.py` (reusable, resumable). Test: `tests/test_crawl_vnstock.py`
(10 tests, vnstock client mocked, no network).

---

## 1. Install and source

- Package: `vnstock` 4.0.2, installed into the system interpreter Python 3.14.6
  (`C:\Users\QUY\AppData\Local\Programs\Python\Python314`), NOT the GPU venv.
- First install attempt (`pip install vnstock`) failed: pip tried to build `numpy` from source and
  the Meson build errored on Python 3.14. Resolved by forcing wheels only:
  `pip install --prefer-binary --only-binary=:all: vnstock` → succeeded (pulled `vnstock-4.0.2`,
  `vnai-2.5.6`, `vnstock_ezchart-0.0.3`, `wordcloud`, etc.).
- Import verified: `import vnstock` works. The banner/promo text vnstock prints on import contains
  emoji that crash the Windows cp1252 console; running under `PYTHONIOENCODING=utf-8` avoids the
  cosmetic `UnicodeEncodeError`. It does not affect data.
- Source: primary `VCI`, fallback `KBS` (rotated by `fetch_ticker`). A live probe fetched FPT via
  `VCI` successfully (see §5), so the HTTP backend is reachable and not IP-locked.

## 2. Resilience / anti-IP-lock

- Crawl is SEQUENTIAL. Each ticker's CSV is written to disk immediately after a successful fetch
  (`crawl_universe`), so an interruption at ticker N keeps ticker 1..N-1.
- `skip_existing=True` (default): on (re-)invocation, any ticker whose output CSV already exists is
  skipped. Re-running resumes where a prior run stopped. Verified behaviorally: re-running
  `crawl_universe` over the populated staging dir reported `ok=104, skipped_existing=104,
  fetched_now=0` (no network calls made).
- Anti-IP-lock: `polite_sleep` between tickers (CLI uses 6s to stay under the vnstock Guest
  ~20 req/min cap), retry-with-exponential-backoff per source (`base_sleep * 2**attempt`), and
  source rotation VCI→KBS on failure.
- Playwright: NOT needed. vnstock's HTTP API (VCI) responded normally; no IP-lock was hit, so the
  Playwright fallback was never triggered and Playwright was not installed.
- Host LLM-API flake: did NOT recur in this session. The crawl completed and all verification ran
  end-to-end without interruption. (As noted in the task, Playwright could not have prevented a host
  flake anyway — it only addresses IP-lock.)

## 3. Per-ticker crawl outcome

- Universe: the 104 stems from `data/raw/prices/vn100/*_ohlcv.csv`.
- **104 / 104 crawled OK.** All 104 output CSVs are present in `data/raw/prices/vn100_vnstock/`,
  each with the schema `date,open,high,low,close,volume`, plain `YYYY-MM-DD` dates, OHLC in
  thousands-VND, through 2026-08-14. Total 357,956 data rows.
- History depth varies by listing date (e.g. FPT/VNM back to 2008-03-11; DSE from 2024-07-01), as
  expected — vnstock returns each ticker's full available adjusted history.

Note on scale: the task assumption "vnstock returns full VND → divide OHLC by 1000" does NOT apply
to the `VCI` source. VCI returns adjusted prices already in thousands-VND. Verified against Yahoo:
FPT 2026-08-14 = open 69.3 / high 69.6 / low 68.0 / close 68.3 / vol 7,123,800 in both sources
(identical). No division is applied by the crawler, and none is needed.

## 4. Verify results (`src/data/verify_raw_prices data/raw/prices/vn100_vnstock`)

- 104 tickers scanned.
- **`zero_vol_price_moved` (the suspicious feed-glitch check): 0 tickers, 0 rows** — as expected
  with the relative-tolerance fix. This is the metric the task flagged; it is clean.
- Leading zero-volume backfill run >20: 0 tickers.
- **HARD defects: 53 tickers** — all are OHLC-consistency / non-positivity violations (NOT the
  volume glitch). Breakdown by check (a ticker can hit more than one):
  - `low_gt_min_oc` (low > min(open,close)): 42 tickers, 109 rows — of which 47 rows fall on
    zero-volume no-trade sessions (a reference-price vs H/L/C mismatch on halted days) and 62 rows
    on traded days.
  - `high_lt_max_oc` (high < max(open,close)): 41 tickers, 122 rows.
  - `nonpositive` (an OHLC field ≤ 0): 5 tickers, 13 rows — 2 all-zero pre-listing backfill rows and
    11 partial-zero glitches (e.g. GEX 2016-08-12 close=0.0 with open/high/low populated and real
    volume 521,300).
- Magnitude of the traded-day OHLC violations (max relative gap per row, 121 rows total ≈ 0.034% of
  357,956 rows):
  - ≤0.5% (1-tick rounding of 2-decimal adjusted prices): 15 rows
  - 0.5–2%: 74 rows
  - 2–10%: 31 rows
  - >10% (real adjustment glitch): 1 row — HCM 2020-06-05 (open 4.14 while low 5.74; the open was
    not split/dividend-adjusted consistently with H/L/C on that date).

For contrast, the Yahoo `vn100/` files report **0 HARD defects** under the same verifier — Yahoo
enforces OHLC consistency (at the cost of the phantom rows and volume glitches documented below).

## 5. vnstock vs Yahoo comparison

Compared on the overlapping date window (so different history lengths do not skew counts). Sample:
VNM, GAS, FPT, HPG, VCB.

### 5a. Phantom holiday rows (Yahoo-only dates)

Yahoo carries rows on Vietnamese market-holiday dates that vnstock correctly omits. In the overlap
window:

| Ticker | Yahoo-only (phantom) rows | Yahoo-only rows that are flat (H=L=O=C) & vol=0 | vnstock-only rows |
|--------|---------------------------|-----------------------------------------------|-------------------|
| VNM    | 134 | 134 (100%) | 5 |
| GAS    | 141 | 141 (100%) | 5 |
| FPT    | 167 | 167 (100%) | 5 |
| HPG    | 167 | 167 (100%) | 5 |
| VCB    | 158 | 158 (100%) | 5 |

- 100% of Yahoo's phantom rows are flat (high=low=open=close) with volume=0 — Yahoo forward-fills
  the prior close across closed sessions. Confirmed exactly for the cited cases: VNM 2013-02-11..15
  (Tet) all = 22.312578 / vol 0; GAS 2013-02-11..15 (Tet) all = 19.913299 / vol 0. Other phantom
  dates are Reunification/Labour (04-30, 05-01), National Day (09-02 / 09-03), New Year (12-31,
  01-01) — i.e. Vietnamese public holidays, not trading days. vnstock omits all of them.
- The reverse ("vnstock-only") is small and legitimate: 5 dates per ticker that are REAL trading
  sessions Yahoo dropped (e.g. FPT 2017-02-02 vol 487,480; 2019-12-03..05; 2020-03-04 — all with
  positive volume and realistic OHLC). vnstock is the more complete calendar in both directions.

### 5b. Volume on dates where Yahoo reported volume = 0

On common dates where Yahoo has volume=0, vnstock supplies real reported volume on the large
majority — i.e. vnstock fixes Yahoo's volume-glitch days without needing `repair_volume.py`:

| Ticker | Yahoo vol=0 common dates | vnstock vol>0 (corrected) | vnstock vol=0 (genuine no-trade) |
|--------|--------------------------|---------------------------|----------------------------------|
| VNM    | 82 | 80 | 2 |
| GAS    | 60 | 58 | 2 |
| FPT    | 71 | 69 | 2 |
| HPG    | 70 | 68 | 2 |
| VCB    | 88 | 86 | 2 |

Example: VNM 2013-01-02 Yahoo vol=0 → vnstock 199,930; FPT 2010-05-25 Yahoo vol=0 → vnstock 247,320.
~97–98% of Yahoo's zero-volume days had genuine trading volume that vnstock captures.

## 6. Recommendation

Adopt vnstock (VCI source) as the VN100 price source, with one guarded caveat.

For:
- Correct trading calendar — omits ~134–167 phantom holiday rows per ticker that Yahoo invents
  (all flat, zero-volume), and additionally recovers ~5 real sessions per ticker Yahoo dropped.
- Trustworthy volume — recovers real reported volume on ~97–98% of Yahoo's volume=0 glitch days,
  which is the original motivation for switching (removes the need for `repair_volume.py`
  interpolation on VN100).
- `zero_vol_price_moved` = 0 (the suspicious price-moved-with-no-volume glitch is absent).
- Longer, adjusted history for many tickers (e.g. FPT/VNM from 2008).

Caveat (must handle before feeding models):
- vnstock introduces OHLC-consistency defects on ~121 traded-day rows (0.034% of rows) plus zero-vol
  reference-price mismatches and a handful of pre-listing / partial-zero glitches (13 nonpositive
  rows). Most are ≤2% one-tick artifacts, but one is a real 38% split-adjustment glitch
  (HCM 2020-06-05) and GEX 2016-08-12 has close=0. A small `clean_ohlc`-style pass is recommended:
  drop/clip nonpositive rows, and clamp low/high to `[min(o,c), max(o,c)]` on the sub-2% artifact
  rows, then re-run `verify_raw_prices` to confirm HARD defects → 0. This is the mirror of the
  cleanup Yahoo already bakes in.

Net: vnstock's calendar and volume are materially more accurate; its OHLC needs the same light
consistency-cleaning that any raw exchange feed needs. Recommended path: keep both, run
`clean_ohlc` on the vnstock staging, and promote to the primary VN100 source after the cleaned
files pass `verify_raw_prices` with 0 HARD defects.

## 7. Data-quality gate note

Pandera schema / Evidently drift: N/A for this staging crawl. The change adds a new raw staging
directory and a crawler/test; it does not touch `data/processed`, the feature pipeline, or any
training manifest. Column-level integrity was instead verified with `verify_raw_prices` (§4), which
is the schema/integrity check appropriate to raw price files. Pandera/Evidently should be run when
(and if) the cleaned vnstock files are promoted into the processed pipeline.

## 8. Commands run

- `pip install --prefer-binary --only-binary=:all: vnstock` → vnstock 4.0.2
- `python -m pytest tests/test_crawl_vnstock.py -v` → 10 passed
- `python -m src.data.verify_raw_prices data/raw/prices/vn100_vnstock` → 53 HARD, 0 zero_vol_price_moved
- `python -m src.data.verify_raw_prices data/raw/prices/vn100` → 0 HARD (Yahoo baseline)
- resumability re-run of `crawl_universe` → 104 skipped_existing, 0 fetched
- live `fetch_ticker('FPT', 2026-08-11..14)` via VCI → scale/values match Yahoo

## 9. Files

- `src/data/crawl_vnstock.py` — resumable sequential crawler (pre-existing; used as-is).
- `tests/test_crawl_vnstock.py` — 10 mocked unit/smoke tests (pre-existing; pass).
- `data/raw/prices/vn100_vnstock/*_ohlcv.csv` — 104 crawled files (untracked; not committed).
- This report: `docs/reports/2026-08-16_vn100_vnstock_crawl_report.md`.
