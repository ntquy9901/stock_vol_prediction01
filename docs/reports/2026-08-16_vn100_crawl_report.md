# VN100 daily OHLCV crawl and data-quality report

Generated: 2026-08-16 07:16

## Source and scope

VN100 constituent set = VN30 (large-cap) + VN70 (mid-cap). Source: HOSE VN100 index composition (2025 basket), per assistant knowledge. No authoritative machine-readable VN100 list existed in the repository (the hardcoded list in `src/data/crawl_vietnam_stocks.py` contains invalid/delisted tickers such as XYZ, VDM and duplicates, so it was not used).

Attempted: 105 tickers. Crawled successfully: 104. Failed to fetch: 1 (['BCG']).

Crawler: Yahoo Finance via yfinance, `.VN` suffix, `interval='1d'`, `start='2006-01-01'`, `auto_adjust=True` (split/dividend adjusted). Output matches the existing top-level convention: plain `YYYY-MM-DD` date, prices in thousands-VND (Yahoo full-VND / 1000), volume unchanged. Rows beyond 2026-08-14 (latest VN trading day) are excluded. Files under `data/raw/prices/vn100/`; the existing 33 top-level files were not modified.

## New vs overlap

Overlap with existing top-level universe (33): ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, NVL, PDR, PLX, POW, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE

New tickers not previously present (72): AAA, ANV, ASM, BCG, BFC, BMP, BSI, BWE, CII, CMG, CTD, CTR, CTS, DBC, DCM, DGC, DGW, DIG, DPM, DSE, DXG, DXS, EIB, EVF, FRT, FTS, GEE, GEX, GMD, HAG, HAH, HCM, HDC, HDG, HHV, HSG, HT1, HVN, IJC, IMP, KBC, KDC, KDH, KOS, NKG, NLG, NT2, OCB, PAN, PC1, PHR, PNJ, PPC, PTB, PVD, PVT, REE, SBT, SCS, SIP, SJS, SZC, TCH, TLG, VCG, VCI, VGC, VHC, VIX, VND, VPI, VSC

## Per-ticker outcome

| ticker | new/overlap | rows | first_date | last_date | leading_flat_run | first_real_date | short_history | high<low | oc_outside | PASS |
|---|---|---|---|---|---|---|---|---|---|---|
| AAA | NEW | 3020 | 2014-12-04 | 2026-08-14 | 0 | 2014-12-04 | False | 0 | 0 | PASS |
| ACB | overlap | 1483 | 2020-12-09 | 2026-08-14 | 0 | 2020-12-09 | False | 0 | 0 | PASS |
| ANV | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| ASM | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| BCM | overlap | 1406 | 2021-03-15 | 2026-08-14 | 0 | 2021-03-15 | False | 0 | 0 | PASS |
| BFC | NEW | 2801 | 2015-10-07 | 2026-08-14 | 0 | 2015-10-07 | False | 0 | 0 | PASS |
| BID | overlap | 3244 | 2014-01-24 | 2026-08-14 | 0 | 2014-01-24 | False | 0 | 0 | PASS |
| BMP | NEW | 3717 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 0 | 0 | PASS |
| BSI | NEW | 3912 | 2011-07-22 | 2026-08-14 | 2 | 2011-07-26 | False | 0 | 0 | PASS |
| BVH | overlap | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| BWE | NEW | 2355 | 2017-07-20 | 2026-08-14 | 5 | 2017-07-27 | False | 0 | 0 | PASS |
| CII | NEW | 3726 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 0 | 0 | PASS |
| CMG | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| CTD | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| CTG | overlap | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| CTR | NEW | 1168 | 2022-02-23 | 2026-08-14 | 0 | 2022-02-23 | False | 0 | 0 | PASS |
| CTS | NEW | 3029 | 2014-12-04 | 2026-08-14 | 0 | 2014-12-04 | False | 0 | 0 | PASS |
| DBC | NEW | 1834 | 2019-07-31 | 2026-08-14 | 0 | 2019-07-31 | False | 0 | 0 | PASS |
| DCM | NEW | 2937 | 2015-03-31 | 2026-08-14 | 0 | 2015-03-31 | False | 0 | 0 | PASS |
| DGC | NEW | 1570 | 2020-07-28 | 2026-08-14 | 0 | 2020-07-28 | False | 0 | 0 | PASS |
| DGW | NEW | 2850 | 2015-08-03 | 2026-08-14 | 0 | 2015-08-03 | False | 0 | 0 | PASS |
| DIG | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| DPM | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| DSE | NEW | 551 | 2024-07-01 | 2026-08-14 | 0 | 2024-07-01 | True | 0 | 0 | PASS |
| DXG | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| DXS | NEW | 1281 | 2021-09-06 | 2026-08-14 | 0 | 2021-09-06 | False | 0 | 0 | PASS |
| EIB | NEW | 4270 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| EVF | NEW | 1189 | 2022-01-12 | 2026-08-14 | 0 | 2022-01-12 | False | 0 | 0 | PASS |
| FPT | overlap | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| FRT | NEW | 2163 | 2018-04-26 | 2026-08-14 | 0 | 2018-04-26 | False | 0 | 0 | PASS |
| FTS | NEW | 2489 | 2017-01-13 | 2026-08-14 | 0 | 2017-01-13 | False | 0 | 0 | PASS |
| GAS | overlap | 3692 | 2012-05-21 | 2026-08-14 | 0 | 2012-05-21 | False | 0 | 0 | PASS |
| GEE | NEW | 523 | 2024-08-14 | 2026-08-14 | 0 | 2024-08-14 | True | 0 | 0 | PASS |
| GEX | NEW | 2228 | 2018-01-25 | 2026-08-14 | 0 | 2018-01-25 | False | 0 | 0 | PASS |
| GMD | NEW | 3717 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 0 | 0 | PASS |
| GVR | overlap | 1665 | 2020-03-17 | 2026-08-14 | 0 | 2020-03-17 | False | 0 | 0 | PASS |
| HAG | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| HAH | NEW | 2960 | 2015-03-11 | 2026-08-14 | 0 | 2015-03-11 | False | 0 | 0 | PASS |
| HCM | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| HDB | overlap | 2242 | 2018-01-05 | 2026-08-14 | 0 | 2018-01-05 | False | 0 | 0 | PASS |
| HDC | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| HDG | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| HHV | NEW | 1192 | 2022-01-20 | 2026-08-14 | 0 | 2022-01-20 | False | 0 | 0 | PASS |
| HPG | overlap | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| HSG | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| HT1 | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| HVN | NEW | 1886 | 2019-05-07 | 2026-08-14 | 0 | 2019-05-07 | False | 0 | 0 | PASS |
| IJC | NEW | 4228 | 2010-04-19 | 2026-08-14 | 3 | 2010-04-22 | False | 0 | 0 | PASS |
| IMP | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| KBC | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| KDC | NEW | 3717 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 0 | 0 | PASS |
| KDH | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| KOS | NEW | 1832 | 2019-07-22 | 2026-08-14 | 0 | 2019-07-22 | False | 0 | 0 | PASS |
| LPB | overlap | 1496 | 2020-11-09 | 2026-08-14 | 0 | 2020-11-09 | False | 0 | 0 | PASS |
| MBB | overlap | 3836 | 2011-11-01 | 2026-08-14 | 0 | 2011-11-01 | False | 0 | 0 | PASS |
| MSN | overlap | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| MWG | overlap | 3123 | 2014-07-14 | 2026-08-14 | 0 | 2014-07-14 | False | 0 | 0 | PASS |
| NKG | NEW | 4043 | 2011-01-14 | 2026-08-14 | 0 | 2011-01-14 | False | 0 | 0 | PASS |
| NLG | NEW | 3453 | 2013-04-08 | 2026-08-14 | 4 | 2013-04-12 | False | 0 | 0 | PASS |
| NT2 | NEW | 2884 | 2015-06-12 | 2026-08-14 | 0 | 2015-06-12 | False | 0 | 0 | PASS |
| NVL | overlap | 2500 | 2016-12-28 | 2026-08-14 | 1 | 2016-12-29 | False | 0 | 0 | PASS |
| OCB | NEW | 1447 | 2021-01-28 | 2026-08-14 | 0 | 2021-01-28 | False | 0 | 0 | PASS |
| PAN | NEW | 3662 | 2012-07-06 | 2026-08-14 | 0 | 2012-07-06 | False | 0 | 0 | PASS |
| PC1 | NEW | 2530 | 2016-11-16 | 2026-08-14 | 0 | 2016-11-16 | False | 0 | 0 | PASS |
| PDR | overlap | 4163 | 2010-07-30 | 2026-08-14 | 0 | 2010-07-30 | False | 0 | 0 | PASS |
| PHR | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| PLX | overlap | 2416 | 2017-04-21 | 2026-08-14 | 0 | 2017-04-21 | False | 0 | 0 | PASS |
| PNJ | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| POW | overlap | 1976 | 2019-01-14 | 2026-08-14 | 0 | 2019-01-14 | False | 0 | 0 | PASS |
| PPC | NEW | 3717 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 0 | 0 | PASS |
| PTB | NEW | 3908 | 2011-07-22 | 2026-08-14 | 3 | 2011-07-27 | False | 0 | 0 | PASS |
| PVD | NEW | 3726 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 0 | 0 | PASS |
| PVT | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| REE | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| SAB | overlap | 2507 | 2016-12-06 | 2026-08-14 | 3 | 2016-12-09 | False | 0 | 0 | PASS |
| SBT | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| SCS | NEW | 2083 | 2018-08-03 | 2026-08-14 | 0 | 2018-08-03 | False | 0 | 0 | PASS |
| SHB | overlap | 3033 | 2014-12-04 | 2026-08-14 | 0 | 2014-12-04 | False | 0 | 0 | PASS |
| SIP | NEW | 1280 | 2021-09-20 | 2026-08-14 | 0 | 2021-09-20 | False | 0 | 0 | PASS |
| SJS | NEW | 3726 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 0 | 0 | PASS |
| SSB | overlap | 1408 | 2021-03-24 | 2026-08-14 | 3 | 2021-03-29 | False | 0 | 0 | PASS |
| SSI | overlap | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| STB | overlap | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| SZC | NEW | 1966 | 2019-01-15 | 2026-08-14 | 0 | 2019-01-15 | False | 0 | 0 | PASS |
| TCB | overlap | 2127 | 2018-06-04 | 2026-08-14 | 0 | 2018-06-04 | False | 0 | 0 | PASS |
| TCH | NEW | 2559 | 2016-10-06 | 2026-08-14 | 5 | 2016-10-13 | False | 0 | 0 | PASS |
| TLG | NEW | 4253 | 2010-03-26 | 2026-08-14 | 0 | 2010-03-26 | False | 0 | 0 | PASS |
| TPB | overlap | 2168 | 2018-04-19 | 2026-08-14 | 0 | 2018-04-19 | False | 0 | 0 | PASS |
| VCB | overlap | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| VCG | NEW | 1469 | 2020-12-29 | 2026-08-14 | 0 | 2020-12-29 | False | 0 | 0 | PASS |
| VCI | NEW | 2372 | 2017-07-07 | 2026-08-14 | 1 | 2017-07-10 | False | 0 | 0 | PASS |
| VGC | NEW | 1870 | 2019-05-29 | 2026-08-14 | 0 | 2019-05-29 | False | 0 | 0 | PASS |
| VHC | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 0 | PASS |
| VHM | overlap | 2148 | 2018-05-17 | 2026-08-14 | 4 | 2018-05-23 | False | 0 | 0 | PASS |
| VIB | overlap | 1504 | 2020-11-10 | 2026-08-14 | 0 | 2020-11-10 | False | 0 | 0 | PASS |
| VIC | overlap | 3726 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 0 | 0 | PASS |
| VIX | NEW | 1452 | 2021-01-08 | 2026-08-14 | 1 | 2021-01-11 | False | 0 | 0 | PASS |
| VJC | overlap | 2462 | 2017-02-28 | 2026-08-14 | 3 | 2017-03-03 | False | 0 | 0 | PASS |
| VND | NEW | 2337 | 2017-08-18 | 2026-08-14 | 0 | 2017-08-18 | False | 0 | 0 | PASS |
| VNM | overlap | 3717 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 0 | 0 | PASS |
| VPB | overlap | 2334 | 2017-08-17 | 2026-08-14 | 0 | 2017-08-17 | False | 0 | 0 | PASS |
| VPI | NEW | 2108 | 2018-06-29 | 2026-08-14 | 0 | 2018-06-29 | False | 0 | 0 | PASS |
| VRE | overlap | 2277 | 2017-11-06 | 2026-08-14 | 6 | 2017-11-14 | False | 0 | 0 | PASS |
| VSC | NEW | 3719 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 0 | 0 | PASS |

## Fetch failures

BCG

## Short-history tickers (< 750 rows)

| ticker | rows | first_date |
|---|---|---|
| DSE | 551 | 2024-07-01 |
| GEE | 523 | 2024-08-14 |

## OHLC-consistency defects (Yahoo source glitches, flagged)

These are present in both adjusted and unadjusted Yahoo feeds, so they are genuine single-day source defects, not a processing error. Flagged via `xfail` for downstream cleaning; they do not block the test suite. Relative tolerance 1e-05.

Two categories:

1. **high < low** (0 rows across 0 tickers): Parkinson-affecting, since Parkinson variance uses only high/low. These must be dropped or repaired before computing volatility. Magnitudes range from ~1% gaps to gross defects (e.g. ACB/TPB 2025-06-04..05 where the adjusted high collapses to a near-zero value). Table below.
2. **open/close outside [low, high]** (0 rows): NOT Parkinson-affecting (high >= low still holds on these rows); they stem from dividend/split back-adjustment applied unevenly across O/H/L/C, the same artifact documented for the top-level files. Per-ticker counts are in the `oc_outside` column of the per-ticker table above.

### high < low rows (Parkinson-affecting)

None.

## Leading synthetic backfill > 20 rows

None.

## Coverage

All 104 crawled tickers end on 2026-08-14.
