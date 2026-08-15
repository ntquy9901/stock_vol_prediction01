# VN100 daily OHLCV crawl and data-quality report

Generated: 2026-08-16 01:11

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
| AAA | NEW | 3020 | 2014-12-04 | 2026-08-14 | 0 | 2014-12-04 | False | 1 | 2 | FLAG |
| ACB | overlap | 1483 | 2020-12-09 | 2026-08-14 | 0 | 2020-12-09 | False | 2 | 2 | FLAG |
| ANV | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 27 | FLAG |
| ASM | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 30 | FLAG |
| BCM | overlap | 1406 | 2021-03-15 | 2026-08-14 | 0 | 2021-03-15 | False | 0 | 1 | FLAG |
| BFC | NEW | 2801 | 2015-10-07 | 2026-08-14 | 0 | 2015-10-07 | False | 0 | 1 | FLAG |
| BID | overlap | 3244 | 2014-01-24 | 2026-08-14 | 0 | 2014-01-24 | False | 1 | 29 | FLAG |
| BMP | NEW | 3717 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 1 | 25 | FLAG |
| BSI | NEW | 3912 | 2011-07-22 | 2026-08-14 | 2 | 2011-07-26 | False | 1 | 27 | FLAG |
| BVH | overlap | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 32 | FLAG |
| BWE | NEW | 2355 | 2017-07-20 | 2026-08-14 | 5 | 2017-07-27 | False | 0 | 1 | FLAG |
| CII | NEW | 3726 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 1 | 33 | FLAG |
| CMG | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 31 | FLAG |
| CTD | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 17 | FLAG |
| CTG | overlap | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 16 | FLAG |
| CTR | NEW | 1168 | 2022-02-23 | 2026-08-14 | 0 | 2022-02-23 | False | 0 | 0 | PASS |
| CTS | NEW | 3029 | 2014-12-04 | 2026-08-14 | 0 | 2014-12-04 | False | 0 | 1 | FLAG |
| DBC | NEW | 1834 | 2019-07-31 | 2026-08-14 | 0 | 2019-07-31 | False | 0 | 1 | FLAG |
| DCM | NEW | 2937 | 2015-03-31 | 2026-08-14 | 0 | 2015-03-31 | False | 0 | 1 | FLAG |
| DGC | NEW | 1570 | 2020-07-28 | 2026-08-14 | 0 | 2020-07-28 | False | 0 | 1 | FLAG |
| DGW | NEW | 2850 | 2015-08-03 | 2026-08-14 | 0 | 2015-08-03 | False | 0 | 1 | FLAG |
| DIG | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 33 | FLAG |
| DPM | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 35 | FLAG |
| DSE | NEW | 551 | 2024-07-01 | 2026-08-14 | 0 | 2024-07-01 | True | 0 | 0 | PASS |
| DXG | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 33 | FLAG |
| DXS | NEW | 1281 | 2021-09-06 | 2026-08-14 | 0 | 2021-09-06 | False | 0 | 1 | FLAG |
| EIB | NEW | 4270 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 21 | FLAG |
| EVF | NEW | 1189 | 2022-01-12 | 2026-08-14 | 0 | 2022-01-12 | False | 0 | 0 | PASS |
| FPT | overlap | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 30 | FLAG |
| FRT | NEW | 2163 | 2018-04-26 | 2026-08-14 | 0 | 2018-04-26 | False | 0 | 1 | FLAG |
| FTS | NEW | 2489 | 2017-01-13 | 2026-08-14 | 0 | 2017-01-13 | False | 0 | 1 | FLAG |
| GAS | overlap | 3692 | 2012-05-21 | 2026-08-14 | 0 | 2012-05-21 | False | 1 | 24 | FLAG |
| GEE | NEW | 523 | 2024-08-14 | 2026-08-14 | 0 | 2024-08-14 | True | 0 | 0 | PASS |
| GEX | NEW | 2228 | 2018-01-25 | 2026-08-14 | 0 | 2018-01-25 | False | 0 | 1 | FLAG |
| GMD | NEW | 3717 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 1 | 32 | FLAG |
| GVR | overlap | 1665 | 2020-03-17 | 2026-08-14 | 0 | 2020-03-17 | False | 0 | 0 | PASS |
| HAG | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 33 | FLAG |
| HAH | NEW | 2960 | 2015-03-11 | 2026-08-14 | 0 | 2015-03-11 | False | 0 | 1 | FLAG |
| HCM | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 35 | FLAG |
| HDB | overlap | 2242 | 2018-01-05 | 2026-08-14 | 0 | 2018-01-05 | False | 0 | 0 | PASS |
| HDC | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 0 | 18 | FLAG |
| HDG | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 34 | FLAG |
| HHV | NEW | 1192 | 2022-01-20 | 2026-08-14 | 0 | 2022-01-20 | False | 0 | 0 | PASS |
| HPG | overlap | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 28 | FLAG |
| HSG | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 38 | FLAG |
| HT1 | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 36 | FLAG |
| HVN | NEW | 1886 | 2019-05-07 | 2026-08-14 | 0 | 2019-05-07 | False | 0 | 1 | FLAG |
| IJC | NEW | 4228 | 2010-04-19 | 2026-08-14 | 3 | 2010-04-22 | False | 1 | 32 | FLAG |
| IMP | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 28 | FLAG |
| KBC | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 39 | FLAG |
| KDC | NEW | 3717 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 1 | 29 | FLAG |
| KDH | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 31 | FLAG |
| KOS | NEW | 1832 | 2019-07-22 | 2026-08-14 | 0 | 2019-07-22 | False | 0 | 1 | FLAG |
| LPB | overlap | 1496 | 2020-11-09 | 2026-08-14 | 0 | 2020-11-09 | False | 0 | 1 | FLAG |
| MBB | overlap | 3836 | 2011-11-01 | 2026-08-14 | 0 | 2011-11-01 | False | 1 | 25 | FLAG |
| MSN | overlap | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 32 | FLAG |
| MWG | overlap | 3123 | 2014-07-14 | 2026-08-14 | 0 | 2014-07-14 | False | 0 | 29 | FLAG |
| NKG | NEW | 4043 | 2011-01-14 | 2026-08-14 | 0 | 2011-01-14 | False | 1 | 31 | FLAG |
| NLG | NEW | 3453 | 2013-04-08 | 2026-08-14 | 4 | 2013-04-12 | False | 1 | 23 | FLAG |
| NT2 | NEW | 2884 | 2015-06-12 | 2026-08-14 | 0 | 2015-06-12 | False | 0 | 1 | FLAG |
| NVL | overlap | 2500 | 2016-12-28 | 2026-08-14 | 1 | 2016-12-29 | False | 0 | 1 | FLAG |
| OCB | NEW | 1447 | 2021-01-28 | 2026-08-14 | 0 | 2021-01-28 | False | 0 | 1 | FLAG |
| PAN | NEW | 3662 | 2012-07-06 | 2026-08-14 | 0 | 2012-07-06 | False | 1 | 31 | FLAG |
| PC1 | NEW | 2530 | 2016-11-16 | 2026-08-14 | 0 | 2016-11-16 | False | 0 | 1 | FLAG |
| PDR | overlap | 4163 | 2010-07-30 | 2026-08-14 | 0 | 2010-07-30 | False | 1 | 34 | FLAG |
| PHR | NEW | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 31 | FLAG |
| PLX | overlap | 2416 | 2017-04-21 | 2026-08-14 | 0 | 2017-04-21 | False | 0 | 1 | FLAG |
| PNJ | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 32 | FLAG |
| POW | overlap | 1976 | 2019-01-14 | 2026-08-14 | 0 | 2019-01-14 | False | 0 | 1 | FLAG |
| PPC | NEW | 3717 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 1 | 32 | FLAG |
| PTB | NEW | 3908 | 2011-07-22 | 2026-08-14 | 3 | 2011-07-27 | False | 1 | 31 | FLAG |
| PVD | NEW | 3726 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 1 | 34 | FLAG |
| PVT | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 36 | FLAG |
| REE | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 34 | FLAG |
| SAB | overlap | 2507 | 2016-12-06 | 2026-08-14 | 3 | 2016-12-09 | False | 0 | 1 | FLAG |
| SBT | NEW | 4277 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 30 | FLAG |
| SCS | NEW | 2083 | 2018-08-03 | 2026-08-14 | 0 | 2018-08-03 | False | 0 | 0 | PASS |
| SHB | overlap | 3033 | 2014-12-04 | 2026-08-14 | 0 | 2014-12-04 | False | 0 | 1 | FLAG |
| SIP | NEW | 1280 | 2021-09-20 | 2026-08-14 | 0 | 2021-09-20 | False | 0 | 0 | PASS |
| SJS | NEW | 3726 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 1 | 31 | FLAG |
| SSB | overlap | 1408 | 2021-03-24 | 2026-08-14 | 3 | 2021-03-29 | False | 0 | 1 | FLAG |
| SSI | overlap | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 36 | FLAG |
| STB | overlap | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 34 | FLAG |
| SZC | NEW | 1966 | 2019-01-15 | 2026-08-14 | 0 | 2019-01-15 | False | 0 | 1 | FLAG |
| TCB | overlap | 2127 | 2018-06-04 | 2026-08-14 | 0 | 2018-06-04 | False | 0 | 1 | FLAG |
| TCH | NEW | 2559 | 2016-10-06 | 2026-08-14 | 5 | 2016-10-13 | False | 0 | 1 | FLAG |
| TLG | NEW | 4253 | 2010-03-26 | 2026-08-14 | 0 | 2010-03-26 | False | 0 | 13 | FLAG |
| TPB | overlap | 2168 | 2018-04-19 | 2026-08-14 | 0 | 2018-04-19 | False | 2 | 3 | FLAG |
| VCB | overlap | 4268 | 2010-02-22 | 2026-08-14 | 0 | 2010-02-22 | False | 1 | 30 | FLAG |
| VCG | NEW | 1469 | 2020-12-29 | 2026-08-14 | 0 | 2020-12-29 | False | 0 | 1 | FLAG |
| VCI | NEW | 2372 | 2017-07-07 | 2026-08-14 | 1 | 2017-07-10 | False | 0 | 1 | FLAG |
| VGC | NEW | 1870 | 2019-05-29 | 2026-08-14 | 0 | 2019-05-29 | False | 0 | 1 | FLAG |
| VHC | NEW | 4268 | 2010-02-22 | 2026-08-14 | 4 | 2010-02-26 | False | 1 | 35 | FLAG |
| VHM | overlap | 2148 | 2018-05-17 | 2026-08-14 | 4 | 2018-05-23 | False | 0 | 1 | FLAG |
| VIB | overlap | 1504 | 2020-11-10 | 2026-08-14 | 0 | 2020-11-10 | False | 0 | 0 | PASS |
| VIC | overlap | 3726 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 1 | 29 | FLAG |
| VIX | NEW | 1452 | 2021-01-08 | 2026-08-14 | 1 | 2021-01-11 | False | 0 | 1 | FLAG |
| VJC | overlap | 2462 | 2017-02-28 | 2026-08-14 | 3 | 2017-03-03 | False | 0 | 0 | PASS |
| VND | NEW | 2337 | 2017-08-18 | 2026-08-14 | 0 | 2017-08-18 | False | 0 | 1 | FLAG |
| VNM | overlap | 3717 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 1 | 21 | FLAG |
| VPB | overlap | 2334 | 2017-08-17 | 2026-08-14 | 0 | 2017-08-17 | False | 0 | 1 | FLAG |
| VPI | NEW | 2108 | 2018-06-29 | 2026-08-14 | 0 | 2018-06-29 | False | 0 | 1 | FLAG |
| VRE | overlap | 2277 | 2017-11-06 | 2026-08-14 | 6 | 2017-11-14 | False | 0 | 1 | FLAG |
| VSC | NEW | 3719 | 2012-04-03 | 2026-08-14 | 0 | 2012-04-03 | False | 1 | 26 | FLAG |

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

1. **high < low** (54 rows across 52 tickers): Parkinson-affecting, since Parkinson variance uses only high/low. These must be dropped or repaired before computing volatility. Magnitudes range from ~1% gaps to gross defects (e.g. ACB/TPB 2025-06-04..05 where the adjusted high collapses to a near-zero value). Table below.
2. **open/close outside [low, high]** (1620 rows): NOT Parkinson-affecting (high >= low still holds on these rows); they stem from dividend/split back-adjustment applied unevenly across O/H/L/C, the same artifact documented for the top-level files. Per-ticker counts are in the `oc_outside` column of the per-ticker table above.

### high < low rows (Parkinson-affecting)

| ticker | high<low_rows | example_dates |
|---|---|---|
| ACB | 2 | 2025-06-04, 2025-06-05 |
| TPB | 2 | 2025-06-04, 2025-06-05 |
| AAA | 1 | 2024-04-10 |
| ANV | 1 | 2014-06-05 |
| ASM | 1 | 2014-06-05 |
| BID | 1 | 2014-06-05 |
| BMP | 1 | 2014-06-05 |
| BSI | 1 | 2014-06-05 |
| BVH | 1 | 2014-06-05 |
| CII | 1 | 2014-06-05 |
| CTD | 1 | 2014-06-05 |
| CTG | 1 | 2014-06-05 |
| DIG | 1 | 2014-06-05 |
| DPM | 1 | 2014-06-05 |
| DXG | 1 | 2014-06-05 |
| EIB | 1 | 2014-06-05 |
| FPT | 1 | 2014-06-05 |
| GAS | 1 | 2014-06-05 |
| GMD | 1 | 2014-06-05 |
| HAG | 1 | 2014-06-05 |
| HCM | 1 | 2014-06-05 |
| HDG | 1 | 2014-06-05 |
| HPG | 1 | 2014-06-05 |
| HSG | 1 | 2014-06-05 |
| HT1 | 1 | 2014-06-05 |
| IJC | 1 | 2014-06-05 |
| IMP | 1 | 2014-06-05 |
| KBC | 1 | 2014-06-05 |
| KDC | 1 | 2014-06-05 |
| KDH | 1 | 2014-06-05 |
| MBB | 1 | 2014-06-05 |
| MSN | 1 | 2014-06-05 |
| NKG | 1 | 2014-06-05 |
| NLG | 1 | 2014-06-05 |
| PAN | 1 | 2014-06-05 |
| PDR | 1 | 2014-06-05 |
| PHR | 1 | 2014-06-05 |
| PNJ | 1 | 2014-06-05 |
| PPC | 1 | 2014-06-05 |
| PTB | 1 | 2014-06-05 |
| PVD | 1 | 2014-06-05 |
| PVT | 1 | 2014-06-05 |
| REE | 1 | 2014-06-05 |
| SBT | 1 | 2014-06-05 |
| SJS | 1 | 2014-06-05 |
| SSI | 1 | 2014-06-05 |
| STB | 1 | 2014-06-05 |
| VCB | 1 | 2014-06-05 |
| VHC | 1 | 2014-06-05 |
| VIC | 1 | 2014-06-05 |
| VNM | 1 | 2014-06-05 |
| VSC | 1 | 2014-06-05 |

## Leading synthetic backfill > 20 rows

None.

## Coverage

All 104 crawled tickers end on 2026-08-14.
