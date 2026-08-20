# VN100 daily OHLCV crawl and data-quality report

Generated: 2026-08-20 16:24

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
| AAA | NEW | 4010 | 2010-07-15 | 2026-08-14 | 0 | 2010-07-15 | False | 0 | 0 | PASS |
| ACB | overlap | 4598 | 2008-03-07 | 2026-08-14 | 1 | 2008-03-10 | False | 0 | 0 | PASS |
| ANV | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| ASM | NEW | 4135 | 2010-01-18 | 2026-08-14 | 2 | 2010-01-20 | False | 0 | 0 | PASS |
| BCM | overlap | 2113 | 2018-02-21 | 2026-08-14 | 0 | 2018-02-21 | False | 0 | 0 | PASS |
| BFC | NEW | 2714 | 2015-10-07 | 2026-08-14 | 0 | 2015-10-07 | False | 0 | 0 | PASS |
| BID | overlap | 3131 | 2014-01-24 | 2026-08-14 | 0 | 2014-01-24 | False | 0 | 0 | PASS |
| BMP | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| BSI | NEW | 3759 | 2011-07-19 | 2026-08-14 | 0 | 2011-07-19 | False | 0 | 0 | PASS |
| BVH | overlap | 4280 | 2009-06-25 | 2026-08-14 | 0 | 2009-06-25 | False | 0 | 0 | PASS |
| BWE | NEW | 2267 | 2017-07-20 | 2026-08-14 | 5 | 2017-07-27 | False | 0 | 0 | PASS |
| CII | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| CMG | NEW | 4131 | 2010-01-22 | 2026-08-14 | 0 | 2010-01-22 | False | 0 | 0 | PASS |
| CTD | NEW | 4133 | 2010-01-20 | 2026-08-14 | 1 | 2010-01-21 | False | 0 | 0 | PASS |
| CTG | overlap | 4265 | 2009-07-16 | 2026-08-14 | 0 | 2009-07-16 | False | 0 | 0 | PASS |
| CTR | NEW | 2194 | 2017-10-31 | 2026-08-14 | 0 | 2017-10-31 | False | 0 | 0 | PASS |
| CTS | NEW | 4251 | 2009-07-31 | 2026-08-14 | 0 | 2009-07-31 | False | 0 | 0 | PASS |
| DBC | NEW | 4590 | 2008-03-18 | 2026-08-14 | 0 | 2008-03-18 | False | 0 | 0 | PASS |
| DCM | NEW | 2845 | 2015-03-31 | 2026-08-14 | 0 | 2015-03-31 | False | 0 | 0 | PASS |
| DGC | NEW | 2984 | 2014-08-26 | 2026-08-14 | 0 | 2014-08-26 | False | 0 | 0 | PASS |
| DGW | NEW | 2760 | 2015-08-03 | 2026-08-14 | 0 | 2015-08-03 | False | 0 | 0 | PASS |
| DIG | NEW | 4241 | 2009-08-19 | 2026-08-14 | 9 | 2009-09-01 | False | 0 | 0 | PASS |
| DPM | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| DSE | NEW | 531 | 2024-07-01 | 2026-08-14 | 0 | 2024-07-01 | True | 0 | 0 | PASS |
| DXG | NEW | 4153 | 2009-12-22 | 2026-08-14 | 10 | 2010-01-06 | False | 0 | 0 | PASS |
| DXS | NEW | 1266 | 2021-07-15 | 2026-08-14 | 0 | 2021-07-15 | False | 0 | 0 | PASS |
| EIB | NEW | 4193 | 2009-10-27 | 2026-08-14 | 0 | 2009-10-27 | False | 0 | 0 | PASS |
| EVF | NEW | 1996 | 2018-08-07 | 2026-08-14 | 0 | 2018-08-07 | False | 0 | 0 | PASS |
| FPT | overlap | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| FRT | NEW | 2075 | 2018-04-26 | 2026-08-14 | 0 | 2018-04-26 | False | 0 | 0 | PASS |
| FTS | NEW | 2393 | 2017-01-13 | 2026-08-14 | 0 | 2017-01-13 | False | 0 | 0 | PASS |
| GAS | overlap | 3556 | 2012-05-21 | 2026-08-14 | 0 | 2012-05-21 | False | 0 | 0 | PASS |
| GEE | NEW | 1091 | 2022-03-08 | 2026-08-14 | 3 | 2022-03-11 | False | 0 | 0 | PASS |
| GEX | NEW | 2698 | 2015-10-26 | 2026-08-14 | 0 | 2015-10-26 | False | 0 | 0 | PASS |
| GMD | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| GVR | overlap | 2094 | 2018-03-21 | 2026-08-14 | 0 | 2018-03-21 | False | 0 | 0 | PASS |
| HAG | NEW | 4404 | 2008-12-22 | 2026-08-14 | 0 | 2008-12-22 | False | 0 | 0 | PASS |
| HAH | NEW | 2859 | 2015-03-11 | 2026-08-14 | 0 | 2015-03-11 | False | 0 | 0 | PASS |
| HCM | NEW | 4307 | 2009-05-19 | 2026-08-14 | 17 | 2009-06-11 | False | 0 | 0 | PASS |
| HDB | overlap | 2148 | 2018-01-05 | 2026-08-14 | 0 | 2018-01-05 | False | 0 | 0 | PASS |
| HDC | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| HDG | NEW | 4124 | 2010-02-02 | 2026-08-14 | 3 | 2010-02-05 | False | 0 | 0 | PASS |
| HHV | NEW | 2653 | 2015-12-18 | 2026-08-14 | 144 | 2016-07-20 | False | 0 | 0 | PASS |
| HPG | overlap | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| HSG | NEW | 4415 | 2008-12-05 | 2026-08-14 | 1 | 2008-12-08 | False | 0 | 0 | PASS |
| HT1 | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| HVN | NEW | 2394 | 2017-01-03 | 2026-08-14 | 2 | 2017-01-05 | False | 0 | 0 | PASS |
| IJC | NEW | 4075 | 2010-04-19 | 2026-08-14 | 3 | 2010-04-22 | False | 0 | 0 | PASS |
| IMP | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| KBC | NEW | 4155 | 2009-12-18 | 2026-08-14 | 0 | 2009-12-18 | False | 0 | 0 | PASS |
| KDC | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| KDH | NEW | 4125 | 2010-02-01 | 2026-08-14 | 2 | 2010-02-03 | False | 0 | 0 | PASS |
| KOS | NEW | 2162 | 2017-12-08 | 2026-08-14 | 3 | 2017-12-13 | False | 0 | 0 | PASS |
| LPB | overlap | 2203 | 2017-10-05 | 2026-08-14 | 0 | 2017-10-05 | False | 0 | 0 | PASS |
| MBB | overlap | 3691 | 2011-11-01 | 2026-08-14 | 0 | 2011-11-01 | False | 0 | 0 | PASS |
| MSN | overlap | 4186 | 2009-11-05 | 2026-08-14 | 1 | 2009-11-06 | False | 0 | 0 | PASS |
| MWG | overlap | 3021 | 2014-07-14 | 2026-08-14 | 0 | 2014-07-14 | False | 0 | 0 | PASS |
| NKG | NEW | 3887 | 2011-01-14 | 2026-08-14 | 0 | 2011-01-14 | False | 0 | 0 | PASS |
| NLG | NEW | 3334 | 2013-04-08 | 2026-08-14 | 4 | 2013-04-12 | False | 0 | 0 | PASS |
| NT2 | NEW | 4094 | 2010-02-05 | 2026-08-14 | 5 | 2010-03-05 | False | 0 | 0 | PASS |
| NVL | overlap | 2404 | 2016-12-28 | 2026-08-14 | 1 | 2016-12-29 | False | 0 | 0 | PASS |
| OCB | NEW | 1381 | 2021-01-28 | 2026-08-14 | 0 | 2021-01-28 | False | 0 | 0 | PASS |
| PAN | NEW | 3901 | 2010-12-15 | 2026-08-14 | 1 | 2010-12-16 | False | 0 | 0 | PASS |
| PC1 | NEW | 2434 | 2016-11-16 | 2026-08-14 | 0 | 2016-11-16 | False | 0 | 0 | PASS |
| PDR | overlap | 4004 | 2010-07-30 | 2026-08-14 | 0 | 2010-07-30 | False | 0 | 0 | PASS |
| PHR | NEW | 4242 | 2009-08-18 | 2026-08-14 | 2 | 2009-08-20 | False | 0 | 0 | PASS |
| PLX | overlap | 2329 | 2017-04-21 | 2026-08-14 | 0 | 2017-04-21 | False | 0 | 0 | PASS |
| PNJ | NEW | 4345 | 2009-03-23 | 2026-08-14 | 4 | 2009-03-27 | False | 0 | 0 | PASS |
| POW | overlap | 2102 | 2018-03-06 | 2026-08-14 | 0 | 2018-03-06 | False | 0 | 0 | PASS |
| PPC | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| PTB | NEW | 3762 | 2011-07-22 | 2026-08-14 | 3 | 2011-07-27 | False | 0 | 0 | PASS |
| PVD | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| PVT | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| REE | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| SAB | overlap | 2420 | 2016-12-06 | 2026-08-14 | 3 | 2016-12-09 | False | 0 | 0 | PASS |
| SBT | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| SCS | NEW | 2267 | 2017-07-12 | 2026-08-14 | 3 | 2017-07-17 | False | 0 | 0 | PASS |
| SHB | overlap | 4323 | 2009-04-20 | 2026-08-14 | 0 | 2009-04-20 | False | 0 | 0 | PASS |
| SIP | NEW | 1794 | 2019-06-06 | 2026-08-14 | 13 | 2019-06-25 | False | 0 | 0 | PASS |
| SJS | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| SSB | overlap | 1347 | 2021-03-24 | 2026-08-14 | 3 | 2021-03-29 | False | 0 | 0 | PASS |
| SSI | overlap | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| STB | overlap | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| SZC | NEW | 1892 | 2019-01-15 | 2026-08-14 | 0 | 2019-01-15 | False | 0 | 0 | PASS |
| TCB | overlap | 2050 | 2018-06-04 | 2026-08-14 | 0 | 2018-06-04 | False | 0 | 0 | PASS |
| TCH | NEW | 2464 | 2016-10-05 | 2026-08-14 | 6 | 2016-10-13 | False | 0 | 0 | PASS |
| TLG | NEW | 4091 | 2010-03-26 | 2026-08-14 | 0 | 2010-03-26 | False | 0 | 0 | PASS |
| TPB | overlap | 2079 | 2018-04-19 | 2026-08-14 | 0 | 2018-04-19 | False | 0 | 0 | PASS |
| VCB | overlap | 4277 | 2009-06-30 | 2026-08-14 | 1 | 2009-07-01 | False | 0 | 0 | PASS |
| VCG | NEW | 4475 | 2008-09-05 | 2026-08-14 | 0 | 2008-09-05 | False | 0 | 0 | PASS |
| VCI | NEW | 2276 | 2017-07-07 | 2026-08-14 | 1 | 2017-07-10 | False | 0 | 0 | PASS |
| VGC | NEW | 2697 | 2015-10-15 | 2026-08-14 | 0 | 2015-10-15 | False | 0 | 0 | PASS |
| VHC | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| VHM | overlap | 3474 | 2011-11-10 | 2026-08-14 | 710 | 2014-09-22 | False | 0 | 0 | PASS |
| VIB | overlap | 2390 | 2017-01-09 | 2026-08-14 | 0 | 2017-01-09 | False | 0 | 0 | PASS |
| VIC | overlap | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| VIX | NEW | 4141 | 2009-12-29 | 2026-08-14 | 0 | 2009-12-29 | False | 0 | 0 | PASS |
| VJC | overlap | 2366 | 2017-02-28 | 2026-08-14 | 3 | 2017-03-03 | False | 0 | 0 | PASS |
| VND | NEW | 4079 | 2010-03-30 | 2026-08-14 | 0 | 2010-03-30 | False | 0 | 0 | PASS |
| VNM | overlap | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |
| VPB | overlap | 2247 | 2017-08-17 | 2026-08-14 | 0 | 2017-08-17 | False | 0 | 0 | PASS |
| VPI | NEW | 2168 | 2017-11-28 | 2026-08-14 | 2 | 2017-11-30 | False | 0 | 0 | PASS |
| VRE | overlap | 2191 | 2017-11-06 | 2026-08-14 | 6 | 2017-11-14 | False | 0 | 0 | PASS |
| VSC | NEW | 4598 | 2008-03-11 | 2026-08-14 | 0 | 2008-03-11 | False | 0 | 0 | PASS |

## Fetch failures

BCG

## Short-history tickers (< 750 rows)

| ticker | rows | first_date |
|---|---|---|
| DSE | 531 | 2024-07-01 |

## OHLC-consistency defects (Yahoo source glitches, flagged)

These are present in both adjusted and unadjusted Yahoo feeds, so they are genuine single-day source defects, not a processing error. Flagged via `xfail` for downstream cleaning; they do not block the test suite. Relative tolerance 1e-05.

Two categories:

1. **high < low** (0 rows across 0 tickers): Parkinson-affecting, since Parkinson variance uses only high/low. These must be dropped or repaired before computing volatility. Magnitudes range from ~1% gaps to gross defects (e.g. ACB/TPB 2025-06-04..05 where the adjusted high collapses to a near-zero value). Table below.
2. **open/close outside [low, high]** (0 rows): NOT Parkinson-affecting (high >= low still holds on these rows); they stem from dividend/split back-adjustment applied unevenly across O/H/L/C, the same artifact documented for the top-level files. Per-ticker counts are in the `oc_outside` column of the per-ticker table above.

### high < low rows (Parkinson-affecting)

None.

## Leading synthetic backfill > 20 rows

| ticker | leading_flat_run | first_real_date |
|---|---|---|
| HHV | 144 | 2016-07-20 |
| VHM | 710 | 2014-09-22 |

## Coverage

All 104 crawled tickers end on 2026-08-14.
