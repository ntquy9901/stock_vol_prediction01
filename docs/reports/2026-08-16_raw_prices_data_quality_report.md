# Raw price files data-quality report

Generated: 2026-08-16 10:11

Source: `data/raw/prices/*_ohlcv.csv` (33 tickers). Read-only; no CSV was modified.

Checks 1-4 (schema, dates, OHLC validity, NaN/inf) are enforced as hard assertions in `tests/test_raw_prices_quality.py`. Checks 5-6 (synthetic backfill, coverage) are reported below.

`leading_synthetic_run` = leading contiguous run of flat rows (high == low), which produce a Parkinson variance of 0. `first_real_date` = first row with high != low. See module docstring for the definitional rationale (backfill mixes zero and tiny-nonzero volume).

## Per-ticker diagnostics

| ticker | rows | first_date | last_date | leading_synthetic_run | first_real_date | flat_frac | zerovol_frac | any_OHLC_violation |
|---|---|---|---|---|---|---|---|---|
| ACB | 4916 | 2006-11-21 | 2026-08-14 | 0 | 2006-11-21 | 0.0120 | 0.0000 | False |
| BCM | 2113 | 2018-02-21 | 2026-08-14 | 0 | 2018-02-21 | 0.0345 | 0.0047 | False |
| BID | 3131 | 2014-01-24 | 2026-08-14 | 0 | 2014-01-24 | 0.0019 | 0.0006 | False |
| BVH | 4280 | 2009-06-25 | 2026-08-14 | 0 | 2009-06-25 | 0.0061 | 0.0005 | False |
| CTG | 4265 | 2009-07-16 | 2026-08-14 | 0 | 2009-07-16 | 0.0026 | 0.0005 | False |
| FPT | 4902 | 2006-12-13 | 2026-08-14 | 6 | 2006-12-21 | 0.0324 | 0.0004 | False |
| GAS | 3556 | 2012-05-21 | 2026-08-14 | 0 | 2012-05-21 | 0.0014 | 0.0006 | False |
| GVR | 2094 | 2018-03-21 | 2026-08-14 | 0 | 2018-03-21 | 0.0110 | 0.0000 | False |
| HDB | 2148 | 2018-01-05 | 2026-08-14 | 0 | 2018-01-05 | 0.0014 | 0.0009 | False |
| HPG | 4673 | 2007-11-15 | 2026-08-14 | 0 | 2007-11-15 | 0.0240 | 0.0004 | False |
| LPB | 1438 | 2020-11-09 | 2026-08-14 | 0 | 2020-11-09 | 0.0007 | 0.0000 | False |
| MBB | 3691 | 2011-11-01 | 2026-08-14 | 0 | 2011-11-01 | 0.0019 | 0.0005 | False |
| MSN | 4186 | 2009-11-05 | 2026-08-14 | 1 | 2009-11-06 | 0.0055 | 0.0005 | False |
| MWG | 3021 | 2014-07-14 | 2026-08-14 | 0 | 2014-07-14 | 0.0040 | 0.0007 | False |
| NVL | 2404 | 2016-12-28 | 2026-08-14 | 1 | 2016-12-29 | 0.0079 | 0.0008 | False |
| PDR | 4004 | 2010-07-30 | 2026-08-14 | 0 | 2010-07-30 | 0.0877 | 0.0357 | False |
| PLX | 2329 | 2017-04-21 | 2026-08-14 | 0 | 2017-04-21 | 0.0013 | 0.0009 | False |
| POW | 2102 | 2018-03-06 | 2026-08-14 | 0 | 2018-03-06 | 0.0005 | 0.0000 | False |
| SAB | 2420 | 2016-12-06 | 2026-08-14 | 3 | 2016-12-09 | 0.0037 | 0.0008 | False |
| SHB | 4323 | 2009-04-20 | 2026-08-14 | 0 | 2009-04-20 | 0.0012 | 0.0000 | False |
| SSB | 1347 | 2021-03-24 | 2026-08-14 | 3 | 2021-03-29 | 0.0022 | 0.0000 | False |
| SSI | 4890 | 2006-12-15 | 2026-08-14 | 0 | 2006-12-15 | 0.0331 | 0.0004 | False |
| STB | 4935 | 2006-10-27 | 2026-08-14 | 1 | 2006-10-30 | 0.0320 | 0.0004 | False |
| TCB | 2050 | 2018-06-04 | 2026-08-14 | 0 | 2018-06-04 | 0.0005 | 0.0000 | False |
| TPB | 2079 | 2018-04-19 | 2026-08-14 | 0 | 2018-04-19 | 0.0005 | 0.0000 | False |
| VCB | 4277 | 2009-06-30 | 2026-08-14 | 1 | 2009-07-01 | 0.0016 | 0.0005 | False |
| VHM | 2058 | 2018-05-23 | 2026-08-14 | 0 | 2018-05-23 | 0.0024 | 0.0005 | False |
| VIB | 2390 | 2017-01-09 | 2026-08-14 | 0 | 2017-01-09 | 0.0088 | 0.0004 | False |
| VIC | 4714 | 2007-09-19 | 2026-08-14 | 1 | 2007-09-20 | 0.0240 | 0.0004 | False |
| VJC | 2366 | 2017-02-28 | 2026-08-14 | 3 | 2017-03-03 | 0.0038 | 0.0008 | False |
| VNM | 4935 | 2006-10-27 | 2026-08-14 | 1 | 2006-10-30 | 0.0274 | 0.0004 | False |
| VPB | 2334 | 2017-08-17 | 2026-08-14 | 0 | 2017-08-17 | 0.0716 | 0.0711 | False |
| VRE | 2277 | 2017-11-06 | 2026-08-14 | 6 | 2017-11-14 | 0.0738 | 0.0720 | False |

## Tickers with leading synthetic backfill > 20 rows (recommended trim point)

None.

VHM is the only ticker with non-trivial leading synthetic backfill. Its flat prefix ends 2014-09-22 (first high != low), but sustained liquid trading only begins 2018-05-23 (first day with volume in the millions; matches the ~2018-05 HOSE listing). The 2014-09..2018-05 window is sparse/illiquid (sporadic tiny volume, frequent flat days). Trimming to 2014-09-22 removes the Parkinson-zero prefix; trimming to 2018-05-23 additionally removes the illiquid warm-up. No other ticker has a leading flat run exceeding 6 rows.

## Check 3: OHLC-consistency violations (hard-fail)

None.

## Check 6: coverage

All 33 tickers end on 2026-08-14.
