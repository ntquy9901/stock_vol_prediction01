# S&P500 daily OHLCV data-quality report (raw + processed)

Generated: 2026-08-21 14:12

Source: `data/raw/prices/sp500/*_ohlcv.csv` (503 tickers) and `data/processed/sp500/*_processed.csv` (503 tickers). Yahoo Finance sourced (auto_adjust). Read-only; no CSV was modified.

## Summary

RAW hard checks (schema, dates, OHLC positivity, NaN/inf) are enforced as assertions. OHLC-consistency, volume-glitch and leading-backfill are reported via `xfail` (documented Yahoo tolerances). PROCESSED checks (schema, dates, values >=0 and <= clip ceiling) are hard assertions.

- RAW rows with high < low: **0** (across 0 tickers)
- RAW rows with open/close outside [low, high]: **0**
- RAW non-positive OHLC rows: **0**
- RAW volume-glitch rows (volume==0 while high!=low): **381** (across 28 tickers)
- RAW leading-flat backfill (> 20 rows): AIG(1820), AMCR(181), AXP(230), BAC(50), BALL(50), CRH(22), CVS(50), ECL(50), GWW(50), HUBB(3932), LNT(50), MKC(50), MRSH(50), NI(50), PNC(50), RVTY(50), SNA(50), SPGI(50), SW(69), TER(50), TRV(50)
- RAW short-history tickers (< 750 rows): 8
- PROCESSED negative values: 0 (tickers: none)
- PROCESSED NaN/inf values: 0 (tickers: none)
- PROCESSED values over clip ceiling 0.1: 0 (tickers: none)
- PROCESSED zero-Parkinson rows: 50292 / 4379391 (1.1484% overall; H==L limit/flat days)
- Coverage: tickers whose last_date != 2026-08-19: none

## RAW high < low rows (Parkinson-affecting)

None.

## RAW volume-glitch rows (volume==0 while price moved)

| ticker | volume_glitch_rows |
|---|---|
| HUBB | 184 |
| HWM | 35 |
| BRO | 29 |
| TFC | 29 |
| MGM | 22 |
| MNST | 21 |
| UDR | 10 |
| CINF | 9 |
| MTB | 6 |
| CHD | 4 |
| FITB | 4 |
| BEN | 3 |
| CAH | 3 |
| JNJ | 3 |
| PGR | 3 |
| NDSN | 2 |
| WFC | 2 |
| WRB | 2 |
| ATO | 1 |
| CNC | 1 |
| FERG | 1 |
| MCHP | 1 |
| MKC | 1 |
| MO | 1 |
| MSI | 1 |
| NTRS | 1 |
| TSN | 1 |
| WM | 1 |

## RAW short-history tickers (< 750 rows)

| ticker | rows | first_date | last_date |
|---|---|---|---|
| HONA | 46 | 2026-06-15 | 2026-08-19 |
| FDXF | 59 | 2026-05-27 | 2026-08-19 |
| Q | 204 | 2025-10-27 | 2026-08-19 |
| SNDK | 380 | 2025-02-13 | 2026-08-19 |
| GEV | 601 | 2024-03-27 | 2026-08-19 |
| SOLV | 602 | 2024-03-26 | 2026-08-19 |
| RDDT | 605 | 2024-03-21 | 2026-08-19 |
| VLTO | 721 | 2023-10-04 | 2026-08-19 |

## RAW coverage

Row counts: min=46, median=8681, max=16266 across 503 tickers.

