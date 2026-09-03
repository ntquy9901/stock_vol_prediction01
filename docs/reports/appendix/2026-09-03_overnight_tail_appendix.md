# Appendix — Overnight-return tail contamination by ticker (data-quality)

Scope: raw daily OHLCV, **not split/dividend-adjusted**. Companion table:
`2026-09-03_overnight_tail_tickers.csv` (per-ticker counts and worst case).

## Definition

For each ticker the overnight log-return is `r_o(t) = ln( O_t / C_{t-1} )`, defined only where
`C_{t-1} > 0`. A day is flagged as a tail day when `|r_o(t)| > 0.20` (twice the ±10% HOSE/HNX daily
price limit); any move beyond this bound is inconsistent with continuous trading and indicates an
unadjusted corporate action, a re-listing gap, penny-price tick rounding, or a bad print. The intraday
Parkinson estimator `σ²_P = ln(H_t/L_t)² / (4 ln 2)` uses same-day high/low only and is invariant to
these overnight discontinuities; overnight- and close-to-close-based estimators (Rogers–Satchell with an
overnight term, Yang–Zhang, close-to-close) are not.

## Market-level incidence

| Market | Tickers | Overnight obs | Tail days (`|r_o|>20%`) | Rate | Tickers affected | Worst overnight |
|---|---|---:|---:|---:|---:|---|
| VN30 (blue-chip) | 33 | 106,615 | 1 | 0.00% | 1 | FPT +37% (2007-08-17) |
| HOSE | 406 | 1,387,670 | 234 | 0.02% | 87 | SMA +4455% (2020-12-08) |
| HNX | 300 | 1,068,976 | 1,100 | 0.10% | 87 | L40 +456% (2021-01-27) |

Incidence falls monotonically with liquidity: HNX carries ~5× the tail-day rate of HOSE and ~50× that of
VN30. The 1,100 HNX tail days reproduce the count reported in the estimator study
(`docs/reports/2026-08-23_1600_volatility_estimator_research.md`).

## Cause categories (illustrative)

| Category | Mechanism | Examples |
|---|---|---|
| Unadjusted split / bonus issue | Prior close on the pre-split scale; single consecutive-day gap | EBS −61% (2007-06-18); S99 −69% (2007-07-31); NBW −84% (2011-03-18) |
| Re-listing after suspension | Multi-year gap billed as one overnight move | L40 +456% (2021-01-27, 2,541-day gap) |
| Penny-price tick rounding | Sub-unit prices; tick rounding yields large % moves; recurs many days | PTX (270 tail days; prev 0.40 → open 0.07) |
| Bad print / decimal error | Physically impossible single-day jump | SMA +4455% (2020-12-08, 7.28 → 331.63) |
| Early-market data era (2007) | Cluster of unadjusted actions during 2007 restructuring | STP, SD5, SD9, VMC, VTV, SDC (worst dates 2007-07…08) |

## Most-affected tickers (by tail-day count)

### HNX (top 15 of 87)

| Ticker | Tail days | Worst % | Worst date | Gap (cal. days) |
|---|---:|---:|---|---:|
| PTX | 270 | −82 | 2020-08-18 | 1 |
| PTD | 55 | −45 | 2014-07-21 | 3 |
| STP | 55 | −39 | 2007-07-03 | 1 |
| SD9 | 54 | −50 | 2007-07-02 | 3 |
| VMC | 54 | −40 | 2007-08-17 | 1 |
| VTV | 54 | −37 | 2007-08-14 | 1 |
| SD5 | 53 | −63 | 2007-07-04 | 1 |
| S99 | 51 | −69 | 2007-07-31 | 1 |
| SDC | 47 | −37 | 2007-07-06 | 1 |
| MCO | 39 | −43 | 2007-08-20 | 3 |
| VNR | 30 | −33 | 2007-07-18 | 1 |
| KSV | 29 | −40 | 2016-11-15 | 1 |
| MVB | 27 | −30 | 2018-11-02 | 1 |
| GDW | 23 | −40 | 2012-05-16 | 1 |
| SJE | 23 | −41 | 2007-07-03 | 1 |

### HOSE (top 15 of 87)

| Ticker | Tail days | Worst % | Worst date | Gap (cal. days) |
|---|---:|---:|---|---:|
| ABR | 23 | −40 | 2019-11-28 | 1 |
| TCI | 13 | −40 | 2020-03-20 | 1 |
| ADP | 11 | −41 | 2017-08-15 | 1 |
| NHH | 11 | −24 | 2018-05-14 | 3 |
| PVP | 8 | −32 | 2017-08-18 | 1 |
| PDV | 7 | −40 | 2018-02-07 | 1 |
| DSC | 6 | +30 | 2020-07-02 | 1 |
| AFX | 5 | +40 | 2017-02-13 | 3 |
| HAP | 5 | −44 | 2001-12-19 | 2 |
| HAS | 5 | −26 | 2009-07-15 | 1 |
| SFC | 5 | −39 | 2009-09-07 | 3 |
| VCA | 5 | −51 | 2014-08-08 | 1 |
| ANT | 4 | +36 | 2019-12-17 | 1 |
| COM | 4 | +50 | 2010-08-27 | 1 |
| HAX | 4 | −41 | 2009-06-29 | 3 |

VHM (2018-05-17, prev 8.41 → 30.99, +268%) is an early-listing reference-price artifact and is trimmed
before 2018-05-23 in the processed pipeline.

## Effect on estimators and handling

On a tail day the intraday Parkinson value is typically ~1e-3 (an ordinary session), whereas the squared
overnight term entering Rogers–Satchell-overnight / close-to-close reaches 0.9–3.5 — three orders of
magnitude larger — and, being squared, dominates the ticker's MSE. Measured on the canonical universe
(`2026-08-23_1600_volatility_estimator_research.md`), overnight/close-to-close targets carry MSE 3.7–4.4×
that of Parkinson while QLIKE (less tail-sensitive) rises only 1.2–1.7×; the gap between the two ratios is
the tail signature. Because these events are rare on VN30 and frequent on HNX, the estimator ranking is
liquidity-dependent.

Handling in this project: (1) Parkinson is the default target across all markets (intraday-only, split-
invariant); (2) where an overnight term is required, the overnight log-return is winsorized at ±0.20; (3)
the recommended root fix is split/dividend-adjusted prices, which removes the corporate-action subset of
these tail days.
