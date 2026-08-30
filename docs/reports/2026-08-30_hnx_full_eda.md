# HNX full EDA — key findings & dirty-data triage

Universe: **299** HNX tickers (raw OHLCV ∩ processed Parkinson), 2005-12-28 .. 2026-08-21, 1,065,741 ticker-days. `parkinson_volatility` is VARIANCE σ², not σ.

## Headline dirty-data figures
- **Zero-Parkinson (H=L illiquid) days: 45.3%** of all ticker-days (target is exactly zero → floored in QLIKE).
- **Liquidity screen keeps 162/299** tickers (≥250 rows and ≤50% zero-Parkinson); **137 dropped**.
- **3467 corrupt OHLC rows** (nonpositive / high<low / open-close outside [low,high]); **486,264 zero-range (H=L)** rows.
- **49 candidate unadjusted-split jumps** (|return|>50%).

## New insights
1. Extreme illiquidity is the dominant data property, not an edge case (45.3% zero-variance days). The usable HNX universe is a minority of the listed universe.
2. Return co-movement is weak: median |corr| = 0.094, only 1/13,038 pairs >0.7 → little for a spatial graph to exploit.
3. OHLC-geometry violations cluster on shared calendar dates across many tickers (data-source seam), not random per-ticker noise.
4. Heavy tails (return excess kurtosis 102.1) + persistent |return| autocorrelation = genuine volatility clustering; supports HAR as a strong baseline.

## Prioritized dirty-data issues
| issue | severity | affects the volatility TARGET/results? |
|---|---|---|
| Zero-Parkinson illiquid days (H=L) | HIGH | YES — floors the target, inflates QLIKE, makes point metrics uninformative for illiquid names; already mitigated by the liquidity screen. |
| Corrupt OHLC rows (open/close outside [low,high], nonpositive) | MEDIUM | YES if used — the estimator already NaNs them; must stay excluded. |
| Unadjusted split/dividend jumps (|ret|>50%) | MEDIUM | YES for close-to-close / overnight estimators; Parkinson (intraday range) is immune. |
| Stale/flat close runs & zero-volume days | LOW-MEDIUM | INDIRECT — a symptom of the same illiquidity captured by the zero-Parkinson screen. |
| Leading backfilled prefix before listing | LOW | COSMETIC once the screen + rolling warm-up drop the early rows. |

## Recommendation for the paper's data-limitations section
Report HNX as a **thin, illiquid market**: 45.3% zero-variance ticker-days, only 162/299 names surviving a ≤50% zero-Parkinson liquidity screen. State that (a) prices are NOT split-adjusted (jumps flagged, Parkinson intraday-range target is robust to overnight gaps), (b) the target is a VARIANCE, (c) the zero-Parkinson floor makes QLIKE floor-sensitive on HNX, and (d) weak cross-sectional correlation limits the headroom for spatial-graph models on this panel.

## Corrupt-bar date clusters (seam)

- 2013-11-22: 54 tickers
- 2013-05-24: 48 tickers
- 2011-10-13: 43 tickers
- 2013-06-21: 32 tickers
- 2007-03-27: 31 tickers
- 2007-12-21: 24 tickers
- 2007-08-06: 23 tickers
- 2007-08-21: 22 tickers
- 2007-07-19: 22 tickers
- 2007-08-08: 21 tickers
- 2007-06-29: 21 tickers
- 2007-07-18: 21 tickers
