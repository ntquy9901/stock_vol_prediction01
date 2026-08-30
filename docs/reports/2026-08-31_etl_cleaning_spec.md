# Consolidated ETL-cleaning specification — dirty-data classes across VN + US markets

Generated 2026-08-30 22:27. Parkinson target = sigma^2 (VARIANCE), H/L only. CPU/pandas only. Every count traces to the audited data.

## Priority: TARGET-affecting vs cosmetic

REAL (moves the delivered Parkinson vol target / results): high<low, nonpositive OHLC, zero-range, NaN/inf. Cosmetic for the Parkinson target (only touches the open/close-using estimators GK/RS/YZ, or leading rows / liquidity flags): open/close-outside, split jumps, stale closes, leading backfill, zero-volume.

Note: Parkinson ln(H/L)^2 and the GK/RS within-day ratios are SCALE-INVARIANT, so an unadjusted split (a uniform price rescale) does NOT change their value on any day -- only Yang-Zhang's overnight term is affected, and only on the split boundary day. Back-adjusting a split therefore does not move the delivered Parkinson target; it matters for close-to-close / overnight estimators and for any level-based (non-ratio) feature.

Prioritised action order (fix first -> last): (1) NaN/inf drop, (2) nonpositive reconstruct, (3) high<low swap/drop, (4) zero-range flag + liquidity screen / vol floor [the dominant target driver], (5) leading-backfill cut, then cosmetic: (6) open/close-outside widen [GK/RS/YZ only], (7) split back-adjust [overnight estimators only], (8) stale/zero-volume flags.

## Per-class detection + cleaning rules

| # | issue | detection rule | cleaning rule | estimators affected | priority |
|---|---|---|---|---|---|
| 1 | high_lt_low | high < low (finite) | swap H<->L if a transposition, else drop the bar | Parkinson, GK, RS, YZ | REAL |
| 2 | open_close_outside | high < max(O,C)*(1-1e-5) or low > min(O,C)*(1+1e-5) | WIDEN range H=max(H,O,C), L=min(L,O,C) (recommended); or clip O/C into [L,H] | GK, RS, YZ (Parkinson immune: H/L only) | cosmetic (Parkinson) |
| 3 | nonpositive | any O/H/L/C <= 0 | reconstruct H/L from positive OHLC (max/min), clamp O/C; else drop | Parkinson, GK, RS, YZ | REAL |
| 4 | zero_range | finite positive high == low | KEEP + FLAG (liquidity screen / vol floor); do NOT delete | Parkinson, GK, RS, YZ | REAL |
| 5 | split_jumps | |1-day simple return| > 50% | back-adjust prior prices by the split factor; else flag+winsorize (does NOT move the Parkinson target) | YZ overnight-boundary only (Parkinson/GK/RS scale-invariant) | cosmetic (Parkinson) |
| 6 | stale_runs | >= 5 identical consecutive closes | flag; optionally drop the run | GK, RS, YZ (close-based); cosmetic for Parkinson | cosmetic (Parkinson) |
| 7 | naninf | non-finite O/H/L/C/volume | drop / impute; must never reach the model | Parkinson, GK, RS, YZ | REAL |
| 8 | zero_volume | finite volume == 0 | flag illiquidity (keep) | none directly (liquidity flag) | cosmetic (Parkinson) |
| 9 | leading_backfill | leading run: constant close + (zero volume or zero range) | cut to the true first-trade date | Parkinson, GK, RS, YZ (leading rows only) | cosmetic (Parkinson) |

## Cross-market dirty-data prevalence (raw ticker-day counts)

| market | tickers | ticker-days | high_lt_low | open_close_outside | nonpositive | zero_range | split_jumps | stale_runs | naninf | zero_volume | leading_backfill |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hnx | 299 | 1,069,451 | 3 | 3441 | 174 | 486273 | 53 | 297529 | 2 | 287238 | 423 |
| hose | 405 | 1,388,181 | 3 | 3541 | 84 | 211378 | 44 | 81466 | 23 | 74120 | 334 |
| vn30 | 33 | 106,648 | 0 | 0 | 0 | 1858 | 0 | 1180 | 0 | 523 | 14 |
| vn100 | 104 | 357,850 | 0 | 0 | 0 | 11968 | 2 | 6825 | 0 | 4197 | 48 |
| sp500 | 503 | 4,379,391 | 0 | 0 | 0 | 50292 | 88 | 49521 | 0 | 25051 | 101 |

Count units: most classes = ticker-days; stale_runs = stale days; leading_backfill = leading rows.

## Raw-vs-processed (does the current ETL already clean it?)

| market | processed rows | processed max | at 0.1 cap | clipped-from-raw (>0.1) | zero processed |
|---|---|---|---|---|---|
| hnx | 1,065,741 | 0.1 | 24 | 24 | 482,724 |
| hose | 1,381,390 | 0.1 | 1 | 1 | 204,668 |
| vn30 | 106,648 | 0.1 | 1 | 1 | 1,858 |
| vn100 | 357,850 | 0.09457 | 0 | 0 | 11,968 |
| sp500 | 4,379,391 | 0.1 | 172 | 172 | 50,292 |

Reading: a nonzero `clipped-from-raw` count is direct evidence the ETL upper-clips the Parkinson target at 0.1 (raw Parkinson exceeded 0.1 but the processed value is 0.1). `zero processed` = zero-range/limit days that pass through to the target and get floored in QLIKE scoring (the target-affecting driver).
