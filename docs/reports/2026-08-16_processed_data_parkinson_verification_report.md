# Processed Data + Parkinson Volatility Verification Report

Date: 2026-08-16
Scope: read-only independent verification of `data/processed/<TICKER>_processed.csv` (33 tickers)
against raw OHLCV in `data/raw/prices/<TICKER>_ohlcv.csv`, and independent recomputation of
Parkinson variance.

## Method

- Structural checks read each processed file directly (row count, dates, columns, finiteness,
  value stats).
- Correctness recomputes Parkinson variance from raw OHLCV independently using
  `parkinson = (np.log(high/low)**2) / (4*np.log(2))`, applies the same date normalization
  (`str.split(' ').str[0]`), drops NaN/inf, clips upper at 0.1, then joins on `date` and compares
  against the stored value.
- Tickers processed sequentially (single process, modest memory) alongside the running GPU job.
- No data files were modified.

## Verdict

- (a) Structural integrity: PASS (33/33 tickers).
- (b) Parkinson correctness: PASS (33/33 tickers). Max absolute difference across all tickers is
  ~1e-16 (IEEE-754 double rounding), 0 rows differ by more than 1e-9.

## 1 + 3. Per-ticker structural + correctness

- `rows` = processed row count. `last` = last date (expected 2026-08-14). `mono` = strictly
  increasing unique dates. `NaN/inf` = count of non-finite parkinson values. `max_abs_diff` =
  max |stored - recomputed| over joined dates. `n_diff` = rows differing > 1e-9. `clip` = rows at
  the 0.1 ceiling. `zeros` = rows with parkinson == 0 (H == L days). All join fully (compared rows
  == processed rows for every ticker).

| Ticker | rows | last | last=08-14 | mono | NaN/inf | max_abs_diff | n_diff | clip | zeros |
|--------|-----:|------|:---:|:---:|:---:|---:|:---:|----:|----:|
| ACB | 4916 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 60 |
| BCM | 2113 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 79 |
| BID | 3131 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 6 |
| BVH | 4280 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 26 |
| CTG | 4265 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 11 |
| FPT | 4902 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 159 |
| GAS | 3556 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 5 |
| GVR | 2094 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 23 |
| HDB | 2148 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 3 |
| HPG | 4673 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 113 |
| LPB | 1438 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 1 |
| MBB | 3691 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 7 |
| MSN | 4186 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 23 |
| MWG | 3021 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 12 |
| NVL | 2404 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 19 |
| PDR | 4004 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 351 |
| PLX | 2329 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 3 |
| POW | 2102 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 1 |
| SAB | 2420 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 9 |
| SHB | 4323 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 5 |
| SSB | 1347 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 3 |
| SSI | 4889 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 163 |
| STB | 4935 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 158 |
| TCB | 2050 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 1 |
| TPB | 2079 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 1 |
| VCB | 4277 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 7 |
| VHM | 3474 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 1408 |
| VIB | 2390 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 21 |
| VIC | 4714 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 113 |
| VJC | 2366 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 9 |
| VNM | 4935 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 136 |
| VPB | 2334 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 167 |
| VRE | 2277 | 2026-08-14 | Y | Y | 0 | 1.0e-16 | 0 | 0 | 168 |

All 33 tickers: exactly 2 columns (`date,parkinson_volatility`), 0 duplicate dates, all values
finite and >= 0, last date == 2026-08-14, full date-join with recompute.

## 2. Value sanity (clip + zeros)

- Clip binding: 0 rows at the 0.1 ceiling across the entire dataset (33/33 tickers, total_clip=0).
  The clip never binds; no sign of pathological raw H/L blow-ups.
- Exact zeros (H == L flat days): 3271 rows total, all valid. Most tickers show a small handful
  (single digits to low hundreds), consistent with occasional flat/limit-locked trading days.
- Notable: VHM has 1408 zero rows (~40% of its series), concentrated in 2011-2017
  (2012-2016 ~245-250 flat days per year). VHM (Vinhomes) listed 2018-05, so its pre-2018 rows are
  backfilled flat data (H == L) in the raw source. This is a raw-data provenance characteristic,
  not a pipeline defect: the pipeline correctly emits parkinson == 0 for H == L. Flagged for
  awareness (pre-2018 VHM carries no intraday range information).

## 4. New-data spot check (window 2026-06-10 .. 2026-08-14)

Last 5 appended rows per ticker; recomputed vs stored Parkinson (raw H, L shown). All 5 tickers
have 48 rows in the window.

VPB (tz-aware datetime + full-VND prices):
```
2026-08-10  H=25850    L=25100    recomp=3.126594e-04  stored=3.126594e-04  diff=7.7e-17
2026-08-11  H=26000    L=25500    recomp=1.359964e-04  stored=1.359964e-04  diff=9.1e-17
2026-08-12  H=25800    L=25350    recomp=1.116685e-04  stored=1.116685e-04  diff=6.8e-17
2026-08-13  H=25800    L=25200    recomp=1.996994e-04  stored=1.996994e-04  diff=7.3e-17
2026-08-14  H=25350    L=24850    recomp=1.431317e-04  stored=1.431317e-04  diff=7.2e-17
```

VRE (tz-aware datetime + full-VND prices):
```
2026-08-10  H=26456    L=25317    recomp=6.984898e-04  stored=6.984898e-04  diff=8.2e-17
2026-08-11  H=26766.7  L=26093.6  recomp=2.339142e-04  stored=2.339142e-04  diff=2.3e-17
2026-08-12  H=26922    L=26248.9  recomp=2.311888e-04  stored=2.311888e-04  diff=8.3e-17
2026-08-13  H=26507.8  L=25472.3  recomp=5.726405e-04  stored=5.726405e-04  diff=5.1e-17
2026-08-14  H=25472.3  L=24488.7  recomp=5.594218e-04  stored=5.594218e-04  diff=3.4e-17
```

LPB (SSI iBoard, thousands-VND prices):
```
2026-08-10  H=53.1  L=52.0  recomp=1.580470e-04  stored=1.580470e-04  diff=5.6e-17
2026-08-11  H=53.8  L=52.3  recomp=2.883926e-04  stored=2.883926e-04  diff=2.8e-17
2026-08-12  H=54.5  L=53.3  recomp=1.787868e-04  stored=1.787868e-04  diff=2.6e-17
2026-08-13  H=54.4  L=52.5  recomp=4.558456e-04  stored=4.558456e-04  diff=2.9e-17
2026-08-14  H=54.0  L=52.0  recomp=5.137193e-04  stored=5.137193e-04  diff=2.8e-17
```

ACB (normal, thousands-VND prices):
```
2026-08-10  H=26.5567  L=26.0325  recomp=1.433522e-04  stored=1.433522e-04  diff=4.2e-17
2026-08-11  H=26.4985  L=26.2073  recomp=4.404020e-05  stored=4.404020e-05  diff=6.8e-21
2026-08-12  H=26.5567  L=26.3237  recomp=2.800932e-05  stored=2.800932e-05  diff=0.0
2026-08-13  H=26.5567  L=25.8578  recomp=2.565389e-04  stored=2.565389e-04  diff=8.6e-17
2026-08-14  H=25.9743  L=25.7414  recomp=2.926002e-05  stored=2.926002e-05  diff=0.0
```

VNM (normal, thousands-VND prices):
```
2026-08-10  H=64.7548  L=63.6188  recomp=1.129802e-04  stored=1.129802e-04  diff=8.2e-17
2026-08-11  H=65.3745  L=63.8253  recomp=2.074471e-04  stored=2.074471e-04  diff=6.8e-17
2026-08-12  H=64.5483  L=63.4122  recomp=1.137309e-04  stored=1.137309e-04  diff=1.1e-17
2026-08-13  H=65.0647  L=63.5155  recomp=2.094514e-04  stored=2.094514e-04  diff=6.6e-17
2026-08-14  H=64.6516  L=62.9991  recomp=2.418018e-04  stored=2.418018e-04  diff=2.9e-17
```

Scale-invariance confirmed: VPB/VRE store full-VND prices (~25000-27000) while ACB/VNM/LPB store
thousands-VND or SSI-scale prices (~26-65). Because Parkinson uses only the H/L ratio, the stored
values are on the same magnitude (~1e-4) regardless of price scale, and match the independent
recompute to double precision. The tz-aware datetime (VPB/VRE) and SSI iBoard format (LPB) do not
corrupt dates or values: the space-split date normalization preserves the calendar date and all
appended rows match.

## 5. Units

The stored quantity is Parkinson VARIANCE (sigma^2), consistent with
`(ln(H/L))^2 / (4 ln 2)`, not standard deviation. Typical magnitude is ~1e-4 to ~1e-3
(per-ticker means ~1.8e-4 to ~7.8e-4; e.g. ACB mean 4.29e-4, VNM mean 1.85e-4). This matches the
variance formula directly: for a representative daily H/L range of ~2% (ln(H/L) ~ 0.02),
0.02^2 / (4 ln 2) ~ 1.4e-4, in line with the observed values. Taking the square root would give
~1e-2 (a standard-deviation scale), which is NOT what is stored. The column is variance, matching
the CLAUDE.md note that `parkinson_volatility` is sigma^2.

## Summary

- Structural integrity: PASS, 33/33.
- Parkinson correctness: PASS, 33/33 (max abs diff ~1e-16, 0 rows > 1e-9).
- Clip never binds (0 hits). Zeros are valid H == L flat days; VHM's 1408 zeros are pre-2018
  backfilled flat raw data (provenance note, not a defect).
- New-data window (2026-06-10..08-14) and the tz-aware (VPB/VRE) / SSI (LPB) formats are correct;
  scale-invariance holds.
- Stored quantity confirmed as variance (sigma^2), ~1e-4 magnitude.
