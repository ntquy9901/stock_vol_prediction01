# Model-Free Graph Screening — Extended (diagnosis doc §7 S5/S6, §15 richer node features)

Extends `reports/model_free_graph_screening.md` (S0–S4, HAR-only node features, no volume). Two gaps
were left open there: (a) a volume-shock graph (S5) needs volume, absent from the processed CSVs; (b)
the null was measured on HAR-only node features, so it did not rule out that those features are simply
too thin (§15). This run supplies volume from the raw OHLCV files and screens richer node features.

## Method (identical to the S0–S4 screen, for comparability)
Each candidate column is appended to the 3 HAR features (daily/weekly/monthly), fit by OLS with
intercept on TRAIN rows only; incremental value = `1 − MSE(HAR+feature)/MSE(HAR-only)` on TEST (and
VAL). Anchors (`FIRST_VALID+9 .. T−h`), the 80/10/10 chronological split, and the `h`-observation
purge at split boundaries mirror `screen_graph.py` exactly. Pooling is date-major, node-minor over all
(ticker, date) test observations. Volume/close are read from the raw `*_ohlcv.csv` files and
inner-reindexed to the Parkinson-panel dates (raw dates are a superset of the processed dates, so the
alignment is exact). Leakage safety: all rolling stats are causal (row `t` uses only rows `[t−win+1..t]`,
`win=22`); all cross-sectional aggregates use only day-`t` observations; the vshock adjacency and the
HAR fit are TRAIN-only. No train-only global standardization is needed because no statistic looks ahead.

Code: `baselines/2026-08-21_har_anchored_residual/code/screen_features.py` (smoke test
`test/test_screen_features.py`, incl. a causality check on the z-score). Raw JSON:
`results/graph_screen/{vn30,vn100,sp500}_features.json`.

Panels: VN30 (33 nodes, 1344 common dates), VN100 (104 nodes, 513 dates), S&P 500 (long-history
subset, min_common=3000 → 457 nodes, 3029 dates) — same node sets as the S0–S4 screen. Volume present
for every node on every panel.

## S5 — volume-shock graph (incremental TEST R² vs HAR)
`vshock[t] = (log(vol[t]) − rollmean_22) / rollstd_22`. Adjacency = TRAIN-only signed Top-5 vshock
correlation. Signal `weighted = Σ_j A_ij vshock[j,t]`; `mean` = equal-weight over the Top-5 neighbours.
PLACEBO = weighted signal with row/column-permuted edges (density/degree preserved).

| panel | h | weighted | mean | PLACEBO | signal−placebo |
|---|---:|---:|---:|---:|---:|
| VN30 | 1 | +0.0029 | +0.0042 | +0.0025 | +0.0004 |
| VN30 | 5 | +0.0038 | +0.0049 | +0.0007 | +0.0031 |
| VN30 | 10 | +0.0021 | +0.0025 | +0.0002 | +0.0019 |
| VN30 | 22 | +0.0003 | +0.0005 | −0.0000 | +0.0003 |
| VN100 | 1 | −0.0062 | −0.0070 | −0.0023 | −0.0039 |
| VN100 | 5 | +0.0128 | +0.0108 | +0.0056 | +0.0072 |
| VN100 | 10 | +0.0065 | +0.0056 | +0.0011 | +0.0054 |
| VN100 | 22 | +0.0002 | +0.0006 | +0.0004 | −0.0002 |
| S&P 500 | 1 | +0.0032 | +0.0036 | +0.0022 | +0.0010 |
| S&P 500 | 5 | −0.0006 | −0.0004 | −0.0014 | +0.0008 |
| S&P 500 | 10 | −0.0023 | −0.0023 | −0.0034 | +0.0011 |
| S&P 500 | 22 | +0.0001 | +0.0001 | +0.0000 | +0.0001 |

Reading: the largest volume-graph effect is VN100 h5 (weighted +1.28%), but its shuffled-edge placebo
is already +0.56%, so the structural-edge contribution over a random-edge control is +0.72%. Every
other panel/horizon is ≤ +0.5% and mostly near or below its placebo; VN100 h1 is negative. The signal
is horizon-isolated (VN100/VN30 peak at h5, gone by h22; S&P 500 is flat-to-negative at h5/h10) and
sign-unstable across panels. No configuration clears ~1% above placebo, and none is consistent across
horizons.

## Richer node features (§15) — incremental TEST R² vs HAR, each appended individually
`own_return`, `abs_return` from close; `market_return`/`market_pk` = cross-sectional mean per date;
`vol_ratio` = pk / market_pk; `vol_of_vol` = causal rolling-22 std of pk; `xsec_disp` = cross-sectional
std of pk per date; `own_vshock` = the volume shock. `ALL` = all eight appended together.

**VN30**
| h | own_ret | abs_ret | mkt_ret | mkt_pk | vol_ratio | vol_of_vol | xsec_disp | own_vshock | ALL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | −0.0000 | +0.0095 | +0.0205 | −0.0003 | −0.0003 | +0.0008 | +0.0057 | +0.0043 | +0.0202 |
| 5 | −0.0053 | −0.0004 | −0.0046 | +0.0000 | +0.0003 | +0.0007 | +0.0005 | +0.0045 | −0.0004 |
| 10 | −0.0040 | +0.0021 | −0.0019 | −0.0035 | −0.0006 | +0.0017 | −0.0052 | +0.0002 | −0.0068 |
| 22 | −0.0060 | −0.0007 | −0.0061 | +0.0028 | +0.0029 | +0.0105 | +0.0037 | +0.0000 | +0.0010 |

**VN100**
| h | own_ret | abs_ret | mkt_ret | mkt_pk | vol_ratio | vol_of_vol | xsec_disp | own_vshock | ALL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.0253 | +0.0198 | +0.0405 | −0.0000 | −0.0000 | +0.0006 | −0.0002 | −0.0075 | +0.0557 |
| 5 | −0.0041 | −0.0011 | +0.0092 | +0.0083 | +0.0019 | −0.0002 | +0.0072 | +0.0005 | +0.0069 |
| 10 | −0.0030 | +0.0002 | −0.0003 | −0.0006 | +0.0005 | +0.0038 | −0.0016 | +0.0009 | +0.0022 |
| 22 | −0.0008 | +0.0006 | +0.0001 | +0.0072 | +0.0043 | +0.0055 | +0.0059 | +0.0004 | +0.0083 |

**S&P 500**
| h | own_ret | abs_ret | mkt_ret | mkt_pk | vol_ratio | vol_of_vol | xsec_disp | own_vshock | ALL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | −0.0068 | −0.0074 | +0.0014 | +0.0183 | +0.0153 | +0.0074 | +0.0055 | +0.0027 | +0.0200 |
| 5 | −0.0140 | −0.0105 | −0.0006 | −0.0001 | +0.0032 | +0.0090 | −0.0100 | −0.0003 | −0.0053 |
| 10 | −0.0053 | −0.0057 | −0.0005 | −0.0017 | +0.0001 | +0.0095 | −0.0015 | −0.0013 | +0.0012 |
| 22 | −0.0016 | +0.0012 | −0.0006 | +0.0026 | +0.0061 | +0.0096 | +0.0014 | −0.0000 | +0.0158 |

Reading:
- The features that exceed 1% do so only at **h1** and only as contemporaneous level co-movement:
  `market_return` (VN30 +2.05%, VN100 +4.05%), `own_return`/`abs_return` (VN100 +2.5%/+2.0%),
  `market_pk`/`vol_ratio` (S&P 500 +1.83%/+1.53%). All collapse to ~0 or negative by h5–h22. This is
  the same horizon-isolated, level-driven pattern seen for the S0–S2 graph signals in the base screen.
- The only feature with a **consistent-sign, cross-horizon** positive is `vol_of_vol` (own rolling
  std of pk): S&P 500 +0.74/+0.90/+0.95/+0.96% across h1/5/10/22, VN30 up to +1.05% (h22), VN100
  +0.38/+0.55% (h10/h22). It reaches ~1% only on the largest panel and is a NODE feature that needs
  no graph.
- `own_vshock` as a plain node feature is ≤ +0.45% everywhere and negative on VN100 h1.
- `ALL` (eight features together) is frequently **lower** than the best single feature and turns
  negative at several horizons (VN30 h10 −0.68%, S&P 500 h5 −0.53%) — multicollinearity/overfit from
  pooling many correlated columns; the per-feature columns are the fair read.

## S6 — sector graph: BLOCKED
`find data -iname "*sector*" -o -iname "*gics*" -o -iname "*constituent*"` returns nothing. No
historically-valid, point-in-time sector/GICS classification exists in the repo. A static present-day
sector map would be a look-ahead (survivorship/reclassification) contaminant, and fetching one online
is out of scope. S6 is therefore not run — BLOCKED pending a historically-valid sector source.

## Verdict
- **Volume-shock spillover (S5) does not clear the bar.** Best structural-over-placebo effect is
  +0.72% (VN100 h5); everywhere else ≤ +0.5%, near/below placebo, horizon-isolated, sign-unstable
  across panels. No graph configuration adds ≥ ~1% above placebo consistently across horizons.
- **Richer node features do not deliver a robust lift beyond HAR.** The >1% cases are h1-only
  contemporaneous level co-movements (market/own return, market pk); they vanish at longer horizons.
  The one cross-horizon-consistent feature, `vol_of_vol`, tops out at ~1% only on S&P 500 and is a
  node feature independent of any graph. The pooled `ALL` set does not stack (often negative).
- **The base null is robust to richer features and to volume.** HAR-only node features are only
  marginally thin (a small, consistent vol-of-vol contribution on the largest panel); adding volume,
  returns, market factors, or a volume-shock graph does not produce a material (>~1% above placebo,
  cross-horizon) OOS improvement on VN30, VN100, or S&P 500. No graph/spillover family qualifies for
  promotion to a corrected GAT.
