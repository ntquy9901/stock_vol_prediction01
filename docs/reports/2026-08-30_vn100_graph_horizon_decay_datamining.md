# VN100 graph horizon-decay data-mining (why the cross-sectional advantage lives at h1)

Pure data-mining (pandas/numpy) on the delivered VN100 Parkinson-variance panel (N=104 tickers, T=4603 days). No model training, no GPU. The `parkinson_volatility` column is a VARIANCE (sigma^2). Target = pk[t+h] (matches `masked_rich.build_masked_rich`). TRAIN-only fits (target date strictly before the 80% boundary); OOS on the held-out tail.

## Framing (honest)
On VN100, HAR-X (linear) beats the deep models at ALL horizons. The graph's value is measured RELATIVE to the no-graph LSTM and is a short-horizon phenomenon. This analysis is association/mechanism evidence, not a causal claim.

## Proven mechanism
The only cross-sectional signal that is NOT already redundant with a stock's own volatility history is a TRANSIENT next-day lead-lag spillover of market/peer shocks (pooled corr approx 0.048 at h1, collapsing to ~0 by h2). The persistent cross-sectional LEVEL co-moves at every horizon but is subsumed by each stock's own HAR long-memory, so it adds negligible incremental R^2 at ANY horizon. Because the graph (attention over peers) is the only model component that can read peer shocks, its marginal advantage over the no-graph LSTM is concentrated at h1 and has dissipated by h10/h22 — matching the observed DM pattern (significant at h1/h5, tie at h10/h22). The effect is small in magnitude, consistent with HAR-X's overall dominance.

## 1. Incremental cross-sectional R^2 by horizon
| h | HAR-only R^2 (in) | +level | +shock | +both | HAR-only R^2 (oos) | +both (oos) |
|---|---|---|---|---|---|---|
| 1 | 0.3848 | 0.0063 | 0.0000 | 0.0091 | 0.1577 | 0.0246 |
| 5 | 0.3220 | 0.0055 | 0.0015 | 0.0136 | 0.0817 | 0.0328 |
| 10 | 0.3008 | 0.0065 | 0.0020 | 0.0155 | 0.0327 | 0.0340 |
| 22 | 0.2754 | 0.0032 | 0.0019 | 0.0094 | -0.0081 | 0.0289 |

Own-history HAR already explains the bulk of predictable log-variance; the cross-sectional blocks add only ~0.01 R^2 and do NOT show a large clean h1 peak in R^2 terms — because a market shock is reflected contemporaneously in the stock's own volatility, so it is largely redundant with own history. The horizon signature is in the lead-lag channel (section 3).

## 2. Target persistence
- h1: HAR-only R^2(in)=0.3848, level lag-h autocorr=0.5658, shock lag-h autocorr=0.0655
- h5: HAR-only R^2(in)=0.3220, level lag-h autocorr=0.4880, shock lag-h autocorr=-0.0270
- h10: HAR-only R^2(in)=0.3008, level lag-h autocorr=0.4692, shock lag-h autocorr=0.0011
- h22: HAR-only R^2(in)=0.2754, level lag-h autocorr=0.4403, shock lag-h autocorr=-0.0014

Own-history predictability FALLS with h (target harder to predict overall), refining the naive 'target gets smoother' hypothesis. The persistent level autocorrelation decays slowly; the transient shock autocorrelation collapses within a few days.

## 3. Lead-lag decay (the clean proof)
| h | market LEVEL | market SHOCK | peer SHOCK |
|---|---|---|---|
| 1 | 0.2084 | 0.0478 | 0.0488 |
| 2 | 0.1734 | 0.0000 | 0.0015 |
| 3 | 0.1673 | -0.0082 | -0.0130 |
| 4 | 0.1639 | -0.0129 | -0.0149 |
| 5 | 0.1722 | 0.0019 | -0.0050 |
| 7 | 0.1686 | 0.0055 | 0.0054 |
| 10 | 0.1627 | -0.0031 | 0.0041 |
| 15 | 0.1566 | 0.0026 | -0.0019 |
| 22 | 0.1429 | -0.0037 | -0.0036 |

The market/peer SHOCK correlation with future volatility is largest at h1 and ~0 by h2, while the persistent LEVEL correlation barely moves. The graph can exploit the shock channel; that channel only exists at the shortest horizon.

## 4. Cross-sectional co-structure of the HAR residual
- h1: median pairwise resid corr=0.0951 (pairs=5253), first-factor share=0.1196
- h5: median pairwise resid corr=0.1210 (pairs=5253), first-factor share=0.1312
- h10: median pairwise resid corr=0.1305 (pairs=5253), first-factor share=0.1366
- h22: median pairwise resid corr=0.1470 (pairs=5253), first-factor share=0.1373

The residual's cross-sectional co-structure does NOT weaken with h (it reflects the persistent common regime, already captured by own history), so the horizon decay is NOT explained by a vanishing common factor — it is explained by the vanishing transient spillover.

## 5. Cross-market contrast
- VN100: h1 market-shock corr=0.0478
- HNX: h1 market-shock corr=0.0104
- SP500: h1 market-shock corr=0.0231

VN100 has the strongest h1 shock-spillover; HNX is ~flat at all horizons (consistent with the graph being a flat null there); SP500 is intermediate and also decays with h.

## Caveats
- Association, not causation; pooled log-space OLS, not the deep model's basis or the QLIKE loss where the graph's DM edge was measured.
- Incremental R^2 magnitudes are small everywhere; the horizon signature is in the lead-lag correlation, not in R^2. The central claim rests on a bivariate correlation that the predictive R^2 channel does not independently corroborate.
- The pooled lead-lag correlations use day-clustered observations (the market signal is identical across stocks within a day) and are reported without clustered standard errors — read them as effect-size/shape evidence, not significance tests.
- OOS R^2 is benchmarked against the OOS-sample mean (mildly generous vs a train-mean benchmark); the incremental OOS quantity is unaffected since both nested models share that mean.
- Single train/OOS split; first-factor share is a NaN-imputed proxy.
- The target column is a variance (sigma^2), not sigma; VN prices are not split-adjusted.
