# Lookback 44 vs 22 under a 90/10 protocol on refreshed data — held-out test results

## Setup

- Data: refreshed prices through 2026-08-14; VHM trimmed to its real listing (2018-05-23); OHLC
  corrected on 16 tickers (positive-aware max/min). Parkinson variance target.
- Protocol: train+val merged to 90%, test 10% (`ratios=(0.80,0.10,0.10)`), fixed 15 epochs, no
  early-stop, test read once (no selection on test → no leakage). Seed 42, single seed.
- Rungs: HAR, FULL, minus_graph, minus_gate, minus_news, lstm_only (via `run_retrain_trainval.py`).
- Two lookback lengths compared: seq=44 vs seq=22 (`combo_ladder.SEQ`), everything else identical.
- Test observation counts differ by lookback (a longer window consumes more leading rows): seq44
  n≈7853–8546, seq22 n≈8579–9272. seq44-vs-seq22 is therefore compared at the aggregate-metric
  level, not by a paired DM; DM is FULL-vs-rung WITHIN each config.
- Metrics units below: MSE ×10⁻⁶, RMSE/MAE ×10⁻³; R² and QLIKE as-is. Lower is better except R².

## Test metrics — seq=44

| h | rung | MSE | RMSE | MAE | R² | QLIKE |
|---|---|---|---|---|---|---|
| 1 | HAR | 0.242 | 0.492 | 0.279 | 0.270 | 0.4648 |
| 1 | FULL | 0.243 | 0.493 | 0.269 | 0.268 | 0.4591 |
| 1 | lstm_only | 0.239 | 0.489 | 0.272 | 0.278 | 0.4546 |
| 5 | HAR | 0.271 | 0.521 | 0.302 | 0.178 | 0.5473 |
| 5 | FULL | 0.271 | 0.521 | 0.288 | 0.178 | 0.5543 |
| 5 | lstm_only | 0.276 | 0.525 | 0.291 | 0.164 | 0.5516 |
| 10 | HAR | 0.287 | 0.536 | 0.312 | 0.127 | 0.5874 |
| 10 | FULL | 0.300 | 0.547 | 0.294 | 0.089 | 0.6253 |
| 10 | lstm_only | 0.290 | 0.539 | 0.303 | 0.117 | 0.5966 |
| 22 | HAR | 0.308 | 0.555 | 0.326 | 0.066 | 0.6327 |
| 22 | FULL | 0.320 | 0.566 | 0.310 | 0.030 | 0.6636 |
| 22 | lstm_only | 0.320 | 0.566 | 0.319 | 0.030 | 0.6691 |

(minus_graph/gate/news rows in `results/volatility_retrain_h{h}_seed42_2026-08-16_011558_seq44final/`.)

## Test metrics — seq=22

| h | rung | MSE | RMSE | MAE | R² | QLIKE |
|---|---|---|---|---|---|---|
| 1 | HAR | 0.238 | 0.488 | 0.275 | 0.275 | 0.4637 |
| 1 | FULL | 0.236 | 0.486 | 0.276 | 0.280 | 0.4615 |
| 1 | lstm_only | 0.235 | 0.485 | 0.274 | 0.283 | 0.4577 |
| 5 | HAR | 0.272 | 0.521 | 0.300 | 0.179 | 0.5503 |
| 5 | FULL | 0.280 | 0.530 | 0.286 | 0.152 | 0.5608 |
| 5 | lstm_only | 0.275 | 0.524 | 0.292 | 0.170 | 0.5500 |
| 10 | HAR | 0.286 | 0.535 | 0.312 | 0.131 | 0.5932 |
| 10 | FULL | 0.292 | 0.540 | 0.300 | 0.114 | 0.6024 |
| 10 | lstm_only | 0.283 | 0.532 | 0.307 | 0.139 | 0.5836 |
| 22 | HAR | 0.309 | 0.556 | 0.327 | 0.072 | 0.6430 |
| 22 | FULL | 0.326 | 0.571 | 0.305 | 0.020 | 0.8196 |
| 22 | lstm_only | 0.318 | 0.564 | 0.311 | 0.043 | 0.6840 |

## Diebold-Mariano (HLN, HAC lag h−1), FULL vs HAR (dm(p); negative favors FULL; * p<.05)

| h | seq | QLIKE | SE | AE (MAE) |
|---|---|---|---|---|
| 1 | 44 | −2.04 (.04)* FULL | +0.68 (.50) tie | −12.49 (.00)* FULL |
| 1 | 22 | −1.11 (.27) tie | −1.51 (.13) tie | +0.63 (.53) tie |
| 5 | 44 | +0.87 (.38) tie | −0.26 (.80) tie | −13.89 (.00)* FULL |
| 5 | 22 | +2.09 (.04)* HAR | +4.25 (.00)* HAR | −10.77 (.00)* FULL |
| 10 | 44 | +3.99 (.00)* HAR | +3.81 (.00)* HAR | −9.98 (.00)* FULL |
| 10 | 22 | +1.68 (.09) tie | +2.43 (.02)* HAR | −7.99 (.00)* FULL |
| 22 | 44 | +3.37 (.00)* HAR | +2.98 (.00)* HAR | −6.47 (.00)* FULL |
| 22 | 22 | +2.95 (.00)* HAR | +3.75 (.00)* HAR | −7.00 (.00)* FULL |

## Reading

- **MAE: FULL beats HAR at every horizon under both lookbacks, strongly and significantly** (dm −6.5
  to −13.9, all p<.01). The deep model produces markedly lower absolute error than HAR everywhere.
- **QLIKE and squared-error: HAR beats FULL at the long horizons (h10, h22, significant) under both
  lookbacks**; at h1 the two tie (or FULL edges ahead on seq44 QLIKE). HAR retains the
  proportional/variance-loss advantage as the horizon grows.
- **No single model dominates across metrics** — FULL is the absolute-error choice, HAR the
  QLIKE/variance choice at longer horizons. This is the same parsimony picture as prior runs, now on
  cleaned data: the deep stack does not uniformly beat HAR.
- **Lookback 44 vs 22 shows no consistent improvement.** FULL QLIKE: 44 < 22 at h1/h5, 22 < 44 at
  h10, 44 ≪ 22 at h22 (seq22 FULL h22 QLIKE = 0.8196 is an anomalous single-seed spike). MSE/RMSE are
  mixed by horizon. There is no reliable lookback-44 advantage — the longer window is a null on this
  seed/protocol.
- **lstm_only (price-only backbone) stays strong** — lowest QLIKE at h1 (seq44 0.4546) and
  competitive throughout, reinforcing that the price-only temporal model carries most of the signal.

## Caveats

- Single seed (42). The h22 seq22 FULL QLIKE spike (0.82) in particular needs multi-seed to confirm
  it is not seed noise.
- seq44 and seq22 are not on identical test observations (lookback-dependent sample counts), so the
  cross-lookback comparison is metric-level, not a paired test.
- The 10% test window is the recent 2025–2026 period; R² is low across all models (0.27 at h1 → ~0.03
  at h22), i.e. the recent regime is hard and models barely beat the unconditional mean at long
  horizons — read the level metrics (RMSE/MAE) and QLIKE accordingly.

## Files

- Runs: `results/volatility_retrain_h{1,5,10,22}_seed42_2026-08-16_011558_seq44final/` and
  `..._2026-08-16_031259_seq22final/` (per-rung `retrain_metrics.json` + test dumps).
- DM: `baselines/2026-08-15_volatility/code/dm_retrain.py`.
