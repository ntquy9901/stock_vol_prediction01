# Results summary (numbers from the result JSONs)

Pooled QLIKE (lower = better), 5 seeds, 22 folds, lookback 22, target = Parkinson variance at t+h.
Best model per row in **bold**. DM = date-clustered Diebold–Mariano p-value.

## VN30 (31 nodes)
| h | HAR | HAR-X | LSTM | VolGA | Best | DM VolGA−LSTM (graph) | DM VolGA−HAR-X |
|--|--|--|--|--|--|--|--|
| 1 | **0.4952** | 0.4967 | 0.5219 | 0.5146 | HAR | 0.179 (n.s.) | 0.157 (n.s.) |
| 5 | 0.5944 | **0.5937** | 0.6327 | 0.6150 | HAR-X | 0.112 (n.s.) | 0.397 (n.s.) |
| 10 | 0.6412 | 0.6394 | **0.6336** | 0.6485 | LSTM | 0.265 (n.s.) | 0.717 (n.s.) |
| 22 | 0.7012 | **0.6987** | 0.7056 | 0.7053 | HAR-X | 0.928 (n.s.) | 0.788 (n.s.) |

## VN100 (102 nodes)
| h | HAR | HAR-X | LSTM | VolGA | Best | DM VolGA−LSTM (graph) | DM VolGA−HAR-X |
|--|--|--|--|--|--|--|--|
| 1 | 0.4983 | 0.5004 | 0.5025 | **0.4916** | VolGA | **0.008 (sig, VolGA)** | 0.177 (n.s.) |
| 5 | 0.5671 | **0.5610** | 0.5763 | 0.5705 | HAR-X | **0.011 (sig, VolGA)** | 0.585 (n.s.) |
| 10 | 0.6005 | **0.6001** | 0.6096 | 0.6149 | HAR-X | 0.229 (n.s.) | 0.520 (n.s.) |
| 22 | 0.6392 | **0.6388** | 0.6479 | 0.6434 | HAR-X | 0.107 (n.s.) | 0.842 (n.s.) |

## Reading (honest)
- **No deep model beats the HAR/HAR-X econometric baseline on QLIKE at any horizon on either
  market** — DM VolGA-vs-HAR-X is n.s. everywhere. HAR/HAR-X win the point QLIKE at most horizons.
- **The graph's marginal value (VolGA − no-graph LSTM) is significant only on VN100 at h1/h5**
  (p=0.008, 0.011) and null on VN30 at every horizon — despite VN30 having higher pairwise
  correlation. ⇒ graph value tracks **node breadth × liquidity × stable edges**, not correlation
  magnitude.
- **Loss-dependent:** the dashboards report DM on QLIKE / squared-error / absolute-error; the graph
  reaches significance on SE/AE at short horizons on VN30 even where QLIKE says n.s. — so the
  marginal-value verdict is reported on all three bases, not QLIKE alone.

## Pooled/transfer VN30 ablation (COMPLETE — all 4 horizons)
Single VN100 panel; Arm 0 trains 31 VN30, Arm 1 trains 102; both score the identical 31 VN30 OOS
points. Headline = paired DM Arm1-vs-Arm0 (A = pooling better; * = p<0.05). VolGA row:
| h | QLIKE Arm0 | QLIKE Arm1 | DM QLIKE | DM SE | DM AE |
|--|--|--|--|--|--|
| 1 | 0.4690 | 0.4789 | 0.147 (B) | 0.172 (A) | **0.000* (A)** |
| 5 | 0.5779 | 0.5809 | 0.666 (B) | 0.306 (A) | **0.000* (A)** |
| 10 | 0.6231 | 0.6308 | 0.405 (B) | 0.506 (A) | **0.001* (A)** |
| 22 | 0.6942 | 0.6722 | 0.263 (A) | **0.021* (A)** | 0.117 (A) |
- Full 4-horizon verdict: widening the training universe does **not** significantly improve VN30
  QLIKE (VolGA n.s. at every horizon; LSTM significant only at h10) — the QLIKE ceiling holds,
  matching the prior Track B A1 null. It **significantly reduces absolute error at h1/h5/h10** (both
  deep models), fading by h22 (where SE becomes significant instead). Net: the benefit of pooling is
  **loss- and horizon-dependent** — a point-error gain at short-to-medium horizons that does not
  translate into a QLIKE gain. Verified against `results/pooled_transfer_vn30/pooled_vn30_h{1,5,10,22}.json`.
  Caveat: Arm 0 uses the VN100 grid + VN100 market factor, so it is NOT the standalone-VN30 run; only
  Arm0-vs-Arm1 is a valid comparison. Dashboard: `docs/reports/2026-09-04_pooled_transfer_vn30_dashboard.html`.
- **Caveat:** Arm 0 ≠ the standalone-VN30 run above (Arm 0 uses the VN100 grid + VN100 `market_pk`).
  Only Arm 0 vs Arm 1 is a valid comparison; do not cross-compare to the standalone tables.

## Data / metric caveats to carry into the paper
- Raw VN prices are **not split/dividend-adjusted**; overnight-based estimators inherit corporate-
  action jumps (see `docs/reports/appendix/2026-09-03_overnight_tail_appendix.md`). Parkinson
  (intraday H/L) is split-invariant and is the chosen target/estimator.
- Target is variance σ² (not σ). QLIKE uses a positivity floor from the canonical config;
  floor-sensitivity on high-zero-range markets (HNX) is documented.
