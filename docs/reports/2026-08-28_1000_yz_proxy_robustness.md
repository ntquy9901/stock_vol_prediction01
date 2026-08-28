# Volatility-proxy robustness (Parkinson vs Yang-Zhang) — paper data (2026-08-28)

Question from the estimator study: is the model ranking robust to the choice of daily volatility proxy?
Re-ran the FULL pipeline (HAR-X / LSTM / LSTM+GAT, 5 seeds, same masked panel/universe/protocol) with the
target set to Parkinson vs a per-day Yang-Zhang variance, written to a SEPARATE tree
`results/masked_rich_yz/<target>/` (delivered `masked_rich_floor1e2` untouched via the new `run(out_subdir=...)`).

Command: `.venv_gpu_encode/Scripts/python.exe scripts/eda/run_yz_robustness.py --panels vn30 vn100 hnx hose --targets parkinson yz_daily`

## Result (5-seed-mean QLIKE; lowest per row-block in bold in the paper)
| Panel | target | HAR-X | LSTM | LSTM+GAT | lowest |
|---|---|---|---|---|---|
| VN100 | Parkinson | **0.512** | 0.621 | 0.552 | HAR-X |
| VN100 | Yang-Zhang | **0.527** | 0.640 | 0.540 | HAR-X |
| VN30 | Parkinson | **0.516** | 0.679 | 0.619 | HAR-X |
| VN30 | Yang-Zhang | **0.722** | 0.775 | 0.777 | HAR-X |
| HOSE | Parkinson | **1.244** | 1.310 | 1.355 | HAR-X |
| HOSE | Yang-Zhang | **1.194** | 1.527 | 1.375 | HAR-X |
| HNX | Parkinson | 1.872 | 1.821 | **1.816** | LSTM+GAT |
| HNX | Yang-Zhang | 1.635 | 1.650 | **1.623** | LSTM+GAT |

**The lowest-QLIKE model is identical under both proxies on every panel** (HAR-X on VN30/VN100/HOSE; LSTM+GAT
on HNX) → the headline ranking is robust to the volatility proxy. YZ lowers absolute QLIKE on the less-liquid
HOSE/HNX (overnight captures near-zero-range days) but raises MSE ~3-4x; the small graph-vs-LSTM QLIKE contrast
is proxy-dependent (sig at VN30 under Parkinson, not under YZ; reverses at HOSE) — consistent with it lying
within seed noise. Added to `docs/paper/soict_harlstmgat_extended.tex` as Section "Sensitivity to the
volatility proxy" (Table `tab:proxy`), 12 pages, 24/24 cells verified vs result.json.

Caveats: per-day YZ (indicator form, not windowed academic YZ); overnight winsorized ±0.20 (raw prices not
split-adjusted); S&P 500 omitted (each cell ~42 min exceeds the background wall-clock; VN panels cover the
paper's scope). Framing: Parkinson stays the PRIMARY target; YZ is a robustness check, not a main-result swap.
