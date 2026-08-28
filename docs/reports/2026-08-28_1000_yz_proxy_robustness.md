# Volatility-proxy robustness (Parkinson vs Yang-Zhang (indicator-form per-day proxy, NOT standard windowed YZ)) — paper data (2026-08-28)

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

## S&P 500 (5-seed, updated)
| target | HAR-X | LSTM | LSTM+GAT | lowest |
|---|---|---|---|---|
| Parkinson (5 seed) | **0.362** | 0.638 | 0.543 | HAR-X |
| Yang-Zhang (5 seed) | 0.565 | **0.553** | 0.562 | LSTM (tie) |

5-seed (each cell ~36 min). Under Parkinson HAR-X is clearly lowest (gap ~0.18); under Yang-Zhang the three models fall within 0.013 (HAR-X 0.565, LSTM 0.553, GAT 0.562) = statistically indistinguishable (within per-seed dispersion), not a clean re-ranking. Added to the robustness table in both papers. The single-seed run had shown GAT lowest (0.549) = seed noise; 5-seed corrects it. Reported honestly as a caveat in the paper.

## (c) Split-adjusted-price PoC — S&P 500 is ALREADY adjusted
The overnight artifacts that inflated Yang--Zhang (540% "gaps", zero prior closes) were on the Vietnamese
(vnstock) data, not the S&P 500. Verified: the bundled S&P 500 OHLCV **matches Yahoo's split/dividend-adjusted
close** exactly — ratio `myraw/yf_Adj_Close` = 1.0000 with 0.0000 std and corr 1.00000 for AAPL/MSFT/NVDA over
2015--2022, and the series is smooth across known splits (e.g. AAPL 4:1 on 2020-08-31: 120.96 -> 125.06, no 4x
jump; it equals Yahoo Adj Close 120.96/125.06, not raw Close 124.81/129.04). So the S&P 500 Yang--Zhang result
is on clean adjusted prices and needs no adjustment (the winsorization is belt-and-suspenders there, catching
only the 2000 AAPL-crash day). The unadjusted-price problem is specific to the Vietnamese OHLCV, where a fully
adjusted comparison is the real future-work item (vnstock adjusted-price availability is uncertain). Paper
caveat (extended + crossmarket) updated to state this. yfinance in base python3; check is CPU/network, run in
parallel with the (b) 5-seed S&P 500 GPU run.
