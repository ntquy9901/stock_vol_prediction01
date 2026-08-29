# Requirements — DY (2014) volatility-spillover graph edge for the HNX GAT model

## Objective
Construct a DIRECTED weighted volatility-connectedness adjacency following Diebold & Yilmaz (2014,
"On the network topology of variance decompositions", *J. Econometrics* 182(1):119-134) EXACTLY, and
test it as the GAT graph edge for the HNX volatility model, against (a) the no-graph LSTM and (b) the
shipped statistical directed vol->PK edge (and the sector-GAT if present). Same MaskedRichNet /
HAR-X / masked panel / per-ticker scalers / QLIKE evaluation — only the EDGE differs.

## Method (source of truth)
Row-normalised generalized-FEVD connectedness matrix on the TRAIN Parkinson-variance panel:
1. VAR(p) on the (stationary) volatility series: `x_t = sum_i Phi_i x_{t-i} + eps_t`, `Sigma = cov(eps)`;
   invertible MA(inf) form `x_t = sum_h A_h eps_{t-h}`.
2. Generalized FEVD (Pesaran-Shin 1998, as DY 2014 Eq. use):
   `theta_ij(H) = (sigma_jj^{-1} * sum_{h=0}^{H-1} (e_i' A_h Sigma e_j)^2) / (sum_{h=0}^{H-1} (e_i' A_h Sigma A_h' e_i))`.
3. Row-normalise: `theta_tilde_ij(H) = theta_ij(H) / sum_k theta_ik(H)` (rows sum to 1). This is the
   directed connectedness network; used directly as adjacency `A[i,j]` = edge j->i (fraction of i's
   H-step FEV due to shocks in j).
4. High-dimensionality (HNX N~154): full unregularised VAR is ill-posed -> elastic-net / LASSO VAR
   (Demirer, Diebold, Liu & Yilmaz 2018, *J. Applied Econometrics* 33(1):1-15). Chosen: VAR(1),
   elastic-net (l1_ratio=0.5), on per-ticker z-scored train series, FEVD horizon H=10. Estimated on
   TRAIN rows only; frozen for val/test.

## Inputs / Outputs
- Input: screened HNX Parkinson-variance processed panel (train rows only), node order = D.tickers.
- Output: `[N,N]` float32 adjacency (Top-K spillover sources per row + self-loop=1.0, matching the
  vol2pk/sector convention so WeightedGATLayer consumes it unchanged); result JSON + report.

## Success criteria / go-no-go
- Connectedness builder: rows sum to ~1 (pre self-loop); directed/asymmetric; finite; train-only.
- Unit tests + CPU smoke forward pass pass; pre-push gate (C0=100 / C1>=95 / ruff-F) passes.
- Experiment: HNX h1, 10 epochs, >=3 seeds if feasible; report all 5 metrics (mean+/-std) +
  date-clustered DM: DY-GAT vs no-graph, DY-GAT vs stat vol2pk (and vs sector-GAT if results exist).
- Verdict: does the DY spillover edge significantly improve QLIKE over no-graph and over the
  statistical edge on HNX? (Report the honest answer, positive or negative.)

## Non-goals / constraints
- Do NOT edit live-training-path files (`baselines/2026-08-21_har_anchored_residual/code/*`,
  `scripts/eda/*`, `submission/soict_lstm_gat/*`) — import read-only. All new code lives here.
- GPU heavily shared: CPU-forced by default (hide CUDA before torch import). No killing processes.
