# Design — DY (2014) spillover-edge GAT ablation (HNX)

## Data flow
1. `run_dy_ablation.build_panel_masked(hnx, cfg, h)` reuses the read-only estimator writer + masked
   panel builder (identical to the sector ablation) -> `MaskedRichData D` (node order = `D.tickers`,
   `D.adj_vol2pk` = shipped statistical edge).
2. `dy_connectedness.train_vol_panel(files, D.tickers, D.d_va[0])` rebuilds the wide Parkinson-variance
   panel aligned to `D.tickers`, keeps only rows strictly before the first VALIDATION target date
   (leakage-safe train window; frozen).
3. `dy_connectedness.build_dy_adjacency(train_panel, p=1, H=10, alpha, l1_ratio, top_k=5)`:
   z-score per ticker (train stats) -> elastic-net VAR(1) equation-by-equation -> `Phi_1`, residual
   `Sigma` -> VMA `A_0..A_{H-1}` -> generalized FEVD `theta` -> row-normalise `theta_tilde` ->
   Top-K spillover sources per row + self-loop=1.0 -> `[N,N]` float32 adjacency (edge j->i).
4. `_train_variant(D, cfg, use_graph=True, adj=dy_adj)` trains MaskedRichNet with the DY edge;
   compare against `stat_GAT_vol2pk` (D.adj_vol2pk), `no_graph_LSTM` (use_graph=False), and (if
   `results/sector_gat_ablation/` exists) the recorded sector-GAT. All 5 metrics + date-clustered DM.

## Key design decisions
- **Convention:** `theta_tilde[i,j]` = fraction of i's FEV from j = edge j->i = `A[target i, source j]`,
  identical to `WeightedGATLayer` / vol2pk convention. Off-diagonal Top-K kept (K=5, same as vol2pk),
  diagonal forced to 1.0 (self-loop) so the WeightedGATLayer masking/attention is unchanged.
- **High-dim fix:** elastic-net VAR(1) per equation (Demirer et al. 2018). z-scoring per ticker makes a
  single penalty meaningful across the tiny-magnitude variance series. VAR(1) + H=10 documented.
- **Leakage:** all VAR estimation on rows strictly before `D.d_va[0]`; frozen for val/test. NaN
  (pre-listing / gaps) imputed by ffill+bfill per column WITHIN the train window only (no val/test).
- **Reporting:** full `theta_tilde` connectedness stats (total connectedness index, avg directional
  degree, row-sum check) recorded alongside the Top-K model adjacency stats.

## SDD gates
- **Simplicity:** one new pure-numpy/sklearn builder + one harness mirroring the sector harness; no new
  abstraction, reuse MaskedRichNet/train/DM/metrics unchanged.
- **Anti-abstraction:** sklearn `ElasticNet` directly; numpy for VMA/FEVD; no wrappers.
- **Performance/Batching:** DY matrix build is CPU/VAR (one-off, per-equation coordinate descent, N
  equations vectorised FEVD). Training reuses `train_masked_rich` (batched `[B,N,...]` tensors, batched
  block adjacency `base * node-mask`, GPU when free) — no batch=1. CPU-forced by default only to avoid
  contending with the shared GPU jobs; batch semantics unchanged.
