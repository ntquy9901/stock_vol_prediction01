# Graph WaveNet ablation on HNX daily volatility (horizon 1)

Date: 2026-08-29. Panel: HNX (154 nodes after the delivered liquidity screen). Horizon: 1. Seeds:
{42, 123, 2026}, 10 epochs (early stop). Device: GPU (RTX 4060). Result:
`results/graphwavenet_ablation/graphwavenet_ablation_hnx_h1.json`. Code:
`baselines/2026-08-29_graphwavenet_ablation/`.

## Question
The delivered 4-edge sweep (statistical correlation, sector, MTGNN-learned, DY-spillover) found no graph
edge beats a no-graph LSTM on HNX h1, with every edge bolted onto the same LSTM temporal branch. This
baseline asks whether a different temporal + spatial architecture — Graph WaveNet: a dilated-causal-conv
(TCN) backbone plus a self-adaptive adjacency — changes that picture. Because Graph WaveNet replaces the
whole temporal + spatial stack (TCN, not LSTM), the GWN-vs-LSTM/HAR comparison is a temporal-backbone
comparison, not a leave-one-out edge ablation. The clean in-family test is the paper's own ablation:
GWN with the self-adaptive graph vs the same GWN with the graph removed (pure TCN).

## Paper fidelity (arXiv:1906.00121; nnzhan/Graph-WaveNet `model.py`)
Reproduced components with the equation → code mapping (verified against the paper and the official code,
2026-08-29; full mapping in `code/gwn_model.py` docstring):

| paper / official code | this module |
|---|---|
| self-adaptive adj `adp = softmax(relu(mm(nodevec1, nodevec2)), dim=1)` = `SoftMax(ReLU(E1·E2ᵀ))` | `GraphWaveNet.adaptive_adjacency()` |
| graph propagation `nconv` `einsum('ncvl,vw->ncwl')` (+ batched `'ncvl,nvw->ncwl'` for the node mask) | `NConv` |
| order-K diffusion `gcn` (concat `[x, Ax, A²x]` → 1×1 `linear`), `c_in=(order·support_len+1)·c_in` | `GCN` |
| dilated causal conv + gated activation `tanh(·)⊙sigmoid(·)`, residual + skip, dilations 1,2,1,2 reset/block | `GraphWaveNet.forward` |
| head `relu(skip)→relu(end_conv_1)→end_conv_2` | `GraphWaveNet.forward` |

Receptive field 13 (1 + 4·3); seq=10 < 13 → causal left-pad (official behaviour). An independent test
recomputes `softmax(relu(E1@E2), dim=1)` in numpy from the raw parameters (not reusing the module forward)
and matches the module (CLAUDE.md named-formula rule).

Deliberate, documented deviations (faithful part unchanged): `out_dim=1` (single h1 target), `in_dim=5`
(delivered node vector), channel widths reduced from the traffic defaults (skip 256→64, end 512→128) for
8 GB VRAM and the smaller daily-vol panel, and no predefined support (this panel has no physical graph, so
`GWN_adaptive` uses only the self-adaptive graph and `GWN_no_adaptive` removes it → pure TCN). `train_gwn`
mirrors the delivered `train_masked_rich` `zscore_floor` path (standardized target, masked-MSE, Adam +
ReduceLROnPlateau, grad clip, early stop, 1e-2·mean floor) on the same masked-union folds, scalers and QLIKE
floor as every sibling baseline; only the network differs.

## Test-set metrics (60,028 obs, 477 dates, 154 nodes)
Ensemble = metric of the seed-averaged prediction; per-seed mean = mean of seed-level metrics (± std).

| model | MSE | RMSE | MAE | QLIKE (ens) | QLIKE (per-seed mean ± std) | R² |
|---|---|---|---|---|---|---|
| LSTM (no graph) | 1.373e-06 | 0.001172 | 0.000605 | **1.8063** | 1.8123 ± 0.0021 | 0.232 |
| LSTM + wGAT (vol→PK) | 1.376e-06 | 0.001173 | 0.000606 | 1.8091 | 1.8133 ± 0.0066 | 0.230 |
| GWN_adaptive | 1.376e-06 | 0.001173 | 0.000606 | 1.8128 | 1.8260 ± 0.0096 | 0.230 |
| GWN_no_adaptive (pure TCN) | 1.378e-06 | 0.001174 | 0.000606 | 1.8139 | 1.8168 ± 0.0030 | 0.229 |
| HAR | 1.416e-06 | 0.001190 | 0.000652 | 1.8284 | — | 0.208 |
| HAR-X | 1.402e-06 | 0.001184 | 0.000646 | 1.8615 | — | 0.215 |

All six fit verdicts are `ok` (no over/under-fit): val→test QLIKE gap ≈ −0.16 (test slightly better than
val) and train→test R² drop ≈ −0.02 for every model (`fit_diagnostics` in the JSON; the pre-push
over/under-fit gate passes on this result).

## Date-clustered Diebold–Mariano (QLIKE primary; SE secondary)
`favors A` means the first-named model has the lower loss.

| comparison | QLIKE p | favors | mean diff | SE p | favors |
|---|---|---|---|---|---|
| GWN_adaptive vs GWN_no_adaptive | 0.355 | A (adaptive) | −0.0012 | 0.0072 | A (adaptive) |
| GWN_adaptive vs LSTM | 0.0044 | B (LSTM) | +0.0061 | 0.080 | B (LSTM) |
| GWN_no_adaptive vs LSTM | 6.9e-05 | B (LSTM) | +0.0073 | 0.00025 | B (LSTM) |
| GWN_adaptive vs HAR | 0.0038 | A (GWN) | −0.0155 | 1.8e-08 | A (GWN) |
| GWN_adaptive vs HAR-X | 4.9e-08 | A (GWN) | −0.0487 | 2.8e-09 | A (GWN) |
| GWN_no_adaptive vs HAR | 0.0032 | A (GWN) | −0.0143 | 6.9e-07 | A (GWN) |

## Findings
1. **The self-adaptive adjacency does not produce a robust volatility-forecast improvement on HNX h1.**
   Within the identical TCN backbone, adding the self-adaptive graph lowers the ensemble QLIKE by 0.0012
   (0.07%), which is not significant under date-clustered DM (QLIKE p=0.355). On the squared-error basis the
   difference reaches significance (SE p=0.0072, mean diff −2.9e-09), but on the per-seed mean the sign
   reverses (GWN_adaptive 1.8260 vs GWN_no_adaptive 1.8168), so the QLIKE effect is within seed variation and
   basis-dependent rather than a stable gain. This is consistent with the four prior graph edges (statistical,
   sector, MTGNN-learned, DY-spillover), all null on this cell — the MTGNN learned adjacency, the closest
   prior art (same author family), was likewise null (DM vs no-graph p=0.228).
2. **The Graph WaveNet temporal backbone does not beat the no-graph LSTM.** The no-graph LSTM has the lowest
   QLIKE (1.8063), below both GWN_adaptive (1.8128, DM p=0.0044 in the LSTM's favour) and GWN_no_adaptive
   (1.8139, DM p=6.9e-05). Swapping the LSTM temporal branch for the WaveNet TCN did not improve the forecast.
3. **All learned models remain above HAR/HAR-X on QLIKE.** Both GWN variants and both LSTM variants beat HAR
   and HAR-X with significant DM margins (e.g. GWN_adaptive vs HAR QLIKE p=0.0038), matching the prior finding
   that deep models improve on the linear HAR baseline on this panel.

## Conclusion
On HNX daily volatility at horizon 1, neither the self-adaptive adjacency nor the WaveNet TCN backbone
overturns the no-graph LSTM: the self-adaptive graph adds no robust QLIKE improvement within GWN (DM p=0.355,
per-seed sign reversal), and the GWN backbone forecasts slightly worse than the LSTM (DM p≤0.004). This is a
fifth independent robustness result on the same cell — a distinct temporal + spatial architecture, learned
graph included, does not change the outcome that the graph mechanism adds no out-of-sample value for HNX
daily volatility, while all deep models continue to beat HAR.

## Caveats
- **BatchNorm over zero-padded nodes** (3-lens review, MAJOR, documented): the official `gwnet` uses
  `BatchNorm2d`, whose statistics pool over `[B, N, T]`; the masked-union panel zero-fills invalid nodes, so
  BN normalizes valid nodes using batch statistics that include the 18% of node-slots that are invalid
  (`valid_node_fraction_test = 0.817`). Kept for architecture fidelity (not swapped for a mask-aware norm).
  It is common-mode across both GWN variants → cancels in the headline in-family adaptive ablation; it does
  not cancel vs LSTM/HAR (no BN), so the GWN-vs-LSTM/HAR comparison is a backbone comparison and the valid
  fraction is reported to bound it.
- GWN uses batch 64 vs the LSTM/GAT batch 16 (a per-architecture optimization hyperparameter, recorded in
  the JSON); channel widths reduced from the paper's traffic defaults (documented above).
- Single panel (HNX) and single horizon (h1), matching the 4-edge sweep's headline cell; not a
  multi-panel/multi-horizon claim.
