# Why the GAT branch never helps volatility forecasting — diagnostic report

Date: 2026-08-21
Scope: `submission/soict_lstm_gat` HAR-LSTM-GAT model. Diagnostics run on VN30, lookback 10, horizon 1
(the config with the largest per-obs QLIKE damage from the graph). Read-only analysis + a small
reproduction run (single seed 42, 12 epochs). No data or model files were modified.

## TL;DR

The GAT branch hurts because it injects a **smoothed, cross-sectional, out-of-sample-meaningless
signal** and then pays for it with **extra head capacity that overfits**. Three mechanisms compound:

1. **Attention has collapsed to uniform averaging** (normalized entropy 0.999; 100% of node-snapshots
   near-uniform). The learned attention weights do not discriminate neighbors at all — the GAT is a
   plain mean over a node's Top-5 partial-correlation neighbors.
2. **The neighbor set it averages over does not transfer out of sample** (train-vs-test Top-5 Jaccard
   0.17; edge-weight correlation 0.19). The frozen train graphical-lasso graph is largely noise
   relative to test-period dependence.
3. **The averaging over-smooths and the wider head overfits** (GAT node-dispersion 0.65× of its input;
   test−train MSE gap 0.22 for GAT vs 0.11 for no-GAT — 2× larger generalization gap; GAT val loss
   bottoms at epoch 8 then rises while no-GAT is still improving at epoch 12).

Net: the graph adds variance, not signal. This reproduces in the quick run (GAT test QLIKE 0.478 vs
no-GAT 0.419) and matches the committed 5-seed result (`results/soict/vn30_lb10_h1/result.json`: GAT
QLIKE 0.4528 vs no-GAT 0.4120, DM favors no-GAT p=1.7e-10).

## What the GAT actually does to the test data (architecture)

- The GAT branch reads only the **raw (normalized) 3 HAR features at the last day t** of the window
  — `node_raw = price[:, :, -1, :]` (`submission/soict_lstm_gat/model.py:72`). It does **not** see the
  temporal window; that is the LSTM branch's job.
- It attends over a **fixed, train-frozen graphical-lasso Top-5 partial-correlation adjacency**
  (`edges.py:51-107`; estimated on TRAIN rows only via `snapshots.py:106` / `run_all.py:51`; Top-5 per
  node at `edges.py:88-99`; masked softmax over neighbors at `model.py:37-40`; neighbor aggregation at
  `model.py:41`).
- Output dimension is `hidden * heads = 64 * 4 = 256` (`model.py:54,56`), concatenated with the 64-dim
  LSTM output, so the head input is **320-dim with the graph vs 64-dim without** (`model.py:57-59`).
  That is the capacity the graph adds.
- Training is pooled MSE loss on the normalized scale (`train.py:47`), evaluated per-ticker
  inverse-transformed (`evaluate.py:22-24`).

So on each test date, every node's GAT contribution is: project its 3 raw HAR numbers, average them
(near-uniformly) with its 5-ish neighbors' projected features, ELU, and hand a 256-dim vector to the
head. The measurements below characterize each step.

## Evidence per hypothesis

Measured on VN30 lb10 h1: 33 nodes, 1050 train / 131 val / 132 test snapshots, 184 directed edges
(excl. self-loops). Quick model: seed 42, 12 epochs.

### H1 — Over-smoothing: SUPPORTED (partial; smoothing yes, collapse no)

Cosine-MAD = mean pairwise (1 − cosine similarity) across the 33 nodes on test snapshots (higher =
more node-specific dispersion):

| representation | cosine-MAD |
|---|---|
| raw node features (GAT input) | 0.815 |
| LSTM branch output [N,64] | 0.139 |
| GAT branch output [N,256] | 0.531 |

The GAT output is **0.65× as dispersed as its own input** — neighbor averaging demonstrably pulls node
representations toward each other. It is not a full collapse (0.53 is still well above zero, and higher
than the LSTM branch's 0.14), so the branch keeps some node identity. Verdict: real smoothing, not
catastrophic collapse. The damage is not "all nodes become identical"; it is "each node is pulled
toward an arbitrary neighbor average."

### H2 — Attention collapse: STRONGLY SUPPORTED (this is the core mechanism)

Normalized attention entropy (per target node over its neighbors, averaged over heads; 1.0 = uniform =
plain neighbor-averaging, 0 = one neighbor dominates):

- mean 0.9991, median 0.9996, p10 0.9985, p90 0.9999
- **100% of (node, snapshot) pairs have normalized entropy > 0.9** (near-uniform).

The multi-head attention learns essentially nothing — `alpha ≈ 1/degree` everywhere. The entire GAT
mechanism reduces to unweighted mean aggregation of the Top-5 neighbors. The "attention" in
"Graph Attention Network" is inert here. Cause: the attention logits are a LeakyReLU of a low-rank
projection of only 3 normalized features, softmaxed over ~5-8 neighbors; there is not enough signal
(and no gradient pressure, since averaging noise is locally optimal) to sharpen the weights.

### H3 — Edge quality / OOS transfer: STRONGLY SUPPORTED (edges are not stable)

Train-frozen glasso Top-5 adjacency vs a glasso adjacency re-estimated on the TEST rows:

- mean Jaccard of Top-5 neighbor sets = 0.167 (median 0.111, min 0.000, max 0.429)
- fraction of nodes with Jaccard ≥ 0.5: **0.000**
- correlation of full off-diagonal adjacency weights (train vs test): 0.193

The partial-correlation graph estimated on the train window barely overlaps the graph the same
estimator would draw on the test window. The frozen structure the GAT averages over is close to noise
out of sample. Even if attention worked, it would be attending over the wrong neighbors.

### H4 — Capacity / overfitting: SUPPORTED

| variant | params | train MSE | val MSE | test MSE | test−train gap | test QLIKE | test R² |
|---|---|---|---|---|---|---|---|
| GAT | 72,833 | 0.784 | 1.222 | 1.005 | +0.220 | 0.478 | 0.166 |
| no-GAT | 55,169 | 0.815 | 1.220 | 0.924 | +0.109 | 0.419 | 0.246 |

(MSE on normalized scale.) The GAT fits the train set better (0.784 < 0.815) but generalizes worse:
its test−train gap is **2× larger**. Training curves confirm: GAT val MSE bottoms at epoch 8 (1.2225)
then rises, while no-GAT val MSE is monotonically improving through epoch 12 (1.2204) — classic
overfitting from the extra 320-vs-64-dim head fed a noisy input. The +17k params buy lower train loss
and higher test loss.

### H5 — Where the GAT hurts: broad, worse on high-vol days

Per-observation QLIKE delta = QLIKE(GAT) − QLIKE(no-GAT); positive = GAT worse.

- overall mean delta +0.059; 56.6% of observations worse; **78.8% of test dates worse** (damage is
  systematic across the calendar, std across dates 0.14, not a few outlier days).
- worst tickers: MSN (+0.239, deg 8), VIB (+0.186), GAS (+0.161), TCB (+0.151), SSB (+0.142). Only
  MWG (−0.068) and a handful of large caps (VCB, FPT, VIC, VRE) see a tiny benefit.
- corr(node out-degree, damage) = +0.08 (weak); corr(node mean-vol, damage) = −0.14 (weak).
- **mean delta on the top-10% highest-vol observations = +0.094 vs +0.055 on the rest** — the graph
  hurts most precisely when a stock's volatility departs from its neighbors, exactly what neighbor-
  averaging would predict.

Damage is broadly distributed (not one pathological ticker), consistent with a systematic mechanism
(smoothing toward arbitrary neighbors) rather than a single bad edge.

## Most likely root cause

The three effects form one chain. The frozen glasso graph is **not out-of-sample-meaningful** (H3),
the attention **fails to weight neighbors** so the GAT is unweighted neighbor-averaging (H2), that
averaging **smooths each node toward its arbitrary neighbors** (H1), and the **extra head capacity fits
this noise on train and generalizes worse** (H4) — hurting broadly and most on high-vol days (H5).

Underneath all of it: the forecasting target is **per-ticker Parkinson variance, which is dominated by
that ticker's own recent volatility** (HAR persistence), already captured by the LSTM branch. There is
little residual cross-sectional signal for a graph to add, and what cross-sectional dependence exists
does not transfer across the train/test boundary on a 33-node panel. So the best thing a cross-
sectional smoother can do is add variance.

## Implications and recommendations

1. **Drop the graph for this model/target.** The evidence (attention collapse + non-transferable edges
   + overfitting + broad OOS damage) is not a tuning artifact; it is structural. This confirms prior
   project findings (memory: "graph adds no out-of-sample value"; EDA Conclusion C; GNNHAR arXiv
   2308.01419 that extra GNN depth does not help) with a mechanistic explanation. The `use_graph=False`
   ablation ("LSTM (w/o GAT)") should be the headline model; the GAT belongs in the paper only as a
   negative-result ablation.

2. **A different edge is unlikely to rescue it.** The problem is not "wrong edge type" — the attention
   collapses to uniform regardless (H2), and the correlation edge was already shown no better in prior
   work. The deeper issue is edge instability out of sample (H3) plus the near-absence of exploitable
   cross-sectional structure once own-ticker persistence is removed. Swapping glasso for correlation /
   k-NN / sector membership would still feed a smoother over an OOS-unstable graph.

3. **If cross-sectional information is wanted at all, use it as a low-dimensional exogenous feature,
   not a learned graph** — e.g. a single market/sector median-volatility regressor added to the HAR-X
   inputs (which the project's HAR-X baseline already does elsewhere). That injects the common factor
   as one stable scalar instead of a 256-dim smoothed vector, avoiding both the capacity blow-up (H4)
   and dependence on a per-pair graph that does not transfer (H3).

4. **Honest bottom line:** for HAR-feature Parkinson-variance forecasting on these small VN/US panels,
   cross-sectional graph structure is simply not useful out of sample. The GAT does not lose because it
   is under-tuned; it loses because there is no stable cross-sectional signal for it to exploit, and its
   only learned behavior here is to average noise.

## Reproduction

```
.venv_gpu_encode/Scripts/python.exe _tmp_gat_diag/diag.py   # VN30 lb10 h1, seed 42, 12 epochs
```

Quick-run test metrics reproduce the committed 5-seed direction and magnitude:

| | quick (1 seed, 12 ep) | committed (5 seeds) `results/soict/vn30_lb10_h1/result.json` |
|---|---|---|
| GAT QLIKE | 0.478 | 0.4528 |
| no-GAT QLIKE | 0.419 | 0.4120 |
| DM (GAT vs no-GAT) | — | favors no-GAT, p=1.7e-10 |
