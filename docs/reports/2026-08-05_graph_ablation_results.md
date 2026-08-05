# No-Graph Ablation — Contribution of the Cross-Stock Graph Structure

Date: 2026-08-05
Scope: ablation isolating the GAT/spatial branch's cross-stock message passing in the
`ParallelLSTMGNN` architecture (`src/lstm_gat_hybrid/model_parallel.py`). Intended for a paper
Ablation Study subsection. This document is a standalone, ready-to-cite results file; folding it
into `docs/paper/*.tex` is a later integration step and is not done here.

## 1. Question

The core architecture runs a per-stock LSTM temporal branch in parallel with a Graph Attention
Network (GAT) spatial branch. The GAT branch aggregates information across stocks through an
adjacency matrix (k-NN over stock co-movement). The two branches are concatenated and passed
through a fusion MLP. The Method section frames this as combining temporal (LSTM) and
spatial/relational (GAT) information.

Ablation question: does the cross-stock graph structure itself contribute, independent of the GAT
module's extra parameters and capacity?

## 2. Method

A single change is made relative to the existing HAR-only backbone runs: on every forward call the
real adjacency matrix fed to the model is replaced with an identity matrix
(`torch.eye(num_stocks)`, broadcast to the batch). With an identity adjacency the GAT layers can
only attend to each node's own features — every off-diagonal entry is 0 and is masked to `-inf`
before the attention softmax, so each node's attention collapses to a weight of 1.0 on itself. The
spatial branch therefore degenerates to a per-node learned transform with the **same parameter
count** as the real model. This isolates "does cross-stock graph structure help" from "does the GAT
module's extra parameters/capacity help" — the standard way to run this ablation. The architecture
and parameter count are unchanged; only the contents of the adjacency matrix change.

Everything else is identical to the reference HAR-only backbone run (produced by
`src/lstm_gat_hybrid/train_parallel_enhanced.py`, `graph_method='knn'`):

- Same data and pipeline: `create_multi_stock_dataloaders_with_graph_method_fixed`, `data/processed`
  (33 tickers), same chronological 70/15/15 split, same per-stock normalization, same outlier
  winsorization (`n_std=3.0`), same data augmentation (`augmentation_prob=0.15`).
- Same model class and config: `ParallelLSTMGNN` via `create_parallel_lstm_gat_model`.
- Same hyperparameters: `learning_rate=0.001`, `batch_size=11`, 20 epochs (no early stopping,
  `patience=min_epochs=20`), `weight_decay=1e-5`, `gradient_clip=0.5`, `lstm_dropout=0.2`,
  `fusion_dropout=0.15`.
- Same protocol: 3 independent seeds (42, 123, 2026), 20 epochs each; best-val-loss checkpoint used
  for test evaluation; 6 mandatory metrics via `src/common/evaluation.py::evaluate_predictions`
  (`n_stocks=` passed, so `directional_accuracy` is the correct per-ticker value, not the
  flatten-biased one).

Implementation: `scripts/ablation_no_graph/run_no_graph_ablation.py`, a self-contained script that
only imports (read-only) the shared model/dataset/eval code. The adjacency substitution happens in
the script's own copied train/validate loops, immediately after each batch is unpacked and before
`model(x, adj_matrix)` is called. No shared/model code was modified.

### 2.1 Verification that the graph is genuinely removed

Before trusting results, a sanity check was run on the first batch of seed 42 (logged in
`results/no_graph_ablation_seed42_*/training_results.json` under `ablation.sanity_check`):

1. Content check: the substituted adjacency equals `torch.eye(33)` exactly for every sample in the
   batch (`identity_adj[0] == torch.eye(33)` → True). The real k-NN adjacency for the same batch had
   402 off-diagonal non-zero edges; the identity has 0.
2. Functional invariance check: with the identity adjacency, perturbing the input features of every
   stock **except** stock 0 leaves stock 0's GNN embedding completely unchanged (max |Δ| = 0.0),
   confirming no message passing across stocks. With the real adjacency the same perturbation
   changed stock 0's GNN embedding substantially (max |Δ| = 2.32), confirming the real graph does
   pass messages and that the substitution — not a wiring bug — is what removes it.

Both assertions passed, so the ablation is genuine (not a silent no-op leaving the real graph in
place).

## 3. Results

Reference "full architecture" numbers are the news-free HAR-only backbone runs with the real k-NN
graph (from `docs/reports/2026-08-03_final_paper_readiness_report.md` §1; per-seed sources listed
there). Both conditions use the identical protocol; the only difference is the adjacency matrix.

Test set, mean ± std across seeds 42 / 123 / 2026 (n=3):

| Metric | No-graph (identity adjacency) | Full architecture (real k-NN graph) | Δ (no-graph − real) | Paired t (n=3, df=2) |
|---|---|---|---|---|
| QLIKE | 0.4657 ± 0.0112 | 0.4603 ± 0.0205 | +0.0053 | t=+0.30, not significant |
| RMSE | 0.002788 ± 0.000058 | 0.002923 ± 0.000090 | −0.000135 | t=−1.64, not significant |
| MAE | 0.0007877 ± 0.0000109 | 0.0008113 ± 0.0000137 | −0.0000236 | t=−1.89, not significant |
| R² | 0.7953 ± 0.0085 | 0.7749 ± 0.0140 | +0.0204 | t=+1.63, not significant |
| DirAcc (per-ticker) | 48.29% ± 0.04 | 48.47% ± 0.35 | −0.18 pp | t=−0.98, not significant |

Paired t critical value at α=0.05, df=2 is 4.303; every |t| above is well below it.

Per-seed no-graph test metrics (source: `results/no_graph_ablation_seed{N}_*/training_results.json`):

| Seed | QLIKE | RMSE | MAE | R² | DirAcc |
|---|---|---|---|---|---|
| 42 | 0.453341 | 0.002726 | 0.0007771 | 0.804362 | 48.25% |
| 123 | 0.475059 | 0.002840 | 0.0007988 | 0.787608 | 48.32% |
| 2026 | 0.468602 | 0.002799 | 0.0007873 | 0.793792 | 48.30% |

MSE is omitted from the table above for brevity (it is RMSE²); it is present in each
`training_results.json`.

## 4. Interpretation

Replacing the real cross-stock graph with an identity adjacency — i.e. disabling all message
passing across stocks while keeping the architecture and parameter count identical — produces **no
statistically significant change on any of the six metrics** (all paired-t |t| < 4.303, n=3). The
absolute differences are small and inconsistent in direction: the real graph is marginally better on
QLIKE (+0.0053) and DirAcc (+0.18 pp), while the no-graph variant is marginally better on RMSE, MAE,
and R². All gaps are within, or comparable to, the seed-to-seed standard deviation.

The reading for the paper's Ablation Study: on this dataset and architecture, the GAT/spatial
branch's benefit does not come from the cross-stock graph structure (correlation/k-NN edges). A
graph carrying real cross-stock relationships performs statistically indistinguishably from one with
no cross-stock edges at all. Whatever the spatial branch contributes is attributable to its
per-node parametric capacity (the learned node transform inside the GAT layers), not to
message passing between stocks. Cross-stock volatility spillover, as encoded by this static k-NN
adjacency, is not a measurable driver of forecast quality here.

This is consistent with the project's broader finding that per-ticker directional signal is weak
(DirAcc near random across conditions) and that continuous-error metrics are dominated by each
ticker's own volatility dynamics, which the LSTM branch and the per-node GAT transform already
capture.

Caveat on sample size: n=3 seeds is the minimum for a paired t-test and is not a strong statistical
sample. The conclusion "the cross-stock graph provides no significant benefit" is well supported by
three matched seeds showing small, inconsistent, non-significant differences, but ≥5 seeds would
strengthen it if the paper needs higher confidence. The ablation was run under the exact protocol
of the reference runs, so the comparison itself is matched and fair.

## 5. Reproduction

```
python scripts/ablation_no_graph/run_no_graph_ablation.py --seeds 42 123 2026
```

Outputs land in `results/no_graph_ablation_seed{N}_<timestamp>/training_results.json` with the same
schema as the reference runs. The first seed additionally logs the sanity-check evidence under
`ablation.sanity_check`.
