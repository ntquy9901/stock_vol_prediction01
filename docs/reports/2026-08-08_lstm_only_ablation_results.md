# LSTM-only Ablation — Does temporal nonlinearity alone beat classical HAR?

Date: 2026-08-08
Scope: fills the missing (no graph, no news) cell of the paper's 2x2 ablation grid
`(graph on/off) x (news on/off)` for the `ParallelLSTMGNN` architecture
(`src/lstm_gat_hybrid/model_parallel.py`). Standalone results file; folding into
`docs/paper/*.tex` is a later integration step and is not done here.

## 1. Question

The core architecture pairs a per-stock LSTM temporal branch with a Graph Attention
Network (GAT) spatial branch and, in the full model, a per-ticker news-gating branch.
Before attributing any credit to the graph or the news, the most basic question is:
does the per-stock LSTM's temporal nonlinearity **alone** — no cross-stock graph, no
news — beat the classical HAR linear regression baseline? And, as a distinct
sub-question, does adding the cross-stock graph help even when no news is present?

## 2. Method

"LSTM-only" is the news-free price-only backbone
(`src/lstm_gat_hybrid/train_parallel_enhanced.py`, `graph_method='knn'`) with the
GAT/spatial branch's cross-stock message passing disabled: on every forward call the
real k-NN adjacency is replaced with a batched identity matrix (`torch.eye(33)`).
With an identity adjacency each GAT node can only attend to itself (off-diagonal
entries are 0, masked to `-inf` before the attention softmax), so the spatial branch
degenerates to a per-node learned transform with the **same parameter count** — the
per-stock LSTM temporal stream is the only pathway carrying signal. This is the exact
identity-adjacency technique, verified the exact same way, as the existing no-graph
ablation (`scripts/ablation_no_graph/run_no_graph_ablation.py`).

Everything else is identical to the reference price-only backbone run:

- Same data/pipeline: `create_multi_stock_dataloaders_with_graph_method_fixed`,
  `data/processed` (33 tickers), chronological 70/15/15 split, per-stock
  normalization, outlier winsorization (`n_std=3.0`), data augmentation
  (`augmentation_prob=0.15`).
- Same model/config: `ParallelLSTMGNN` via `create_parallel_lstm_gat_model`.
- Same hyperparameters: `learning_rate=0.001`, `batch_size=11`, 20 epochs (no early
  stopping, `patience=min_epochs=20`), `weight_decay=1e-5`, `gradient_clip=0.5`,
  `lstm_dropout=0.2`, `fusion_dropout=0.15`.
- Same protocol: 3 seeds (42, 123, 2026), best-val-loss checkpoint used for test
  evaluation; 6 metrics via `src/common/evaluation.py::evaluate_predictions`
  (`n_stocks=` passed, so `directional_accuracy` is the correct per-ticker value, not
  the flatten-biased one).

Implementation: `scripts/ablation_no_graph/run_lstm_only_ablation.py`, a self-contained
script that only imports (read-only) the shared model/dataset/eval code. The adjacency
substitution happens in the script's own copied train/validate loops, right after each
batch is unpacked and before `model(x, adj_matrix)`. No shared/model code was modified.

### 2.1 Verification that the graph is genuinely removed

Sanity check on the first val batch of seed 42 (logged under
`ablation.sanity_check` in
`results/lstm_only_ablation_seed42_2026-08-08_141334/training_results.json`):

1. Content: the substituted adjacency equals `torch.eye(33)` exactly for every sample
   (`identity_adj[0] == torch.eye(33)` → True). The real k-NN adjacency for the same
   batch had 402 off-diagonal non-zero edges; the identity has 0.
2. Functional invariance: with the identity adjacency, perturbing the input features of
   every stock **except** stock 0 leaves stock 0's GNN embedding completely unchanged
   (max |Δ| = 0.000e+00), confirming no cross-stock message passing. With the real
   adjacency the same perturbation changed stock 0's GNN embedding substantially
   (max |Δ| = 2.319), confirming the real graph does pass messages and that the
   substitution — not a wiring bug — is what removes it.

Both assertions passed. Additionally, the three seeds reproduced the existing no-graph
ablation (`results/no_graph_ablation_seed{42,123,2026}_2026-08-05_*`) to full floating
precision (e.g. seed 42 QLIKE 0.45334091782569885 in both), because that ablation
applies the same identity substitution to the same news-free backbone with the same
seeds. This is an independent reproduction, and it also clarifies the grid labels
(see §6).

## 3. Results — full 2x2 grid

Test set, mean ± std across seeds 42 / 123 / 2026 (n=3) for the three trained cells;
HAR is a single deterministic linear fit. Sources: LSTM-only —
`results/lstm_only_ablation_seed{42,123,2026}_2026-08-08_*`; price-only backbone —
`results/parallel_lstm_gnn_knn_2026-08-03_230722`,
`..._seed123_2026-08-03_234613`, `..._seed2026_2026-08-04_000327`; FULL —
`results/per_ticker_gate_2026-08-03_230821`, `..._2026-08-04_000448`,
`..._2026-08-04_002252`; HAR — `results/har_baseline_2026-08-05_224208`.

The 2x2 design (cell → model):

| | No graph (identity adjacency) | Graph (k-NN, k=8) |
|---|---|---|
| **No news** | LSTM-only (this run) | Price-only backbone |
| **News (gated)** | not run — does not exist (see §6) | FULL model (per-ticker gate) |

Metrics (lower is better for MSE/RMSE/MAE/QLIKE; higher for R²/DirAcc):

| Model (cell) | MSE | RMSE | MAE | R² | QLIKE | DirAcc |
|---|---|---|---|---|---|---|
| **HAR (classical linear)** | 4.760e-06 | 0.0021817 | 0.0005754 | 0.7419 | 0.5493 | 48.65% |
| **LSTM-only** (no graph, no news) | 7.777e-06 ± 3.2e-07 | 0.0027884 ± 0.0000579 | 0.0007877 ± 0.0000109 | 0.7953 ± 0.0085 | 0.4657 ± 0.0112 | 48.29% ± 0.04 |
| **Price-only backbone** (graph, no news) | 8.551e-06 ± 5.3e-07 | 0.0029233 ± 0.0000904 | 0.0008113 ± 0.0000137 | 0.7749 ± 0.0140 | 0.4603 ± 0.0205 | 48.47% ± 0.35 |
| **FULL** (news + graph) | 7.479e-06 ± 5.3e-07 | 0.0027336 ± 0.0000958 | 0.0007930 ± 0.0000123 | 0.8031 ± 0.0139 | 0.4430 ± 0.0185 | 47.77% ± 0.52 |

Per-seed LSTM-only (source: each seed's `training_results.json`):

| Seed | MSE | RMSE | MAE | R² | QLIKE | DirAcc |
|---|---|---|---|---|---|---|
| 42 | 7.431e-06 | 0.002726 | 0.0007771 | 0.8044 | 0.4533 | 48.25% |
| 123 | 8.068e-06 | 0.002840 | 0.0007988 | 0.7876 | 0.4751 | 48.32% |
| 2026 | 7.833e-06 | 0.002799 | 0.0007873 | 0.7938 | 0.4686 | 48.30% |

## 4. HAR vs LSTM-only (the motivating comparison)

Difference = LSTM-only (mean, n=3) − HAR (single value):

| Metric | HAR | LSTM-only | Δ (LSTM − HAR) | Better model |
|---|---|---|---|---|
| MSE | 4.760e-06 | 7.777e-06 | +3.02e-06 | **HAR** |
| RMSE | 0.0021817 | 0.0027884 | +0.0006067 (+28%) | **HAR** |
| MAE | 0.0005754 | 0.0007877 | +0.0002123 (+37%) | **HAR** |
| R² | 0.7419 | 0.7953 | +0.0533 | **LSTM-only** |
| QLIKE | 0.5493 | 0.4657 | −0.0836 (−15%) | **LSTM-only** |
| DirAcc | 48.65% | 48.29% | −0.36 pp | ~tied (both ~random) |

**Answer: mixed, not a clean win.** LSTM's temporal nonlinearity alone does **not**
uniformly beat classical HAR. It clearly wins on the volatility-appropriate QLIKE
(0.4657 vs 0.5493, ~15% lower — consistent across all 3 seeds, all below HAR's 0.549)
and on pooled R² (0.795 vs 0.742). But HAR is clearly better on the plain squared/
absolute error metrics — RMSE (+28%) and MAE (+37%) — which is unsurprising: HAR's OLS
fit directly minimizes squared error, and the monthly HAR term (coefficient 0.81)
captures most of the persistence in daily-frequency Parkinson volatility. Directional
accuracy is a tie near the ~48% random level for both — neither predicts the sign of
day-to-day volatility change.

Caveat on cross-model comparison: the HAR baseline is fit and scored on per-stock
stacked rows (`src/har_baseline`, test_size = 21,154 single-target rows), whereas the
LSTM cells are scored on multi-stock date-aligned windows. Absolute-error magnitudes
are therefore indicative rather than an exact matched comparison; QLIKE and DirAcc are
more scale-robust but still population-dependent. The split ratios and target
(`target_5d`, 5-day-ahead Parkinson volatility) are the same. This is the same HAR
reference used as the project's baseline elsewhere.

## 5. Does the graph help WITHOUT news? (paired t-test, n=3)

Comparison LSTM-only → price-only backbone (adding the real k-NN graph, no news present
in either). Paired t-test, n=3, df=2, critical value t=4.303 at α=0.05 (the convention
used elsewhere in this project). Δ = backbone − LSTM-only:

| Metric | Δ (backbone − LSTM-only) | Paired t | Significant? |
|---|---|---|---|
| MSE | +7.7e-07 | t = +1.63 | no |
| RMSE | +0.0001350 | t = +1.64 | no |
| MAE | +0.0000236 | t = +1.89 | no |
| R² | −0.0204 | t = −1.63 | no |
| QLIKE | +0.0053 | t = −0.30 | no |
| DirAcc | +0.18 pp | t = +0.98 | no |

**Adding the cross-stock graph provides no statistically significant benefit on any
metric when no news is present** (all |t| < 4.303). The point estimates even lean
slightly the wrong way on the continuous-error metrics (backbone marginally worse than
LSTM-only on MSE/RMSE/MAE/R², marginally better only on DirAcc, +0.18 pp), all within
seed-to-seed noise. This matches, and independently reproduces, the existing no-graph
ablation (`docs/reports/2026-08-05_graph_ablation_results.md`): the GAT branch's
contribution does not come from cross-stock graph structure.

For contrast (context from prior work, `docs/reports/2026-08-03_final_paper_readiness_report.md`):
adding **news** on top of the graph (backbone → FULL) IS significant on QLIKE (t=−6.22),
RMSE (t=−9.38), MSE and R² (|t|≈9.7), though not on MAE (t=−3.79) or DirAcc (t=−2.97).
So on this dataset the ordering of component value is: news (significant) ≫ graph
(not significant).

## 6. Clarification on the 2x2 grid labels

During this work an inconsistency in the informal grid labeling was found and is worth
recording. The existing `scripts/ablation_no_graph/run_no_graph_ablation.py` imports the
**news-free** price-only backbone (`create_parallel_lstm_gat_model`,
`create_multi_stock_dataloaders_with_graph_method_fixed`) and applies identity
adjacency — i.e. it is the **(no news, no graph)** cell, the same as this LSTM-only run
(hence the bit-identical numbers), **not** the (news, no graph) cell. The full news
model lives separately in
`baselines/2026-07-26_per_ticker_news_gate_baseline/code/train_per_ticker_gate.py`.

Consequences:

- The **(news, no graph)** cell does **not** currently exist as a trained run. Producing
  it would require applying the same identity-adjacency substitution to the news
  (per-ticker gate) model, which is a separate follow-up, not covered here.
- The question "does the graph help?" has, to date, only been answered **without** news
  present (this run and the 2026-08-05 ablation both test LSTM-only vs backbone). A
  matched test of the graph's contribution **with** news present would compare the
  missing (news, no graph) cell against the FULL model and has not been run.

## 7. Interpretation summary

- LSTM alone does not cleanly beat HAR: it improves the volatility-appropriate QLIKE
  (~15%) and R², but loses on RMSE/MAE where HAR's OLS is strong; direction is random
  for both.
- The cross-stock graph adds no statistically significant value without news (n=3), and
  its point estimates are neutral-to-slightly-negative on error metrics.
- News is the only component with a statistically significant effect (on QLIKE/RMSE/
  MSE/R²), consistent with the project's headline finding.
- n=3 seeds is the minimum for a paired t-test; ≥5 seeds would strengthen the
  non-significant graph conclusion if the paper needs higher confidence.

## 8. Reproduction

```
python scripts/ablation_no_graph/run_lstm_only_ablation.py --seeds 42 123 2026
```

Outputs land in `results/lstm_only_ablation_seed{N}_<timestamp>/training_results.json`
(same schema as the reference runs). The first seed additionally logs the sanity-check
evidence under `ablation.sanity_check`.
