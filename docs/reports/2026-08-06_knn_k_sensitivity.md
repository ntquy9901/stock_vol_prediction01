# k-NN k Sensitivity — Does "the graph doesn't help" hold across k?

Date: 2026-08-06 (k=4/k=16 seed 42), extended 2026-08-08 (k=4 seed 123)
Scope: bounded sensitivity check on the k-NN neighbour count of the cross-stock graph in the
price-only (HAR-only) `ParallelLSTMGNN` backbone. Companion to
`docs/reports/2026-08-05_graph_ablation_results.md` (the no-graph / identity-adjacency ablation),
which found no significant difference between the real k=8 graph and no graph at all. This document
asks whether that null result is specific to k=8.

## 1. Question

The k-NN graph degree `k` (`config.top_k_neighbors`) was never tuned for this project's VN30 data;
`k=8` is inherited from the reference architecture (Sonani et al. 2025), and
`docs/project/PAPER_ANALYSIS_SONANI_2025.md` flags it as "tune for our data". The 2026-08-05 no-graph
ablation showed that at k=8, removing the cross-stock graph entirely (identity adjacency) produces no
statistically significant change on any of the 6 metrics (3 seeds, all paired-t |t| < 1.9). Open
question: is "the graph doesn't help" robust across k, or would a different k reveal a real effect
that k=8 happens to miss?

## 2. Method

Same training protocol as the reference k=8 run
(`src/lstm_gat_hybrid/train_parallel_enhanced.py`, `graph_method='knn'`, 20 epochs, no early stopping,
`lr=0.001`, `batch_size=11`, `weight_decay=1e-5`, `gradient_clip=0.5`, `lstm_dropout=0.2`,
`fusion_dropout=0.15`), same data pipeline
(`create_multi_stock_dataloaders_with_graph_method_fixed`, `data/processed`, 33 tickers, chronological
70/15/15 split, per-stock normalization, `n_std=3.0` winsorization, `augmentation_prob=0.15`), same
model (`create_parallel_lstm_gat_model`), same best-val-loss checkpoint for test evaluation, same
6-metric evaluation with per-ticker `directional_accuracy` (`n_stocks=` passed). One variable changed:
the k-NN neighbour count.

Runs performed:

- k=4, seed 42 — `results/k_sensitivity_k4_seed42_2026-08-06_060944`
- k=16, seed 42 — `results/k_sensitivity_k16_seed42_2026-08-06_060952`
- k=4, seed 123 — `results/k_sensitivity_k4_seed123_2026-08-08_062549` (added to test whether k=4's
  seed-42 edge over k=8 is real or seed noise; the apparent "best" k warranted a second seed)

k=8 (reference) and the identity/no-graph baseline are **not** rerun; their seed-42 numbers are cited
from existing results (see §4).

### 2.1 How k was varied without editing shared config

`config.top_k_neighbors` is the only lever that controls the k-NN degree: it is read at graph-build
time in `graph_utils_fixed.DynamicGraphBuilder.build_correlation_graph` as
`self.config.top_k_neighbors`. Note that the `k_neighbors=` argument accepted by
`create_multi_stock_dataloaders_with_graph_method_fixed` is stored on the dataset but is **not** used
for graph construction — that wrapper builds its own internal `LSTMGATConfig()`
(`dataset_with_graph_method.py`), and the graph builder reads `top_k_neighbors` from it. So passing a
config instance to the wrapper is not possible (it exposes no such parameter).

The sensitivity script (`scripts/ablation_no_graph/run_k_sensitivity.py`) therefore patches the class
attribute `LSTMGATConfig.top_k_neighbors` in-process only, before the wrapper instantiates its
internal config, and restores it afterward. This is an in-memory, per-process override: `config.py` on
disk is unchanged, and other concurrent/future runs are separate Python processes that import the
default `k=8` fresh. No shared file was modified. The training loop itself is reused by direct import
of `train_epoch` / `validate` from `train_parallel_enhanced.py` (real adjacency, no substitution).

### 2.2 Edge-count verification (the lever works)

Before trusting results, the constructed adjacency's off-diagonal non-zero edge count (first val
batch, same counting convention as the 2026-08-05 report) was logged for each k. Edges scale roughly
linearly with k, as expected for a symmetrized k-NN graph (symmetrization makes average degree exceed
k):

| k | off-diagonal non-zero edges (batch 0, 33 nodes) | avg off-diagonal degree | source |
|---|---|---|---|
| 4 | 214 | 6.48 | this run (both seeds identical — graph is deterministic given the data window) |
| 8 (reference) | 402 | ~12.18 | `docs/reports/2026-08-05_graph_ablation_results.md` §2.1 |
| 16 | 716 | 21.70 | this run |

214 → 402 → 716 tracks the halving/doubling of k (214 ≈ 0.53 × 402; 716 ≈ 1.78 × 402), confirming k=4
built a genuinely sparser graph and k=16 a genuinely denser one than the k=8 reference. The lever
changed what it was supposed to change.

## 3. Results

Test set, seed 42, all four conditions side by side (one variable changed: the graph). k=8 and
no-graph are cited, not rerun.

| Metric | k=4 | k=8 (reference) | k=16 | No-graph (identity) |
|---|---|---|---|---|
| MSE | 7.739e-06 | 9.153e-06 | 8.543e-06 | 7.431e-06 |
| RMSE | 0.002782 | 0.003025 | 0.002923 | 0.002726 |
| MAE | 0.0007931 | 0.0008257 | 0.0008196 | 0.0007771 |
| R² | 0.7963 | 0.7590 | 0.7751 | 0.8044 |
| QLIKE | 0.4461 | 0.4839 | 0.4574 | 0.4533 |
| DirAcc (per-ticker) | 48.45% | 48.09% | 48.07% | 48.25% |

Sources: k=4 `results/k_sensitivity_k4_seed42_2026-08-06_060944/training_results.json`; k=8
`results/parallel_lstm_gnn_knn_2026-08-03_230722/training_results.json` (the seed-42 HAR-only backbone
run cited in `docs/reports/2026-08-03_final_paper_readiness_report.md` §1); k=16
`results/k_sensitivity_k16_seed42_2026-08-06_060952/training_results.json`; no-graph seed-42 row from
`docs/reports/2026-08-05_graph_ablation_results.md` §3 (its MSE = RMSE²).

Second seed for k=4 (test set):

| Metric | k=4 seed 42 | k=4 seed 123 | k=4 mean (n=2) |
|---|---|---|---|
| QLIKE | 0.4461 | 0.4503 | 0.4482 |
| RMSE | 0.002782 | 0.002803 | 0.002792 |
| MAE | 0.0007931 | 0.0007949 | 0.0007940 |
| R² | 0.7963 | 0.7932 | 0.7947 |
| DirAcc | 48.45% | 47.29% | 47.87% |

Source: `results/k_sensitivity_k4_seed123_2026-08-08_062549/training_results.json`.

For reference, the k=8 backbone's own 3-seed spread (from
`docs/reports/2026-08-03_final_paper_readiness_report.md` §1) is QLIKE 0.4603 ± 0.0205, R² 0.7749 ±
0.0140, DirAcc 48.47% ± 0.35 — i.e. seed-to-seed variation on QLIKE (± ~0.02) is comparable to the
gaps between k values in the table above.

## 4. Interpretation

**Directional accuracy: no effect at any k.** Across all four seed-42 conditions DirAcc sits in a
0.38-percentage-point band (48.07%–48.45%), and k=4's second seed drops to 47.29%. Every value is near
the ~48% level the project has repeatedly measured as effectively random for per-ticker direction. No
k value moves directional skill.

**Continuous-error metrics: small, non-monotonic, within seed noise.** There is no monotonic trend in
k. On QLIKE the ordering is k=4 (0.446) < no-graph (0.453) < k=16 (0.457) < k=8 (0.484); the reference
default k=8 is in fact the least favourable of the four on QLIKE, RMSE and R² at seed 42. But this k=8
value is a single seed, and it sits above the k=8 3-seed mean (0.460 ± 0.020) — seed 42 was an
unfavourable draw for k=8. k=4's two seeds (0.446, 0.450; mean 0.448) fall just below the k=8 mean but
inside its per-seed spread, so k=4's apparent advantage over k=8 is not distinguishable from
seed-to-seed variation at this sample size.

**Against the no-graph baseline, no k wins consistently.** At seed 42 the no-graph identity variant is
best on MAE and R², k=4 is best on QLIKE and RMSE, and they differ by only ~0.007 QLIKE and ~0.005 R².
None of the graph densities (k=4, 8, 16) produces a real, consistent advantage over having no
cross-stock edges at all.

**Conclusion.** The 2026-08-05 finding — that the cross-stock graph structure does not measurably help
this backbone — is not an artifact of the specific choice k=8. It holds across k ∈ {4, 8, 16}: the
sparser and denser graphs perform within seed noise of both k=8 and of the no-graph baseline, and none
recovers a benefit attributable to cross-stock message passing. If anything, k=8 (the value inherited
from Sonani et al. 2025) is a mild local worst-case on continuous metrics here, so the untuned default
is not hiding a real graph effect — retuning k does not reveal one.

**Caveat on sample size.** This is a bounded one-variable sensitivity check, not a definitive study:
k=8 and k=16 rest on a single seed each and k=4 on two seeds, while the measured seed-to-seed std on
QLIKE (± ~0.02) is comparable to the between-k gaps. The conclusion should be read as "no k in the
tested range shows an effect large enough to stand out above single-seed noise", not as a
multi-seed-significant claim. A full multi-seed k sweep (≥3 seeds per k, paired t-tests) would be
required to make a hard statistical statement; the existing k=8-vs-no-graph comparison (3 matched
seeds, non-significant) remains the strongest single piece of evidence, and this sweep is consistent
with it.

## 5. Reproduction

```
python scripts/ablation_no_graph/run_k_sensitivity.py --k 4 --seed 42
python scripts/ablation_no_graph/run_k_sensitivity.py --k 16 --seed 42
python scripts/ablation_no_graph/run_k_sensitivity.py --k 4 --seed 123
```

Each writes `results/k_sensitivity_k{K}_seed{seed}_<timestamp>/training_results.json`, with the
constructed-graph edge count logged under `sensitivity_check.edge_verification`. k=8 and the no-graph
baseline are cited from existing results (not rerun).
