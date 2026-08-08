# News + No-Graph Ablation — Does the cross-stock graph help when news IS present?

Date: 2026-08-08
Scope: fills the genuinely-missing 4th cell — (news, NO graph) — of the paper's 2x2
ablation grid `(graph on/off) x (news on/off)`, so the graph's marginal contribution
can be isolated *in the presence of news*. Standalone results file; folding into
`docs/paper/*.tex` is a later integration step and is not done here.

## 1. Question

The core architecture has three components: a per-stock LSTM temporal branch, a Graph
Attention Network (GAT) cross-stock spatial branch, and (in the full model) a
per-ticker news-gating branch. Three of the four `(graph on/off) x (news on/off)`
cells were already trained:

- (no graph, no news): LSTM-only ablation — `docs/reports/2026-08-08_lstm_only_ablation_results.md`
- (graph, no news):    price-only backbone — `src/lstm_gat_hybrid/train_parallel_enhanced.py`
- (graph, news):       FULL model — `baselines/2026-07-26_per_ticker_news_gate_baseline/`

The 4th cell — **(news, NO graph)** — was never actually run (see §6, Correction note).
Its absence meant the question *"does the cross-stock graph help when news is present?"*
had never been answered by a matched comparison. This run answers it: it trains the
FULL news model with the cross-stock graph disabled, so the graph's effect with news
present is measured as **FULL (news + graph) vs. news + no-graph (this run)**.

## 2. Method

The (news, no graph) model is the FULL per-ticker-gated news model
(`PerTickerGatedNewsBaseline`) with the GAT branch's cross-stock message passing
disabled: on every forward call the real k-NN adjacency is replaced with a batched
identity matrix (`torch.eye(33)`) before it reaches `model(x_har, adj, x_news)`. With
an identity adjacency each GAT node can only attend to itself (off-diagonal entries
are 0, masked to `-inf` before the attention softmax), so the spatial branch
degenerates to a per-node transform with the **same parameter count** — the news
branch is fully present; only the cross-stock graph is removed. This is the same
identity-adjacency technique, verified the same way, as the sibling ablations
(`scripts/ablation_no_graph/run_no_graph_ablation.py`,
`scripts/ablation_no_graph/run_lstm_only_ablation.py`).

Everything else is identical to the FULL cell's production runs
(`train_per_ticker_gate.py`):

- Same data/pipeline: `create_dual_news_dataloaders` with the real dual-group news
  panel (`data/features/dual_group_news_panel.parquet`, 96.97% date-match coverage),
  `data/processed` (33 tickers), chronological 70/15/15 split, per-stock
  normalization, outlier winsorization (`n_std=3.0`).
- Same model/config: `PerTickerGatedNewsBaseline`, `n_feat=146`, `d_news=64`,
  `dropout=0.5`, `num_stocks=33`, HAR-3 features on the temporal branch.
- Same hyperparameters: `lr=5e-3` (all params), `gate_lr=0.05` (the `gate_logits`
  parameter group only — kept as its own optimizer group exactly as the original),
  `weight_decay=1e-5`, `batch_size=32`, `gradient_clip=1.0`, MSE loss, Adam,
  `ReduceLROnPlateau(patience=5, factor=0.5)`.
- Same protocol: 3 seeds (42, 123, 2026); 20 epochs as two 10-epoch legs (the FULL
  run's 10+10 resume chaining, since `train_per_ticker_gate.py` caps invocations at
  `MAX_EPOCHS=10`) — each leg re-seeds and builds a fresh optimizer/scheduler/early-
  stopping, leg 2 loads leg 1's best checkpoint with its best-val counter reset, so the
  test model is the best epoch among 11-20, matching how the FULL cell was selected.
- Same metrics: `src/common/evaluation.py::evaluate_predictions` with `n_stocks=33`
  passed, so `directional_accuracy` is the correct per-ticker value, not the
  flatten-biased one.

Implementation: `scripts/ablation_no_graph/run_news_no_graph_ablation.py`, a
self-contained script that only imports (read-only) the shared model/dataset/eval
code and the FULL news baseline's model/dataloader. The adjacency substitution happens
in the script's own copied train/validate loops, right after each batch is unpacked and
before `model(x_har, adj, x_news)`. No shared or baseline code was modified. The ONLY
variable changed relative to the FULL cell is the adjacency content.

### 2.1 Verification that the graph is genuinely removed

Sanity check on the first val batch (logged under `ablation.sanity_check` in each
first-seed `training_results.json`):

1. Content: the substituted adjacency equals `torch.eye(33)` exactly for every sample
   (`identity_adj[0] == torch.eye(33)` → True). The real k-NN adjacency for the same
   batch had **402 off-diagonal non-zero edges**; the identity has 0.
2. Functional invariance: with the identity adjacency, perturbing the input features of
   every stock **except** stock 0 leaves stock 0's GNN embedding completely unchanged
   (`max |Δ| = 0.000e+00`), confirming no cross-stock message passing. With the real
   adjacency the same perturbation changed stock 0's GNN embedding substantially
   (`max |Δ| = 2.288` for the seed-42 batch, `3.011` for the seed-123 batch),
   confirming the real graph does pass messages and that the substitution — not a
   wiring bug — is what removes it.

Both assertions passed on every invocation (asserted in-code; the run aborts if either
fails). Verification evidence:
`results/news_no_graph_ablation_seed42_2026-08-08_153353/training_results.json`
(`ablation.sanity_check`).

## 3. Results — completed 2x2 grid

Test set, mean ± std across seeds 42 / 123 / 2026 (n=3). Sources: news+no-graph (this
run) — `results/news_no_graph_ablation_seed{42,123,2026}_2026-08-08_*`; LSTM-only and
price-only backbone — cited from `docs/reports/2026-08-08_lstm_only_ablation_results.md`;
FULL — cited from `docs/reports/2026-08-03_final_paper_readiness_report.md`
(`results/per_ticker_gate_2026-08-03_230821`, `..._2026-08-04_000448`,
`..._2026-08-04_002252`).

The 2x2 design (cell → model):

| | No graph (identity adjacency) | Graph (k-NN, k=8) |
|---|---|---|
| **No news** | LSTM-only | Price-only backbone |
| **News (gated)** | news + no-graph (**this run**) | FULL model (per-ticker gate) |

Metrics (lower is better for MSE/RMSE/MAE/QLIKE; higher for R²/DirAcc):

| Model (cell) | MSE | RMSE | MAE | R² | QLIKE | DirAcc |
|---|---|---|---|---|---|---|
| **LSTM-only** (no graph, no news) | 7.777e-06 ± 3.2e-07 | 0.0027884 ± 5.79e-05 | 0.0007877 ± 1.09e-05 | 0.7953 ± 0.0085 | 0.4657 ± 0.0112 | 48.29% ± 0.04 |
| **Price-only backbone** (graph, no news) | 8.551e-06 ± 5.3e-07 | 0.0029233 ± 9.04e-05 | 0.0008113 ± 1.37e-05 | 0.7749 ± 0.0140 | 0.4603 ± 0.0205 | 48.47% ± 0.35 |
| **News + no-graph** (news, no graph) — this run | 7.204e-06 ± 7.9e-07 | 0.0026813 ± 1.45e-04 | 0.0007752 ± 2.26e-05 | 0.8104 ± 0.0208 | 0.4480 ± 0.0222 | 48.19% ± 0.13 |
| **FULL** (news + graph) | 7.479e-06 ± 5.3e-07 | 0.0027336 ± 9.58e-05 | 0.0007930 ± 1.23e-05 | 0.8031 ± 0.0139 | 0.4430 ± 0.0185 | 47.77% ± 0.52 |

Per-seed news + no-graph (source: each seed's `training_results.json`):

| Seed | MSE | RMSE | MAE | R² | QLIKE | DirAcc |
|---|---|---|---|---|---|---|
| 42 | 8.114e-06 | 0.0028486 | 0.0008012 | 0.78638 | 0.47332 | 48.14% |
| 123 | 6.728e-06 | 0.0025939 | 0.0007598 | 0.82288 | 0.43203 | 48.10% |
| 2026 | 6.768e-06 | 0.0026016 | 0.0007646 | 0.82183 | 0.43864 | 48.34% |

Observation: the two **news** cells (news+no-graph and FULL) beat the two **no-news**
cells on QLIKE and RMSE regardless of the graph, and the news+no-graph cell has the
lowest MSE/RMSE and highest R² of all four cells. News is the component driving the
improvement; the graph moves the numbers by less than one seed's worth of noise.

## 4. Graph-contribution tests (two distinct paired t-tests)

Paired t-test, n=3, df=2, critical value `t=4.303` at α=0.05 (the convention used
elsewhere in this project). The two tests answer two different questions.

### 4.1 Graph's effect WITHOUT news — backbone vs LSTM-only

Cited from `docs/reports/2026-08-08_lstm_only_ablation_results.md` §5 (Δ = backbone −
LSTM-only, adding the real k-NN graph with no news present in either cell):

| Metric | Δ (backbone − LSTM-only) | Paired t | Significant? |
|---|---|---|---|
| MSE | +7.7e-07 | +1.63 | no |
| RMSE | +0.0001350 | +1.64 | no |
| MAE | +0.0000236 | +1.89 | no |
| R² | −0.0204 | −1.63 | no |
| QLIKE | +0.0053 | −0.30 | no |
| DirAcc | +0.18 pp | +0.98 | no |

No metric is significant; point estimates even lean slightly against the graph on the
continuous-error metrics. **The cross-stock graph does not help without news.**

### 4.2 Graph's effect WITH news — FULL vs news + no-graph (the newly-answerable question)

Δ = FULL − (news + no-graph), n=3, computed from per-seed test metrics of the two runs
(FULL: `results/per_ticker_gate_*`; news+no-graph: this run). A positive Δ on
MSE/RMSE/MAE/QLIKE means the FULL (graph-on) model is *worse*; a positive Δ on R²/DirAcc
means FULL is *better*.

| Metric | Δ (FULL − news+no-graph) | Paired t | Significant? | Direction |
|---|---|---|---|---|
| MSE | +2.75e-07 | +1.73 | no | graph slightly worse |
| RMSE | +5.23e-05 | +1.75 | no | graph slightly worse |
| MAE | +1.78e-05 | +2.95 | no | graph slightly worse |
| R² | −0.00725 | −1.73 | no | graph slightly worse |
| QLIKE | −0.00500 | −2.35 | no | graph slightly better |
| DirAcc | −0.43 pp | −1.89 | no | graph slightly worse |

Per-seed Δ (FULL − news+no-graph): QLIKE = [−0.00923, −0.00259, −0.00316];
RMSE = [−5.9e-06, +6.94e-05, +9.33e-05]; DirAcc = [−0.58, −0.73, +0.02] pp.

**Adding the cross-stock graph on top of news provides no statistically significant
benefit on any metric** (all |t| < 4.303). The point estimates split: the graph is
marginally better only on QLIKE (Δ = −0.005, t = −2.35, not significant) and marginally
*worse* on the other five metrics (news+no-graph has lower MSE/RMSE/MAE and higher
R²/DirAcc). All differences are within seed-to-seed noise.

## 5. Interpretation — honest reading of the real numbers

- **The cross-stock graph is non-significant in BOTH conditions.** Without news
  (backbone vs LSTM-only) it was non-significant on all six metrics (§4.1); with news
  (FULL vs news+no-graph) it is likewise non-significant on all six metrics (§4.2). The
  graph does not become useful once news is present — the hypothesis that "the graph
  helps when news is present, even though it didn't when news was absent" is **not
  supported** by these numbers.
- The point estimates, in both conditions, lean neutral-to-slightly-*against* the graph
  on the plain-error metrics: on QLIKE the graph is marginally better with news
  (−0.005) but marginally worse without news (+0.0053), i.e. no consistent sign. There
  is no metric on which the graph both improves the score and reaches significance in
  either condition.
- This **strengthens the parsimony argument**: the cross-stock GAT message-passing can
  be removed from the architecture without a statistically detectable loss, in either
  the news-free or the news-present configuration. On this dataset the ordering of
  component value is unchanged: news (statistically significant vs no-news on
  QLIKE/RMSE/MSE/R² — see the final-paper-readiness report) ≫ graph (non-significant in
  both conditions).
- **Sample-size caveat:** n=3 seeds is the minimum for a paired t-test, not a strong
  statistical standard. Two of the WITH-news metrics have moderate |t| (MAE 2.95, QLIKE
  2.35) that would warrant ≥5 seeds before a firm "graph is useless with news" claim,
  though even the point estimates there favour *dropping* the graph, so more seeds would
  not change the parsimony conclusion in the graph's favour.

## 6. Correction note — the earlier mislabeled ablation

The earlier report `docs/reports/2026-08-05_graph_ablation_results.md` presented its
"no-graph ablation" (`scripts/ablation_no_graph/run_no_graph_ablation.py`) as isolating
the graph's contribution *in the FULL, news-present model*. That is not what the script
does: `run_no_graph_ablation.py` imports the **news-free** price-only backbone
(`create_parallel_lstm_gat_model` + `create_multi_stock_dataloaders_with_graph_method_fixed`,
which have no news branch anywhere) and applies identity adjacency to it. It therefore
measured the graph's contribution **when news is ABSENT** (backbone vs LSTM-only, which
turned out non-significant) — the **(no news, no graph)** cell, bit-identical to the
LSTM-only run — **not** the news-present ablation it was labeled as.

Consequences for the record:

- The genuinely-missing cell was **(news, NO graph)**, which did not exist as a trained
  run until this one. The 2026-08-08 LSTM-only report §6 already flagged this labeling
  inconsistency; this run closes it.
- The question "does the graph help when news is present?" (FULL vs news-minus-graph)
  was **never actually answered** by the 2026-08-05 report despite its framing. §4.2
  above answers it for the first time: it does not (non-significant on all six metrics).

Per project convention the earlier `2026-08-05_graph_ablation_results.md` is left
as-is (history preserved); this note documents the correction so the paper cites the
right comparison.

## 7. Reproduction

```
python scripts/ablation_no_graph/run_news_no_graph_ablation.py --seeds 42 123 2026
```

Outputs land in `results/news_no_graph_ablation_seed{N}_<timestamp>/training_results.json`
(same schema/metrics as the FULL runs, plus `ablation.sanity_check` on the first seed of
each invocation). Each seed is independent and fully deterministic (re-seeded, no data
shuffling): seeds 123 and 2026 were re-run after the first 3-seed launch was terminated
mid-seed-123 by memory pressure from concurrent training jobs, and reproduced the
interrupted run's per-epoch losses to full precision (e.g. seed 123 epoch 1 val loss
0.951491 in both), confirming determinism. Best epochs used for test: seed 42 → 20,
seed 123 → 20, seed 2026 → 18 (best-of-leg-2 selection, matching the FULL protocol).
