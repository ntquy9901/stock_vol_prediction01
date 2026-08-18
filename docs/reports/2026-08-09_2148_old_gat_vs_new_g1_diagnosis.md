# Diagnosis — why the old Track-A LSTM-GAT beat HAR but the new Track-B G1 only ties HAR

Date: 2026-08-09
Scope: read-only diagnostic. No code, results, ledger, or worktree touched. Track-B code read from
committed branch `feature/masked-gnn` via `git show`; Track-A code read from the master tree.

## 0. The numbers in question

| Model (source) | test QLIKE | test RMSE | test R² | test DirAcc |
|---|---|---|---|---|
| Track-A HAR-R linear (`results/har_baseline_2026-08-05_224208/test_metrics.csv`) | 0.5493 | 0.002182 | 0.7419 | 48.65 |
| Track-A price-only LSTM-GAT backbone (`soict2026_draft_v3.tex` `\qlikeBB` etc.) | 0.4603 | 0.002923 | 0.7749 | 48.47 |
| Track-A gated-news LSTM-GAT (`\qlikeNews`) | 0.4430 | 0.002734 | 0.8031 | 47.77 |
| Track-B HAR per-ticker OLS (`classical_baselines_h5_2026-08-09_182129.json` test) | 0.5793 | 0.002290 | 0.7667 | 48.40 |
| Track-B G1 masked-GNN (`ladder_consistent_h5_2026-08-09_154402.json` test) | 0.5759 | 0.002305 | 0.7635 | 48.22 |
| Track-B P1 price-LSTM / P2 +news / P3 gate (same JSON) | 0.5648 / 0.5599 / 0.5765 | 0.002265 / 0.002270 / 0.002313 | 0.7718 / 0.7706 / 0.7620 | — |

Track-A: GAT beats HAR by ~0.09 QLIKE and +0.03–0.06 R². Track-B: G1 vs HAR QLIKE Δ = −0.0034
(tie), R² Δ = −0.003 (tie). The HAR number itself moves 0.5493 → 0.5793 between tracks.

---

## 1. Old GAT node features

Three HAR features per node — **not** the "22 features: HAR + technical" claimed in CLAUDE.md §4.
The sequence tensor is built from exactly three columns
(`src/lstm_gat_hybrid/dataset_presplit.py:125`):

```
x_seq = stock_feats[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].iloc[i:i+self.seq_length].values
```

HAR is generated split-locally (`dataset_with_graph_method.py:257 _generate_har_for_split`,
`generate_har_features`), i.e. daily = `parkinson_volatility`, weekly = 5-day rolling mean, monthly =
22-day rolling mean. `seq_length = 22` (`config.py:25`). The paper "backbone" is HAR-only; the
"22 technical features" line in CLAUDE.md §4 was never the implemented paper model. No technical
indicators (RSI, MACD, volume, …) enter the node.

## 2. Old GAT edges

k-NN over Pearson correlation of stock volatilities, **top-k = 8**, symmetric/undirected, one static
adjacency per split (not per-timestep dynamic in the paper runs).
`src/lstm_gat_hybrid/config.py:37 top_k_neighbors = 8`; dataloader default
`dataset_with_graph_method.py:313 k_neighbors: int = 8`, `graph_method='knn'`. Construction
(`src/lstm_gat_hybrid/graph_utils.py:91-110`):

```
# Use top-k neighbors instead of threshold for sparse graphs
k = min(self.config.top_k_neighbors, self.num_stocks - 1)
threshold = edge_weights_sorted[-k]           # keep only top-k edges
adj_matrix = np.where(np.abs(adj_matrix) >= threshold, adj_matrix, 0)
adj_matrix = self._normalize_adjacency(adj_matrix)
```

Edge weight = mean pairwise correlation of volatility (`build_correlation_graph`); a
correlation-threshold (`|corr|>0.7`) variant also exists (`graph_correlation.py:27`) but the paper
runs use `graph_method='knn'` (stated in `docs/reports/2026-08-05_graph_ablation_results.md §2`).
The no-graph ablation replaces this adjacency with `torch.eye(33)`.

## 3. New G1 node features + edges, contrasted

New G1 (`git show feature/masked-gnn:baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/`):

- **Price node features: the SAME three HAR features.** `scaling.py:100`
  `feature_order=(train_targets, "har_weekly", "har_monthly")`; `_har_features` (`scaling.py:199`)
  = `[parkinson_volatility, rolling(5).mean, rolling(22).mean]`, standardized per ticker. `seq_length
  = 22`, `horizon = 5` (`data.py:331`).
- **Plus a PhoBERT news branch** encoded by a second LSTM and concatenated
  (`models.py:40 PooledPriceNewsLSTM`; news via `x_news`/`news_mask`, publication-cutoff safe).
  G1 wraps the P3 (price+news+gate) encoders frozen and adds one residual message-passing layer over
  the concatenated node representation (`models.py:163 GraphAblationModel`, `use_gnn=True`;
  `_ResidualMessagePassing` at `models.py:124`).
- **Edges: the same family — masked k-NN-8 correlation.** Ladder JSON: `"adjacency": {"mode":
  "knn", "top_k": 8}`; graph nodes are present-node masked (absent nodes zeroed and never attended:
  `models.py:140-160`), adjacency is graph-bound to the training window (leakage-safe), avg 5.87
  off-diagonal edges per present row.

**What changed old → new:** node features were **not thinned** — both use the identical 3 HAR price
features; the new node is actually **richer** (adds a PhoBERT news channel). Edges are the same
k-NN-8 correlation family, now present-masked and train-bound. What changed is the **evaluation
basis and pooling**, not the feature/edge content (see §4).

---

## 4. Why old > HAR but new ≈ HAR — ranked causes

### Rank 1 (dominant): apples-to-oranges evaluation basis — different test observation sets.

The Track-A GAT-vs-HAR comparison is not measured on the same points, and this alone explains the
gap. Proof from the reported metrics themselves:

- The Track-A GAT has **worse** RMSE than HAR (0.002923 > 0.002182) yet **higher** R² (0.7749 >
  0.7419). On one observation set R² = 1 − MSE/Var(y) is monotone-decreasing in MSE, so higher RMSE
  **cannot** give higher R². The only way both hold is a different denominator (different target
  variance = different test set). Back-solving Var(y) = MSE/(1−R²): GAT ⇒ 8.54e-6/0.2251 = **3.80e-5**;
  HAR ⇒ 4.76e-6/0.2581 = **1.84e-5**. The GAT test target variance is ~2.06× the HAR test target
  variance — they are scored on different windows.
- The two protocols are explicitly different. HAR
  (`results/har_baseline_2026-08-05_224208/model_info.json`) is **point-wise, pooled across all
  tickers, ~80/20** (train 84549 / test 21154) with no windowing and no common-date restriction. The
  GAT is **windowed 70/15/15** (`_split_raw_data_by_date`, `train/val/test = 0.7/0.15/0.15`) over the
  **common-date intersection of all 33 tickers** (`_reindex_to_common_dates`,
  `dataset_with_graph_method.py:66`), which drops every date not traded by all 33 tickers — a
  systematically later, higher-variance window (consistent with the 2× variance above).
- QLIKE = mean of (σ̂²/σ² − ln(σ̂²/σ²) − 1) is per-observation and level-sensitive; computed on two
  different windows it is not comparable, so "0.4603 < 0.5493" is not a like-for-like win.
- The draft already concedes this (`soict2026_draft_v3.tex:631-638`): "The two baselines also use
  different evaluation protocols (point-wise 80/20 for HAR, windowed 70/15/15 for the deep model),
  which shifts the test window and the pooling."

The Track-B ladder was built precisely to remove this confound: HAR and G1 are scored on the
**exact same pooled val/test keys + raw targets + the same `train.evaluate_records` scorer**
(`classical_baselines_h5_*.json` `basis_note`; ladder `basis`: "identical val/test observations").
On that shared basis the RMSE/R² inversion disappears — back-solved Var(y) is 2.248e-5 (HAR) vs
2.246e-5 (G1), i.e. the **same** observation set — and G1 QLIKE 0.5759 ties HAR 0.5793. That is the
honest, protocol-matched answer, and it is a tie.

### Rank 2 (contributes to the Track-A headline, not to a real HAR gap): the normalizer-application effect on the deep model's own number.

`soict2026_draft_v3.tex:569`: "the backbone's feature normalizer had been fit but never applied, so
earlier models trained on raw, unnormalized volatility; applying it moved the headline QLIKE from
about 0.55 to about 0.46." This 0.55→0.46 is the **deep model's own** QLIKE before/after a training
fix — it is not a HAR-vs-GAT causal quantity, but it is what pushes the reported GAT number below
HAR's 0.5493. It rides on top of the Rank-1 basis mismatch (the 0.46 is still measured on the
windowed common-date set). This is the same "fit scaler, forget `.transform()`" family documented in
project memory (`project_normalizer_fit_never_applied_recurrence`).

### Rank 3 — refuted: "the new model lost richer features/edges."

Refuted. Both models use the **identical 3 HAR price features** (§1 vs §3) and the **same k-NN-8
correlation edge family** (§2 vs §3). No technical indicators were ever in the Track-A paper node,
so none were dropped. The new node adds a news channel, so it is richer, not thinner. There is **no
actionable "re-enrich the Track-B node with lost features" lead** from this comparison — the parity
is exact on price features.

### Rank 4 — refuted as the explanation: units.

Both tracks forecast the **same units**. `classical_baselines_h5_*.json` `target_units` confirms
`parkinson_volatility` is the Parkinson **variance** σ² = (ln(H/L))²/(4 ln 2); the Track-A HAR
(`model_info.json`, "Parkinson volatility from processed CSV files") and Track-A GAT both consume
this same column. So the σ-vs-σ² concern does **not** explain the 0.5493-vs-0.5793 HAR shift; that
~5% shift is the split/pooling/ticker-window difference (point-wise 80/20 pooled vs per-ticker OLS
on the masked 70/15/15 ladder keys), the same Rank-1 basis effect seen on HAR alone.

### Rank 5 — largely refuted: leakage/optimistic-eval in the old GAT.

The **current** (v3-cited, 2026-08-05) Track-A GAT runs are leakage-guarded: split-first with HAR
generated per split (`_generate_har_for_split`, no cross-split HAR warm-up), scaler fit on **train
only** (`dataset_with_graph_method.py:508` `train_dataset.feature_normalizers[...].fit(train_features)`,
copied to val/test), P1.2 common-date alignment, and the corrected per-ticker DirAcc
(`n_stocks=` passed). The historically inflated artifacts — DirAcc 68–72% from the flatten-order bug
and QLIKE 0.55 from the unapplied normalizer — were already corrected before v3
(`soict2026_draft_v3.tex:560-570`). So the residual GAT-over-HAR edge is **not** a leakage artifact;
it is the Rank-1 basis mismatch.

---

## 5. Bottom line

The old LSTM-GAT's apparent superiority over HAR is **not** a genuine feature/edge advantage the new
G1 lost (both use identical 3 HAR features + k-NN-8 correlation edges; G1 additionally has news), and
**not** a units difference (both σ²). It is **primarily (a) an apples-to-oranges evaluation basis**:
HAR was scored point-wise 80/20 pooled while the GAT was scored windowed 70/15/15 on the all-33-ticker
common-date intersection — a different, ~2× higher-variance test window, proven by the
mathematically impossible "worse RMSE but higher R²" pattern (Var_GAT 3.80e-5 vs Var_HAR 1.84e-5).
A secondary contributor is **(b) a training-side normalizer fix that moved the deep model's own QLIKE
0.55→0.46**, layered on that mismatched window. It is **not (c) a leakage artifact** in the current
runs (split-first, train-only scaler, P1.2 alignment, corrected DirAcc).

When HAR and G1 are finally put on one basis (identical pooled val/test keys, raw targets, and
scorer) in the Track-B ladder, the gap collapses: G1 QLIKE 0.5759 vs HAR 0.5793 (Δ −0.0034), R²
0.7635 vs 0.7667 — a tie, with a now-consistent RMSE/R² relationship confirming a shared observation
set. This **validates the current honest-parsimony finding**: on a like-for-like basis the graph
model does not beat classical HAR on the level metrics, and the earlier "GAT beats HAR by a lot"
headline was an artifact of comparing two different test sets, not a real advantage now missing from
G1.

### Sources
- `results/har_baseline_2026-08-05_224208/{test_metrics.csv,model_info.json}`
- `docs/reports/ladder_consistent_h5_2026-08-09_154402.json` (P0–G1 test)
- `docs/reports/classical_baselines_h5_2026-08-09_182129.json` (HAR/HARQ/GARCH test, `target_units`)
- `docs/paper/soict2026_draft_v3.tex:25-65` (macros), `:560-570` (corrections), `:631-638` (protocol)
- `docs/reports/2026-08-05_graph_ablation_results.md` (k-NN graph, protocol, seeds)
- `src/lstm_gat_hybrid/dataset_presplit.py:125`, `dataset_with_graph_method.py:66,257,313,508`,
  `graph_utils.py:91-110`, `config.py:25,37`
- `git show feature/masked-gnn:.../code/{models.py,data.py,scaling.py,run_pilot.py}`
