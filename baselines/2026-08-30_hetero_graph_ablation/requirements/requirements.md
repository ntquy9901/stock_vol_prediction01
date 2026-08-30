# Requirements — Heterogeneous 2-relation GNN probe (HNX h1)

## Objective
Test whether keeping the linear-correlation edge and the non-linear (Apriori-lift) association edge as
SEPARATE relation types — with INDEPENDENT per-relation convolution weights and per-relation Min-Max weight
normalization — overturns the "graph does not help HNX" result. This is the 7th HNX graph edge probe.

Prior context (6 probes, all null on HNX h1 vs a no-graph LSTM; no-graph LSTM QLIKE ~= 1.81):
statistical vol->PK, sector ICB, MTGNN-learned, DY-spillover, Graph-WaveNet, and the SQUASHED corr+lift
(`baselines/2026-08-29_corrlift_ablation`). The squashed corr+lift was significantly WORSE (DM QLIKE
p=0.0037) largely because at the paper thresholds (|rho|>0.7, lift>1.7) the graph is near-empty
(3 corr + 12 lift edges over 154 nodes).

## Two design points (user request)
1. **Heterogeneous / multi-relational graph — do NOT squash.** Two separate relations, each with its OWN
   convolution weights (the network learns WHEN to weight the linear vs the non-linear signal):
   - `linear_corr`   — edges from Pearson |rho| on daily returns.
   - `nonlinear_assoc` — edges from Apriori lift on notable-move co-occurrence.
   Aggregate the two relation-specific node updates before the head. AGGREGATION = **SUM** (documented choice:
   keeps head input dim identical to the single-relation MaskedRichNet -> a clean controlled comparison; SUM
   is the PyG `HeteroConv` default aggregation).
2. **Per-relation Min-Max weight normalization to [0,1].** The two edge-weight spaces differ (|rho| after
   threshold in (0.25,1]; lift in (1.2,max]). Each relation's fired edge weights are Min-Max scaled to [0,1]
   INDEPENDENTLY with a TRAIN-ONLY scaler before feeding `edge_weight`, so gradients are not biased toward one
   relation's larger scale. (A tiny eps=1e-6 lower clip keeps the weakest edge present in the GAT mask, which
   treats weight 0 as "no edge"; values remain in [0,1].)

## Thresholds (user-chosen — DENSER than the paper; frame honestly)
- `linear_corr`: |rho| > **0.25** (paper uses 0.7).
- `nonlinear_assoc`: lift > **1.2** (paper uses 1.7).
These LOWERED thresholds DEPART from arXiv:2502.15813 to give each relation a non-trivial graph so the
heterogeneous architecture has something to propagate over. This is a denser-graph heterogeneous VARIANT
motivated by the paper's §3.2, NOT the paper's faithful thresholds. At rho~0.25 many edges are weak/noisy —
acknowledged honestly. Report per-relation edge density + degree distribution at these thresholds.

## Inputs / reuse (READ-ONLY, hard isolation)
- Masked panel, 5 node features, HAR-X anchor, per-ticker StandardScaler, QLIKE floor, chronological
  splits+seeds, RMR helpers (`_pred_dict/_ens/_metrics/_dm_all/_split_metrics/_ens_split/seed_metric_stats`,
  `OF.classify_fit`, `_batches`, `MaskedRichNet`, `train_masked_rich`): imported read-only from
  `baselines/2026-08-21_har_anchored_residual/code`.
- corr/lift builders (`pearson_corr`, `move_events`, `pairwise_lift`, `daily_returns`, `load_close_wide`,
  `build_corrlift_adjacency`): imported read-only from `baselines/2026-08-29_corrlift_ablation/code/corrlift_edge.py`.
- Two SEPARATE adjacencies (one per relation) with the new thresholds + per-relation Min-Max normalization are
  built in NEW code (`hetero_edges.py`).

## No leakage
Both adjacencies (returns, correlations, supports, lifts, AND the Min-Max min/max) are computed from TRAIN
ROWS ONLY (close rows strictly before the train/val boundary `D.d_va[0]`) then frozen. Since the graph is
built train-only, the Min-Max min/max are train-only by construction.

## Controlled comparison (same folds/seeds)
- `no_graph_LSTM`     — MaskedRichNet(use_graph=False).
- `hetero_2rel_GAT`   — HeteroRichNet (the new model; two relations, independent weights).
- `squashed_lowered_GAT` — MaskedRichNet(use_graph=True) on the SINGLE squashed corr+lift adjacency at the
  SAME lowered thresholds (0.25/1.2). Isolates "hetero vs squash" from "dense vs sparse".
- Context (report only, no retrain): prior squashed@paper-thresholds (0.7/1.7) QLIKE = 1.8192 from
  `results/corrlift_ablation/corrlift_ablation_hnx_h1.json`.

## Run config
- Panel HNX, horizon 1, 10 epochs (early stop), 3 seeds {42,123,2026}.
- GPU: `.venv_gpu_encode/Scripts/python` (torch 2.6.0). `torch.cuda.is_available()` = single source of truth
  for the device label. SINGLE process only. Small batch (~16-32) to stay under 8GB VRAM.
- Date-clustered DM (QLIKE): hetero vs no_graph, hetero vs squashed_lowered.

## Acceptance criteria (go / no-go)
- [ ] Two adjacencies built TRAIN-ONLY at 0.25 / 1.2; per-relation Min-Max to [0,1]; symmetric + self-loop=1.
- [ ] HeteroRichNet has INDEPENDENT conv weights per relation (distinct Parameter objects; grads diverge when
      the two adjacencies differ).
- [ ] All 5 metrics (MSE/RMSE/MAE/QLIKE/R2) + per-seed stats for the 3 trained variants, on the SAME folds/seeds.
- [ ] result.json carries train_metrics + val_metrics + fit_diagnostics + learning_curves per variant.
- [ ] Date-clustered DM (QLIKE/SE/AE) hetero vs no_graph and hetero vs squashed_lowered.
- [ ] Report: metric table + DM + fit verdict + per-relation edge density/degree + hetero-vs-squash finding +
      honest framing of lowered thresholds; separate "denser graph" from "heterogeneous handling" before
      crediting the architecture.
- [ ] Tests (unique basenames) pass; pre-push gate green (C0=100%/C1>=95% on changed lines, ruff F clean,
      lessons-regression, overfit-evidence).

## Success / interpretation
A 7th null (hetero does not beat no-graph LSTM under DM) is a valid, strong result — reported straight.
If hetero DOES help, quantify with DM p-values + seed-stability, and explicitly separate the effect of
"denser graph" vs "heterogeneous handling" before crediting the architecture.
