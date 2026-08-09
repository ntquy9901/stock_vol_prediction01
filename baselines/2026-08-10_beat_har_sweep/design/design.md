# Beat-HAR Solution Sweep — Design (plan)

Date: 2026-08-10. Isolated baseline; imports pilot code READ-ONLY from
`baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/` (no pilot file modified).

## Data flow
1. `ladder_consistent.build_basis(device, stamp)` builds the seed-independent basis ONCE: pooled+news
   manifest, masked kNN-8 graph, graph-bound train set, graph_store (train-only scalers). Reused for
   C1/C2/C3/C5/C6. C4 rebuilds the basis with expanded node features.
2. Per seed (42/123/2026): build the MSE graph-safe P3 backbone exactly as the fair ladder
   (`build_graph_bound_p3_warm_start` 4ep + `build_graph_safe_p3_checkpoint` 1ep), load
   `GraphAblationModel`, precompute the frozen `base` cache (train/val/test) via `_build_shared_graph_base`.
   The base cache depends only on price/news inputs + frozen encoders, NOT on adjacency — so it is
   REUSED across every adjacency variant (C1 knn, C3/C5 spillover, C6 learned).
3. Per config: run a graph-stage head-training loop (this baseline's `sweep._train_graph_head`) with
   the config's loss (`mse`|`qlike`), adjacency (per-snapshot knn | frozen spillover | learned), and
   head wiring (monolithic | HAR-residual). Trainable = message-passing + head (+ learned-A embeddings
   for C6). Encoders/gate frozen. Weight decay 1e-5, grad clip 1.0, Adam. 20 graph epochs.
4. Evaluate val+test present-node predictions with `train.evaluate_records` (raw-scale, all 6 metrics)
   → write `predictions.json` / `predictions_test.json` (raw preds + raw targets, ordered) for DM.

## Differentiable QLIKE loss (`code/qlike_torch.py`)
- `snapshot_qlike_loss(pred_norm, target_norm, target_mean, target_std, presence, eps, floor_eps)`.
- Denormalize: `pred_raw = pred_norm*std + mean`, `target_raw = target_norm*std + mean` (per-node
  mean/std gathered by ticker from `model.target_mean/target_std`, the SAME train-fit target scaler).
- Positivity: `pred_raw` already floored by `apply_graph_head`; loss additionally softplus-floors
  `pred_raw` at `floor_eps` and clamps `target_raw>=eps` so QLIKE is defined and gradients are finite.
- QLIKE per element = `ratio - log(ratio) - 1`, `ratio = target_raw/pred_raw`. Averaged over PRESENT
  nodes per snapshot, then meaned over snapshots (identical weighting to `_mean_snapshot_mse`).
- Applied at the graph-head stage. Backbone kept MSE-identical to the fair ladder (design decision:
  isolates the loss lever on an identical basis + reuses base cache; the head — hidden→1, which sets
  the level — is retrained under QLIKE, so the loss reshapes the level, not only the residual).

## Per-config modules
- C1: knn-8 adjacency (unchanged), loss=qlike, monolithic head.
- C2 (`code/har_residual.py`): pooled HAR OLS fit on graph-bound train (last-timestep 3 HAR features →
  normalized target), frozen. Graph branch = `_ResidualMessagePassing` + zero-init residual Linear on
  the frozen base; `final_norm = har_norm + residual`; positivity floor via model scaler; loss=qlike.
  Zero-init residual ⇒ initial prediction == HAR ⇒ structurally floored at HAR.
- C3 (`code/spillover.py`): `directed_spillover_adjacency(train_vol_panel)` = generalized-FEVD
  Diebold–Yilmaz connectedness from a train-window VAR(p) on the per-ticker volatility panel,
  row-normalized, diagonal=1 (self-loop). Fit on TRAIN dates only, frozen; placed into every snapshot
  (masked to present nodes). loss=qlike.
- C4 (`code/rvx_features.py`): expand node price features 3 HAR → 3 HAR + GK + RS + overnight variance
  (σ² units from daily OHLC). Rebuild basis with `price_dim=6`; new train-only scalers. loss=qlike.
- C5: spillover adjacency with diagonal zeroed (omit self-loop) + directed top-k in {4,8,12,16};
  isolated present nodes get a self-loop fallback to satisfy the message-passing invariant. loss=qlike.
- C6 (`code/learned_adjacency.py`): `A = ReLU(tanh(alpha*(E1 E2^T - E2 E1^T)))`, top-k sparsified, from
  33×d input-independent learnable embeddings (no cross-time leakage), trained jointly under qlike.
- C7: infeasible on the per-(ticker,date) panel (no article-level multi-ticker structure); gated.

## Leakage safety
Spillover VAR and learned-A embeddings never see val/test targets: VAR fit on train volatility panel
only; learned embeddings are per-ticker parameters (input-independent). Base cache and scalers are the
fair-ladder train-only artifacts. Positivity floor on denormalized preds. Present masking unchanged.

## Simplicity / Anti-Abstraction gates
- Reuse pilot helpers (`_build_shared_graph_base`, `_stacked_snapshot_inputs`, `_mean_snapshot_mse`,
  `evaluate_records`, `_write_graph_predictions`, `diebold_mariano`) directly — no re-wrap.
- One compact `sweep.py` training loop parametrized by loss/adjacency/head, instead of editing the
  pilot `_run_one_graph_model` (keeps the fair ladder untouched; lower blast radius).
- No config/flag surface beyond the seven configs.

## DM / significance (`code/dm.py`)
Per seed: Diebold–Mariano on the identical per-observation QLIKE (and squared-error for RMSE) loss
vectors of config vs P0, on aligned held-out observations. Across seeds: paired-t on the 3 seed-mean
QLIKE deltas + consistent-sign check. Report per-seed DM stat/p + across-seed paired-t.
