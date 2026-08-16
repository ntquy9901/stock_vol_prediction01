# GNNHAR-inspired optimizations (P1 QLIKE-loss, P2 1-hop+MAD, P3 regime-split) — Implementation Plan

> Spec+plan for experiments derived from arXiv:2308.01419 (see
> `docs/reports/2026-08-16_1212_gnnhar_2308.01419_optimization_experiments.md`). Extends the
> delivered baseline `baselines/2026-08-15_volatility` (no new baseline folder — these are
> loss/architecture variants of the existing FULL model on the same basis).

**Goal:** Test whether QLIKE training loss, 1-hop GAT, and regime-split reporting change the
parsimony-null verdict, per the paper's two headline findings (QLIKE-loss > MSE-loss; 1-hop ≥
multi-hop) and its regime analysis.

**Leakage/consistency invariants (ENFORCED — memory H2):** every compared model uses the IDENTICAL
positivity floor and QLIKE epsilon; edge/scaler/floor are train-only; DM = HLN with HAC lag h−1;
report all 6 metrics × 4 horizons. Select models on VALIDATION, never test.

---

## P1 — QLIKE training loss

**Files:** Modify `code/train_resume.py`; modify `code/run_ablation.py`; Test
`test/test_qlike_loss.py`, `test/test_ablation_qlike.py`.

Design: model.forward returns normalized pred `p_norm = (floored - mean)/std` where `floored ≥
POSITIVITY_EPSILON`. Physical positive pred = `p_norm*std_node + mean_node = floored`; physical
target = `t_norm*std_node + mean_node` (Parkinson variance, positive). QLIKE matches
`src/common/evaluation.qlike_loss`: clamp both at `epsilon=1e-8`, `mean(ratio - log(ratio) - 1)`,
masked over present nodes. `std_node/mean_node` gathered from `model.scaler_std/mean[ticker_ids]`.

- `_masked_qlike(pred_norm, target_norm, presence, mean_node, std_node, epsilon=1e-8)` — torch,
  differentiable, presence-masked.
- `train_with_resume(..., loss="mse")` — add `loss` param; training step and `_val_loss` both use
  the selected loss (QLIKE val-selection when `loss="qlike"`), so best-val checkpoint is chosen on
  the same criterion (paper: select on validation).
- `run_ablation.py`: add a `loss` arg threaded to `_train`; produce FULL/minus_*/lstm_only under
  `loss="qlike"` alongside the existing MSE run (new results dir suffix `_qlike`). HAR stays OLS
  (linear, MSE-fit) as the external baseline — its QLIKE is still reported.

TDD:
1. `test_masked_qlike_matches_eval` — build pred_norm/target_norm + scaler; assert `_masked_qlike`
   == numpy `qlike_loss` on the denormalized present values (atol 1e-6).
2. `test_masked_qlike_presence` — absent nodes excluded (mask=0 changes result correctly).
3. `test_train_qlike_runs` — tiny synthetic snaps, `loss="qlike"`, one+ epoch, finite best_val,
   val-selection uses qlike.

## P2 — configurable GAT depth (1-hop) + MAD

**Files:** Modify `code/model.py`; add `code/mad.py` (or a fn in model.py); Test
`test/test_gat_depth.py`, `test/test_mad.py`.

Design: `VolatilityModel(..., gat_layers: int = 2)`. `gat1: price_dim → hidden*heads` (=256);
`gat2` built only if `gat_layers == 2`. gnn branch dim stays `hidden*heads` either way (head
unchanged). `gat_layers=1` = 1-hop; existing default 2 preserves current behavior/results. Expose
node embeddings for MAD (optional forward return or a helper computing GAT output).

MAD: `mad(emb, presence) = mean over present-node pairs of (1 - cosine_similarity)`; lower MAD =
more over-smoothed. Pure function on `[N, d]` embeddings + presence.

TDD:
1. `test_gat_layers_one_forward` — `gat_layers=1` forward returns `[B,N]`, finite.
2. `test_gat_layers_two_default` — default builds gat2; 1-layer has no gat2 attr.
3. `test_mad_identical_zero` — identical embeddings → MAD 0; orthogonal → MAD 1 (atol).
4. `test_mad_presence` — absent nodes excluded from pairs.

Wire 0-hop (use_graph=False) / 1-hop / 2-hop into ablation as a graph-depth sub-study; log MAD of
gat1 vs gat2 output on test snaps.

## P3 — regime-split (calm/turbulent) metrics + DM

**Files:** Add `code/regime_report.py`; Test `test/test_regime_report.py`. Pure post-hoc analysis
on existing `predictions_test.json` dumps (rows: `ticker_id, target_date, target_raw,
prediction_raw`), no retrain.

Design: `split_regime(rows, turbulent_frac=0.10)` → label each obs turbulent if its `target_raw`
is in the top `turbulent_frac` quantile (regime by realized target volatility, as paper). Recompute
the 6 metrics (reuse `evaluate_predictions` + `qlike_loss`) per regime for each model. DM per regime
between two models' per-obs loss series (QLIKE and SE families), HLN small-sample, HAC lag h−1
(reuse existing `dm_report` DM core if importable, else implement HLN inline with a test).

TDD:
1. `test_split_regime_quantile` — known targets → correct turbulent set at 10%.
2. `test_regime_metrics` — per-regime metrics computed on the right subset.
3. `test_regime_dm` — DM sign/finite on synthetic where model A dominates in turbulent only.

## Execution after code+tests green
Run on GPU venv sequentially (memory: sequential GPU): (a) P1 QLIKE ablation h∈{1,5,10,22}; (b) P2
depth sub-study 0/1/2-hop; (c) P3 regime report over the FULL/HAR dumps from (a) and the existing
MSE run. Then DM FULL-QLIKE vs HAR per horizon + per regime. Report to `docs/reports/`, ledger
entry with real gate_results, push after each task.
