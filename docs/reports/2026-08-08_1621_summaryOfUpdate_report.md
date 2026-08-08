# Pooled news/GNN pilot — G1 positivity fix and G0-vs-G1 graph ablation

Branch: `feature/pooled-news-gnn-pilot` (worktree). Seed 42. git_sha at run: `298464f` (code fix
committed as `8055413`).

## Objective

G1 (graph message-passing ON) was safety-blocked: 1.78% of validation predictions denormalized to
nonpositive volatility, exceeding the 1% guard, so no valid G1 result existed. Fix the positivity
parameterization without changing the split/scaler/provenance rules, then produce the G0-vs-G1
causal comparison and position the GNN against HAR and P1.

## Root cause

Evidence: `temp/agent_g1_positivity_diagnostic_output/positivity_diagnostic.json`.

- At epoch 0 both G0 and G1 predict a narrow band (~-0.19 normalized), all denormalizing positive.
- After 5 epochs G0's minimum normalized prediction reaches -0.48, staying above every ticker's
  positivity boundary `-mean/std` (range [-0.58, -0.86]); all raw predictions stay positive.
- G1's minimum reaches -0.87. The 97 nonpositive predictions (1.78%) concentrate in the 7 tickers
  with the least-negative boundaries:

  | ticker_id | nonpositive/count | min_pred_norm | boundary `-mean/std` |
  |---|---|---|---|
  | 0 | 52/165 | -0.708 | -0.582 |
  | 1 | 24/165 | -0.842 | -0.614 |
  | 27 | 10/165 | -0.651 | -0.631 |
  | 18 | 7/165 | -0.872 | -0.737 |
  | 7 | 2/165 | -0.805 | -0.765 |
  | 15 | 1/165 | -0.733 | -0.709 |
  | 30 | 1/165 | -0.779 | -0.726 |

The only structural difference between G0 and G1 is `_ResidualMessagePassing`. Its added capacity
widens the normalized-prediction variance; the unconstrained linear head on the standardized scale
does not structurally enforce `raw = z*std + mean > 0`, so the widened lower tail crosses the
boundary for low-`mean/std` tickers. This is a lower-tail overshoot, not a scaler/denormalization
bug and not the classic "scaler fit but never applied" bug.

## Fix

`GraphAblationModel` (`code/models.py`) applies a denormalized-scale positive floor to the
prediction, identically for G0 and G1:

```
raw       = z * std + mean            # per-ticker denormalize (train-fitted scaler)
raw_pos   = eps * softplus(raw/eps) + eps   # eps = 1e-6
z_safe    = (raw_pos - mean) / std    # renormalize; evaluation path unchanged
```

The floor is an identity for `raw >> eps` (the bulk, so the spread is preserved — no collapse) and
smoothly maps the sub-`eps` tail to a strictly positive value. It uses only the existing per-ticker
`target_scaler.mean/std` (the same values `inverse_targets` uses), so no new statistic is derived and
the scaler/manifest/provenance are unchanged. Configured in `run_pilot._run_one_graph_model` after
provenance validation, before device transfer (buffers move with the model).

Softplus-at-head (the initial preference) was rejected on evidence: Parkinson volatility (~1e-3) sits
in softplus's constant-offset region, so a naive head softplus destroys the scale and forces
abandoning normalized-MSE training (which would also change G0 and touch provenance); a
normalized-space soft-clamp has a ~1-std transition that distorts the common near-threshold
predictions and would bias the comparison. Full rationale:
`code_review/code_review_2026-08-08_g1_positivity.md`; design addendum: `design/design.md` §9.

## Files changed

| Path | Purpose |
|---|---|
| `code/models.py` | `POSITIVITY_EPSILON`, positivity buffers, `configure_positivity`, `_apply_positivity`, forward floor |
| `code/run_pilot.py` | Call `model.configure_positivity(store)` in `_run_one_graph_model` |
| `test/test_models.py` | 5 tests: strict positivity, no-collapse spread, scaler/provenance unchanged, checkpoint-load compatibility, end-to-end safety gate |
| `code_review/code_review_2026-08-08_g1_positivity.md` | 3-layer adversarial review of the change |
| `design/design.md` | §9 addendum documenting the parameterization decision |

## Verification

- TDD: 5 tests written first, RED confirmed (4 failed for the missing feature; the backward-compat
  test passed by design), then GREEN.
- `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/ -q`: **92 passed**.
- `ruff check` on the three changed files: passed.
- `git diff --check`: clean.
- GPU smoke (3 tickers, 1 epoch, CUDA): G1 completes with finite metrics (qlike 0.549), no
  nonpositive raise.

## G0-vs-G1 result (full 33 tickers, 5 epochs, seed 42, CUDA)

Results: `results/pooled_news_gnn_g0g1_2026-08-08_154500_seed42/` (`graph_validation_comparison.json`,
per-config `results.json`, `provenance.json`). Denormalized-scale validation metrics, per-ticker
DirAcc convention:

| Metric | G0 (graph OFF) | G1 (graph ON) | Better |
|---|---|---|---|
| MSE | 6.78e-06 | 8.12e-06 | G0 |
| RMSE | 0.0026106 | 0.0028496 | G0 |
| MAE | 0.0006629 | 0.0007065 | G0 |
| R² | 0.6997 | 0.6421 | G0 |
| QLIKE | 0.8388 | 4.3811 | G0 |
| DirAcc | 49.24% | 47.67% | G0 |

Nonpositive-prediction fraction: G1 = 0% (floor guarantees raw > 0) — the ≤1% safety gate now
passes. QLIKE 4.38 is finite (down from the pre-fix unguarded 682) but inflated by the floored
tail; the floor-independent metrics (RMSE/MAE/R²/DirAcc) also favor G0. G0 reproduces the handoff G0
(rmse 0.00260, qlike 0.825, r² 0.701, DirAcc 49.39%).

**Verdict: graph message-passing hurts.** G1 is worse than its own no-graph control G0 on all six
metrics; paired val-loss delta (G1−G0) = +0.110.

## Position vs HAR (P0) and P1

Reference (full 33 tickers, pooled per-ticker validation basis):

| Config | RMSE | MAE | R² | QLIKE | DirAcc |
|---|---|---|---|---|---|
| P0 (HAR) | 0.001485 | 0.000480 | 0.735 | 0.517 | 48.54% |
| P1 (price-LSTM) | 0.001467 | 0.000461 | 0.741 | 0.524 | 48.64% |
| G0 (graph OFF) | 0.002611 | 0.000663 | 0.700 | 0.839 | 49.24% |
| G1 (graph ON) | 0.002850 | 0.000707 | 0.642 | 4.381 | 47.67% |

G0/G1 use the common-date-aligned graph validation basis while P0/P1 use per-ticker windows, so this
cross-comparison is indicative, not a matched paired test. Directionally, both graph variants show
higher RMSE and lower R² than HAR/P1, and G1 is the worst. Neither the graph control (G0) nor
message-passing (G1) beats HAR or the price-only pooled LSTM (P1).

## Recommendation

Keep the cross-stock graph message-passing (GNN) as an **ablation only, not the main architecture**.
Within the matched pair it underperforms its no-graph control; and neither graph variant improves on
HAR (P0) or the pooled price LSTM (P1). HAR remains the reference baseline for the paper.

## Risks / follow-ups

- The absolute floor eps=1e-6 is below the physical volatility scale, so QLIKE penalizes the floored
  G1 tail heavily. The architectural verdict does not depend on the floor magnitude (RMSE/MAE/R²/
  DirAcc are floor-independent and already favor G0). A physically-motivated per-ticker floor (e.g.
  minimum training volatility) would give a fairer G1 QLIKE but requires adding a scaler field; out of
  scope for this fix.
- The graph runner remains a per-snapshot / per-sample Python loop; the full 33-ticker run is
  launch-bound and slow under concurrent CPU load. No batching was introduced (would alter update
  semantics/provenance).

## Definition of Done

- [x] Root cause established from diagnostic evidence before any fix (systematic-debugging).
- [x] Fix is minimal and preserves split/scaler/provenance/leakage rules.
- [x] TDD: failing tests first, then implementation; 92 tests pass; ruff clean; git diff --check clean.
- [x] Safety gate passes (G1 nonpositive fraction 0% ≤ 1%).
- [x] 6 mandatory metrics reported on denormalized scale, per-ticker DirAcc.
- [x] Results saved under `results/` with timestamp + seed + git_sha provenance.
- [x] 3-layer adversarial code review recorded.
- [ ] Push: not performed (worktree/branch-only per task constraints; do not push/merge).
- [ ] diff-cover C0/C1: not run (tooling not installed in repo, per CLAUDE.md tooling gaps).
