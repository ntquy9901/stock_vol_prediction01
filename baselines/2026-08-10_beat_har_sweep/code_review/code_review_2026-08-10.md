# Beat-HAR Sweep — Adversarial Code Review (2026-08-10)

Scope: `baselines/2026-08-10_beat_har_sweep/code/` (qlike_torch, spillover, adjacency_ops,
rvx_features, sweep, analyze, build_report). Pilot code (`2026-08-08_...`) imported read-only, not
modified, and out of scope. Three lenses: Blind Hunter (hidden bugs), Edge-Case Hunter, Acceptance
Auditor (does it meet the spec / leakage contract).

## Findings and resolutions

### F1 (HIGH, fixed) — QLIKE loss floor perturbed the physical scale
First `snapshot_qlike_loss` used a softplus floor `eps*softplus(raw/eps)+eps` with eps=1e-6 on ~1e-4
variance values; at ratio ~100 this shifted `pred_raw` by ~1e-6, so QLIKE was non-zero at a perfect
prediction and did not match `evaluation.qlike_loss`. Caught by `test_zero_at_perfect_prediction` and
`test_matches_numpy_qlike_on_uniform_present` (RED). Fixed: hard clamp identical to the numpy eval
QLIKE; positivity smoothness is supplied upstream by `GraphAblationModel._apply_positivity` before the
loss sees the prediction. Now bit-matches the eval metric on the healthy bulk.

### F2 (HIGH, verified) — leakage surfaces on the graph structure
- Spillover VAR (`load_train_volatility_panel`) filters `date <= graph.train_end_date` BEFORE fitting;
  the single directed matrix is frozen and reused for val/test snapshots. No val/test date enters the
  VAR. `test_spillover` checks directionality/asymmetry on train-only panels.
- Learned adjacency embeddings are input-independent `nn.Parameter`s (no dependence on any observation),
  so the learned graph cannot encode future targets.
- Base cache + scalers are the fair-ladder train-only artifacts; present-node masking unchanged;
  positivity floor on denormalized predictions. The val/test observation set is asserted identical to
  the consistent ladder (n_val 14418, n_test 14464) by `build_basis`'s one-basis invariant.

### F3 (MEDIUM, fixed) — self-loop / message-passing invariant under omit-self (C5)
`_ResidualMessagePassing` requires every present node to have a self-loop OR a neighbour. With
self-loops removed (C5), mutual-kNN can leave a present node isolated. `mask_static_adjacency(omit_self)`
keeps directed top-k out-edges and adds a self-loop fallback for isolated present nodes
(`test_static_omit_self_isolated_fallback`). Learned adjacency adds a present-node self-loop diagonal
(`test_mask_learned_adds_self_loop_and_masks_absent`).

### F4 (MEDIUM, accepted design) — QLIKE applied at the graph-head stage, not the backbone
The plan notes QLIKE should ideally be applied at both backbone and graph-head stages. This sweep keeps
the backbone MSE-identical to the consistent ladder and retrains only the graph-stage head +
message-passing under QLIKE. Rationale (recorded in design.md): (a) it isolates the loss/adjacency
lever on an IDENTICAL fair basis and reuses the base cache across configs; (b) the head (hidden→1, which
sets the prediction level) IS retrained under QLIKE, so the loss reshapes the level, not only the
message-passing residual. This is a deliberate, disclosed limitation, not a silent shortcut.

### F5 (MEDIUM, guarded) — degenerate VAR / FEVD
`directed_spillover_adjacency` rejects <2 series, too-few observations, constant series, non-finite or
non-positive residual variance, and zero-total-variance FEVD rows (`test_rejects_degenerate_panel`).
Row-normalization is asserted (`test_shape_finite_rows_normalized`).

### F6 (LOW, fixed) — DM significance on heavy-tailed loss differentials
An early analyze test asserted a "win" on synthetic data whose P0 predictions floored below zero,
producing extreme QLIKE outliers → correctly non-significant DM (heavy-tailed loss diff → large HAC
variance). Fixed the TEST data to strictly-positive predictions; the DM code was correct. This is a
reminder that a large mean QLIKE delta is not a win without DM support — enforced in
`_analyze_variant.beats_P0_qlike_dm` (all-negative sign + per-seed DM p<0.05 + paired-t p<0.05).

### F7 (LOW, noted) — C4 (HAR-RV-X) feature math implemented + tested but not wired to a 6-feature
backbone
`rvx_features` (GK/RS/overnight in σ² units) is unit-tested and the raw OHLC exists
(`data/raw/prices/*_ohlcv.csv`). Full C4 wiring needs a 6-feature backbone, which requires extending the
pilot per-ticker preprocessor (`TickerPreprocessor.fit` hard-asserts 1 feature == target) — a change to
shared pilot code beyond this run's isolation scope. Reported as deferred, not claimed as run.

### F8 (LOW, noted) — C7 (news-as-edge) infeasible
The precomputed news panel is per-(ticker, date) PhoBERT vectors with no article-level multi-ticker
structure, so co-mention edges cannot be built from it. Marked infeasible per the plan's feasibility
gate; not run.

## Verdict
No open HIGH/MEDIUM correctness or leakage findings. 28 unit + 6 smoke tests green; ruff clean on the
baseline. The primary risk is scientific, not code: DM significance on 33 daily series is the real
arbiter and is reported honestly (see the results report).
