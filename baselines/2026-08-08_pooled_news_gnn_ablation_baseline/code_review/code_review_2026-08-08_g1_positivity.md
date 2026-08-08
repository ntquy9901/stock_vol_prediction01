# G1 Positivity Parameterization Review

## Scope

Reviewed the change that enforces strictly positive denormalized graph predictions for G0/G1:
`GraphAblationModel` positivity buffers, `configure_positivity`, `_apply_positivity`, the forward
tail, and the single call site in `run_pilot._run_one_graph_model`. P0-P3 pooled screening, the
scaler (`scaling.py`), the data manifests, and provenance checks were not changed.

## Root cause (evidence)

`temp/agent_g1_positivity_diagnostic_output/positivity_diagnostic.json`: G0 and G1 both start
(epoch 0) with normalized predictions clustered near -0.19, all denormalizing positive. After 5
epochs G0's min normalized prediction reaches -0.48 (still above every ticker's `-mean/std`
positivity boundary in [-0.58, -0.86]); G1's reaches -0.87, crossing the boundary for low-`mean/std`
tickers, so 97/5445 = 1.78% denormalize to raw <= 0 and QLIKE diverges to 682. The only structural
difference between G0 and G1 is `_ResidualMessagePassing`; its added capacity widens the normalized
prediction variance so the lower tail crosses the denormalization positivity boundary. It is not a
scaler/denormalization bug (both are applied correctly) and not the classic "scaler fit but never
applied" bug.

## Design decision (softplus-at-head vs denormalized floor)

The user preference was softplus + epsilon at the head. Evaluated against the evidence and the
documented "LSTM-GNN Normalization Failure" (Softplus on the normalized output collapsed
predictions to 0):

- The defect is a lower-tail overshoot (~1.78%), not a whole-distribution scale problem, so a
  whole-output activation is disproportionate.
- Parkinson volatility is ~1e-3, deep in softplus's constant-offset region; a naive softplus at the
  head destroys the scale and forces abandoning the normalized-MSE training contract (which would
  also change G0 and touch provenance).
- Near-threshold predictions are common (G0 min normalized -0.48 vs boundary -0.58), so a
  normalized-space softplus soft-clamp (transition width ~1 std) would distort the bulk and bias the
  G0/G1 comparison.

Chosen fix: a denormalized-scale positive floor `raw_pos = eps*softplus(raw/eps) + eps` with
`eps = 1e-6` (three orders below the ~1e-3 volatility scale). This is softplus-based on the
denormalized scale (honoring the preference where it is safe), is an identity for `raw >> eps` (so
the bulk spread is preserved — no collapse), smoothly floors the sub-`eps` tail to strictly
positive, and is renormalized so the model still emits normalized predictions and the evaluation /
inverse-transform path is unchanged. Applied identically to G0 and G1 (shared `forward`).

## Blind Hunter

No critical/major finding. `configure_positivity` reads only per-ticker `target_scaler.mean/std`
(the same train-fitted values `PreprocessorStore.inverse_targets` already uses) — no new statistic,
no future information, scaler object not mutated (verified by
`test_configure_positivity_does_not_mutate_scaler_or_provenance`). Buffers move with the model
(configured before `model.to(device)`), so device alignment holds on the CUDA path.

## Edge Case Hunter

No critical/major finding.
- Unconfigured model: `_positivity_configured` defaults False, forward is byte-identical to the
  prior behavior (existing G0/G1 forward/gradient tests still pass); backward compatible.
- `F.softplus` reverts to the linear branch for large inputs, so `raw/eps` up to ~1e4 does not
  overflow; very negative `raw/eps` underflows softplus to 0 and the `+ eps` term keeps the result
  strictly > 0.
- Batched vs unbatched: `target_mean[ticker_ids]` is gathered on the flat ticker vector and reshaped
  to the output shape (row-major, matching how ticker_ids was flattened) — covered by the existing
  batched-equals-unbatched test remaining green.
- Missing preprocessor raises rather than silently skipping.

## Acceptance Auditor

No critical/major finding. Requirement met: G1 nonpositive fraction 0% (floor guarantees raw > 0),
so the <=1% safety gate passes. Split/scaler/provenance unchanged: `scaling.py`, the graph-safe P3
checkpoint provenance, and the manifests are untouched; the P3 checkpoint state_dict keys are
unchanged (`test_positivity_buffers_do_not_break_p3_checkpoint_loading`). Tests added first (RED
verified) then made to pass: strict positivity, no-collapse (spread preserved), scaler/provenance
unchanged, checkpoint-load compatibility, and an end-to-end `_run_one_graph_model` safety-gate test.

## Known limitation

The absolute floor `eps = 1e-6` is below the physical volatility scale, so the ~1.78% of G1 tail
predictions it floors are heavily penalized by QLIKE (which punishes under-prediction). This is an
honest reflection of message-passing producing non-physical near-zero predictions, not an artifact
of the fix; the floor-independent metrics (RMSE, MAE, R2, DirAcc) already show G1 <= G0, so the
architectural verdict does not depend on the floor magnitude.
