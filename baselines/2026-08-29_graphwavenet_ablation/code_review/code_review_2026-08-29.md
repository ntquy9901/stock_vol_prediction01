# Code review — Graph WaveNet ablation (2026-08-29)

3-lens adversarial review (Blind Hunter / Edge Case Hunter / Acceptance Auditor + performance-GPU lens) of
`code/gwn_model.py`, `code/run_gwn_ablation.py`, and the four test files, compared against the reference
`run_masked_rich.py` (delivered training path) and `masked_rich.py` (panel builder). Scope excluded
`archive/` per CLAUDE.md. Fast unit tests green (25 passed, 100% line + 100% branch on both code modules).

## CRITICAL
None. The GWN reproduction is structurally faithful to `nnzhan/Graph-WaveNet` `model.py`; `train_gwn`
mirrors `train_masked_rich`'s `zscore_floor` path line-for-line (standardized target, masked-MSE with
`tmb.sum().clamp(min=1)`, Adam + `ReduceLROnPlateau(0.5, patience=2)`, grad clip, early-stop on val
masked-MSE, denorm `max(pred·t_std+t_mean, 1e-2·t_mean+1e-12)`). No data leakage (train-only scalers reused,
never recomputed on val/test; target never fed as input). Verified correct: adaptive-adjacency source
masking (`adp·nmask[:, :, None]` + `einsum('ncvl,nvw->ncwl')` zeroes invalid SOURCE rows, never targets);
causal left-pad; skip/residual temporal truncation; `x[:, 0, :, -1]` readout; BatchNorm-with-batch-1 safe
(pooled over N·T).

## MAJOR — resolved by documentation (fidelity retained)
1. **BatchNorm2d pools statistics over zero-padded invalid nodes.** Inherent to applying the paper's BN to a
   masked-union panel (invalid nodes zero-filled). NOT swapped for a mask-aware norm — that would deviate
   from the faithful GWN architecture. It is COMMON-MODE across both GWN variants, so it cancels in the
   headline in-family adaptive ablation, and does not cancel vs LSTM/HAR (no BN). **Resolution:** documented
   in `gwn_model.py` docstring + `design.md`; the result JSON now reports `valid_node_fraction_test` to
   bound the cross-family effect; the report frames GWN-vs-LSTM/HAR as a backbone comparison.

## MINOR
2. Per-epoch full-train re-inference rebuilds train batches from numpy for the learning curve (perf, not
   correctness; inherited verbatim from the delivered path). Accepted for fidelity of the comparison;
   noted as follow-up.
3. "Paper's w/o adaptive" wording overclaimed (the paper's Table-4 row keeps predefined road supports; this
   panel has none, so removing adaptive leaves no graph conv). **Resolution:** reworded in docstring +
   runner header + report to "adaptive graph removed from the adaptive-only model."
4. Batch size differs across compared models (GWN 64 vs LSTM/GAT 16). Legitimate per-architecture
   hyperparameter, recorded in the result JSON; flagged for the paper's fairness framing.
5. `# pragma: no branch` best-state guard over-asserts under a NaN epoch-0 loss. Guarded in practice by the
   positivity floors; inherited from the reference. Accepted.

## Test-quality items — resolved
6. Added `test_forward_finite_with_partially_invalid_nmask` (a snapshot with some `nmask=0` nodes stays
   finite through the full stack).
7. Added `test_adaptive_and_no_adaptive_differ_and_param_presence` (adaptive owns `nodevec1`/`gconv` and
   yields a different prediction; no-adaptive has neither).

Positives recorded by the reviewer: named-formula rule satisfied
(`test_adaptive_adjacency_matches_paper_formula_independent_recompute` recomputes `softmax(relu(E1@E2),
dim=1)` in numpy from raw params, not reusing the module forward); `run_training` covered as an integration
test with training stubbed and a `train<val<test` MSE-ordering assertion confirming split wiring; the smoke
test asserts `overfit_check.check_result_evidence(res)` passes (gate-compatible evidence).
