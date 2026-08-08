# Summary — Pooled News and GNN Ablation Pilot Specification

## Changes

- Added the requirements and architecture design for a horizon-5 pooled Price/News/Gate pilot and
  a separate matched GNN OFF/ON ablation.
- Defined temporal split, deterministic loader order, train-only preprocessing, news cutoff,
  scaler selection, raw-target evaluation, manifest equality, screening, and confirmation gates.
- Kept implementation isolated from existing baselines; no source code or existing baseline was
  modified.

## Files

- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/requirements/requirements.md` — source of
  truth for the pilot.
- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/design/design.md` — architecture, data
  flow, component boundaries, and test strategy.
- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code_review/code_review_2026-08-08.md` —
  three-layer specification review and dispositions.

## Verification

- Placeholder/ambiguity scan: passed; no `TBD`, `TODO`, or `[NEEDS CLARIFICATION]` remains.
- Markdown diff check: passed.
- Tests: not run; this phase changes documentation only and introduces no executable behavior.
- Coverage: not run; there are no executable changed lines.
- Lint: not run; no configured Markdown linter exists in the project.
- Smoke training: not run; implementation is gated on written-spec review.

## Review result

Blind Hunter and Edge Case Hunter findings were incorporated where they affected leakage,
confounding, metric validity, or reproducibility. The acceptance audit confirmed the written design
matches the approved pilot scope. Two suggestions were intentionally deferred: shuffled training
and cross-boundary validation/test context, because both would change user-approved project
conventions.

## Risks and next step

The global-date graph protocol and per-ticker pooled protocol intentionally use different sample
sets. Only G0 versus G1 supports a causal GNN comparison. Implementation must not begin until the
written specification is reviewed and approved.
