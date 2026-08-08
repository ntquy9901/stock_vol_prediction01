# Code Review — Pooled News and GNN Ablation Pilot Specification

## Scope

- `requirements/requirements.md`
- `design/design.md`

No implementation code was reviewed or created in this phase.

## Review layers

### Blind Hunter

The review identified five high-severity and nine medium/low specification issues. The accepted
findings were resolved as follows:

- separated the pooled objective from a causal pooled-versus-panel claim;
- defined a global-date split for the graph protocol to prevent mixed split labels;
- paired G0/G1 initialization, batches, optimizer settings, and frozen encoders;
- constrained graph parameter fitting to graph training data;
- defined positive prediction handling and a fail-fast floor-rate threshold;
- restricted test evaluation to one validation-selected architecture;
- made promotion and three-seed retention rules operational;
- exempted closed-form HAR from neural epoch requirements;
- specified missing-news behavior, timestamp cutoff, directional-accuracy aggregation, and ticker
  manifest persistence.

The suggestion to shuffle pooled training was not accepted because the user explicitly selected
the conservative `shuffle=False` protocol and the experiment requires matching the established
LSTM-GNN/news convention.

The suggestion to generate validation/test windows with pre-boundary history was not accepted for
this pilot because the approved project convention is split-first, split-local HAR/window
generation. Changing that convention would add a second data-protocol experiment.

### Edge Case Hunter

The review identified seven high-severity and six medium issues. The accepted findings were
resolved by:

- defining one shared eligibility manifest and persisted exclusion reasons;
- hashing IDs, tensors, masks, raw targets, and preprocessing versions;
- defining exact horizon indexing and graph node/split invariants;
- defining the Asia/Ho_Chi_Minh forecast-origin timestamp;
- restricting learned news transformations to eligible training news;
- defining winsorization order and preservation of raw evaluation targets;
- specifying zero-variance scaler behavior;
- defining QLIKE positivity handling and directional-accuracy weighting.

### Acceptance Auditor

The final artifacts were checked against the approved horizon-5 matrix, per-ticker 70/15/15 split,
`shuffle=False`, train-only per-ticker scaling, raw-scale metrics, identical manifests within each
ablation family, five-epoch screening, and ten-epoch three-seed confirmation. No unresolved
placeholder or `[NEEDS CLARIFICATION]` marker remains.

## Result

Specification review passed after the corrections above. Implementation and experimental results
remain pending user review of the written specification.
