# Code review — heterogeneous 2-relation GNN probe (3-lens adversarial)

**Date:** 2026-08-30
**Scope:** `code/hetero_edges.py`, `code/hetero_model.py`, `code/run_hetero_ablation.py` (+ requirements /
design / tests). The delivered pipeline they import READ-ONLY (`masked_rich`, `run_masked_rich`,
`estimator_forecast_ablation`, `volatility_estimators`, `config`, `overfit_check`) and the corrlift primitives
(`corrlift_edge`) are out of scope except at the seams.
**Method:** 3-lens adversarial review — Blind Hunter (hidden bugs), Edge Case Hunter (boundaries), Acceptance
Auditor (does it meet the task) — plus a dedicated PERFORMANCE lens (batch=1 / GPU-underuse / per-step
transfer / main-thread-only loop).

## Outcome
No CRITICAL or MAJOR findings. Three MINOR (one accepted-defensive, two documentation-precision). The edge
math (per-relation thresholding, Min-Max, symmetry, self-loop, leakage freeze), the model (independent
per-relation conv weights, SUM aggregation, mask-aware batched forward), and the runner wiring (three variants
on the same D/folds/seeds/floor; over/under-fit evidence from the correct splits) were verified correct.

## Findings + resolution

### MINOR — m1: Min-Max maps the weakest edge to exactly 0, which the GAT would read as "no edge"
`WeightedGATLayer` derives its attention mask from `adjacency != 0`, so a pure Min-Max `(w-lo)/(hi-lo)` would
DROP the single minimum-weight edge per relation (and any ties at the min). **Resolution (by design, not a
patch):** `apply_minmax` lower-clips to `EPS=1e-6` so the weakest fired edge stays present under the mask while
all values remain in [0,1]. Documented in the module docstring + design.md; a test asserts the min maps to EPS
(not 0) and the max maps to 1.0, and that every normalized weight is in [0,1].

### MINOR — m2: hetero has ~2x the GAT parameters of the squashed variant (capacity confound)
`hetero_2rel_GAT` carries TWO GAT branches (linear + non-linear) vs the squashed variant's ONE, so a hetero
win could be capacity, not the heterogeneous handling. **Resolution:** the head input dim is held IDENTICAL
(SUM aggregation -> `hidden + hidden*heads`, same as the single-relation `MaskedRichNet`), so only the GAT
message-passing capacity differs — which is exactly the "separate relations" treatment under test. The report
states the parameter asymmetry explicitly and, per the task, separates "denser graph" from "heterogeneous
handling" before crediting the architecture; the null result is robust to the confound (extra capacity did not
help).

### MINOR — m3: leakage boundary is `D.d_va[0]`, marginally looser than the delivered edges' `last_tr_row`
Same seam as the corrlift probe: the frozen graph cut at the first VALIDATION target date ingests the
~horizon purge-gap rows that the delivered `adj_corr`/`adj_vol2pk` exclude. **No evaluation leakage** — every
close row fed to returns/corr/lift/Min-Max has `date < d_va[0] <=` every val/test target. `last_tr_row` is not
exposed on `D` and reproducing it would require editing the read-only `masked_rich.py`. **Resolution:** kept at
`D.d_va[0]` (a genuinely non-leaking cut that the requirements specify) with the looseness stated in the
docstring, design.md, and the report.

## Performance lens (ENFORCED)
- **Batching:** fully batched `[B,N,seq,5]`; both relation adjacencies batched `[B,N,N]` via
  `base * nmask.unsqueeze(1)`. No per-item Python loop in the hot path (the only Python loops are the epoch
  loop and the mini-batch index generator `RMR._batches`, both unavoidable). NOT batch=1.
- **GPU:** device from `torch.cuda.is_available()`; `X_tr`/masks/target-scaler preloaded to device once; the
  training hot loop does no host<->device copy per step beyond indexing preloaded GPU tensors. `.item()`/`.cpu()`
  only at epoch boundaries (val/train MSE), not in the inner batch loop.
- **Cost of the 2nd relation:** hetero doubles GAT FLOPs (two branches) but stays batched; 154 nodes at batch
  32 is well under 8 GB VRAM (verified: the run completes on the RTX 4060 GPU venv). Acceptable and documented.
- No correctness/leakage was traded for speed (mask, time order, per-ticker scaler all preserved).

## Verified correct (no action)
- Per-relation edges: `|rho| > 0.25` (linear), `lift > 1.2` (non-linear); diagonal fires dropped; symmetric;
  self-loop forced to 1.0; `np.isfinite(adj).all()` on real HNX. Independent-recompute test for the edge count.
- Per-relation Min-Max is TRAIN-ONLY (graph built on rows `< D.d_va[0]`); a poison-post-cutoff test proves the
  min/max + edges are frozen. Empty-relation and degenerate (all-equal) branches handled.
- `HeteroRichNet`: two `WeightedGATLayer` instances = distinct Parameter objects; a grad-divergence test
  (identical init, differing adjacencies -> different grads) proves independence; a "change adj_nl -> change
  output" test proves the non-linear relation is actually consumed; SUM aggregation head dim asserted.
- `train_hetero_rich` mirrors `train_masked_rich` semantics: per-node zscore target scaler, shared 1e-2*mean
  QLIKE positivity floor (identical across the 3 compared models), Adam + ReduceLROnPlateau, early stop on val
  MSE, per-epoch learning curves. Only the delivered `zscore_floor` output param is supported (minimum code).
- Runner: three variants on the SAME `D`, `cfg.seeds`, `cfg.qlike_floor`; adjacencies built once and frozen
  across seeds; DM includes hetero-vs-no_graph and hetero-vs-squashed_lowered (+ squashed-vs-no_graph).
- Over/under-fit evidence: `train_metrics`/`val_metrics` fed from the correct split arrays; `classify_fit` in
  the right positional order; per-seed learning curves stored. A test pins the train<val<test split ordering.
- Guards (empty val split, empty test split, non-finite forward output, `<2` processed files) all raise.

## Tests / gate
- `test/test_hetero_edges.py` (12) + `test/test_hetero_model.py` (7) + `test/test_hetero_runner.py` (13) = 32
  pass under the GPU venv.
- Coverage on changed code: **C0 line 100%, C1 branch 100%** across all three new modules.
- `ruff check --select F` clean (the `E702` semicolon findings are the repo's warn-only house style).
