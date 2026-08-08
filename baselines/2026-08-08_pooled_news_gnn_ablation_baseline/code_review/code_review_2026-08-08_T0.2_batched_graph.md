# Code Review — T0.2 Batched Graph Path

Scope: `code/run_pilot.py` batching of (1) the graph-safe P3 builders and (2) the
`_run_one_graph_model` training loop; plus tests in `test/test_models.py`.

## Forward-only determination (Part 1 gate)

`build_graph_bound_p3_warm_start` and `build_graph_safe_p3_checkpoint` both run
`optimizer.zero_grad()/backward()/optimizer.step()` per sample (run_pilot.py lines
111/120/121 and 185/197/198 pre-change). They TRAIN P3, they are not forward-only.
Therefore the "safe, numerically-identical" Part-1 path does not apply; batching them
is an approved R10 SGD re-baseline. GPU profile (8 tickers, 2 epochs) shows the two
builders are 40.4% + 35.2% of graph-phase wall-clock, so this is where the speedup lives.

## Findings and resolutions

- Blind Hunter — target scaling: each sample keeps its own ticker `target_scaler`
  inside `_pooled_training_batches` (mixed-ticker chunks scale per row). Same values as
  the per-sample path. No scaler/leakage change.
- Edge Case Hunter — batch-1 identity: `test_warm_start_train_batch_size_one_matches_per_sample_reference`
  asserts the new bs=1 loop reproduces the historical per-sample checkpoint weights exactly.
- Edge Case Hunter — invalid batch size: both builders and `_run_one_graph_model` raise
  `ValueError(... train_batch_size ...)` for `< 1` (tested).
- Edge Case Hunter — frozen encoders: the `price_encoder.grad is not None` RuntimeError
  guard is preserved inside the batched training loop; batched forward still wraps the
  frozen encoders in `torch.no_grad()` (models.py unchanged). Existing frozen-encoder tests pass.
- Edge Case Hunter — loss weighting: training epoch loss uses snapshot-count weighting
  (`loss * len(snapshots)` summed / total), equal to the prior mean-over-snapshots at any
  batch size; validation still uses `_mean_snapshot_mse` (unchanged).
- Acceptance Auditor — positivity: denormalized positivity floor and the `evaluate_records`
  nonpositive<=1% gate are untouched; `test_graph_training_positivity_gate_holds_on_batched_run`
  exercises a negative-prone head under batched training and asserts finite metrics +
  nonpositive<=1%. Live run: G0/G1 nonpositive fraction = 0.0%.
- Acceptance Auditor — provenance: the graph manifest content hashes and `graph_train_hash`
  in the batched run are byte-identical to the Task 7 run; only the (shared) graph-safe P3
  weights re-baseline. G0 and G1 load the same checkpoint, so the ablation stays fair.

## Residual notes (non-blocking)

- Batched builder training uses per-chunk tensors built on the fly (bounded memory); it does
  not pre-stack all pooled samples. No unbounded growth.
- Training order remains deterministic/chronological (unshuffled), documented in-code.

No HIGH/MEDIUM findings left open.
