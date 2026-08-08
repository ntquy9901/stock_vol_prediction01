# Batching and CUDA Update

## Changes

- Pooled P1–P3 loaders now support batch size 256 by default, cached normalized targets, pinned host memory and non-blocking CUDA transfers.
- Pooled train/validation losses are weighted by actual sample count, including non-divisible final batches.
- Graph validation supports snapshot batching while graph training retains one-update-per-snapshot semantics.
- Graph validation loss preserves equal weighting per snapshot, including non-divisible final batches.

## Verification

- Baseline tests: `87 passed`.
- Ruff: passed.
- `git diff --check`: passed.
- Independent review: PASS; no findings for temporal split, scaler, shuffle, provenance, checkpoint/resume or CUDA transfer.

## Benchmark note

Three-ticker one-epoch benchmarks did not show end-to-end speedup because preprocessing dominates. Batched graph validation produced parity metrics with serial validation. Larger pooled runs are expected to benefit more from batch size 256.

## Known follow-up

Full 33-ticker GNN 5-epoch run completed G0 but G1 was blocked by the existing positivity safety guard (`1.78%` non-positive predictions versus `1%` threshold). No longer training process remains.
