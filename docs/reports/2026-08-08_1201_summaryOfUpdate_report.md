# Graph GPU Path Summary

## Change

Added graph-phase `--device {auto,cpu,cuda}` selection. The runner validates graph hashes and
checkpoint provenance before moving graph-safe P3, G0, G1, or a snapshot's tensors to the chosen
device. Results record device, PyTorch, and CUDA metadata.

## Files

- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py`: device path,
  provenance validation, deterministic CUDA seeding, and runtime metadata.
- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/models.py`: retains checkpoint
  graph provenance for the runner-level revalidation.
- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_models.py`: CPU smoke and
  CUDA-unavailable rejection tests.
- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code_review/code_review_2026-08-08_gpu_path.md`:
  adversarial review result.

## Verification

- `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test -v`: 83 passed.
- `ruff check baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test --exclude .agents --exclude .Codex --exclude _bmad --exclude archive --exclude data`: passed.
- `git diff --check`: passed.
- Diff coverage: Not run. The repository-level diff-coverage gate has no baseline-local coverage
  target configuration for this isolated code directory.

## Review

Three review layers found no critical or major issue. Details are recorded in the baseline-local
review file.

## Limitation

Graph updates remain one Python loop iteration per snapshot. No graph snapshot batching is added.
