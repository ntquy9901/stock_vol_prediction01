# Code Review — Alignment-Loss Baseline (2026-07-18)

**Tool:** `/code-review`-style agent pass (1 agent, correctness focus across all 3 new
2026-07-18 baselines together) + self-fix, per CLAUDE.md DoD.

## Findings relevant to this baseline

- **`F.normalize` + cosine alignment loss: CONFIRMED CORRECT.** Agent verified normalization is
  applied once inside `model_alignment.py::forward()` (not skipped/re-applied inconsistently at
  the `train_alignment.py` call site), and that the alignment loss backprops into both
  `align_har`/`align_news` AND (via `har_embed`/`news_rep`) into the shared HAR/news encoders
  alongside the main MSE loss. No fix needed.
- **Test coverage gap (CONFIRMED):** smoke test never imported/called `train_alignment.py`'s
  real `train_epoch` (only tested `forward()` and a standalone `alignment_loss()` call) — a bug
  like a swapped/misapplied `lambda_align` in the real training loop wouldn't have been caught.
  **Fixed:** added `test/test_train_loop.py`.
- **Informational (not a finding):** `self.har.fusion.parameters()` is frozen in `__init__` but
  `fusion` (the reused `ParallelLSTMGNN`'s own internal MLP) is never called — only
  `get_embeddings()` is used. The freeze is dead code with no functional effect. Same pattern
  exists in all 3 new baselines and in the sibling `EmbeddingBaseline`; not changed here to keep
  this review's diff focused on real bugs.

## Final state

4/4 pytest pass (3 smoke + 1 train-loop integration).
