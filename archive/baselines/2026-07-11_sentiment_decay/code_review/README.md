# code_review/ — Sentiment Decay Baseline

Adversarial review (CLAUDE.md §5 + §3.F). File: `code_review_<YYYY-MM-DD>.md`.

## Trạng thái (11/07/2026)
- Chưa có file review. Code (compute_decay.py) đơn giản (~30 dòng) + test. Review sau khi smoke pass.
- Trọng tâm review: decay chronological đúng (không per-window), schema reuse không break pipeline, mask=0/1 đúng.
