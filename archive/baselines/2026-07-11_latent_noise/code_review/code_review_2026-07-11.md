# Code Review — Latent Noise Baseline (2026-07-11)

**Baseline:** `baselines/2026-07-11_latent_noise/`
**Date:** 2026-07-11
**Scope:** `code/model_latent_noise.py`, `code/train_latent_noise.py`, `test/test_latent_noise.py`
**Method:** adversarial self-review + pytest (7/7 pass).

## Verdict
Pipeline cô lập đúng (subclass read-only EmbeddingBaseline, không sửa src/sibling). Noise logic
đúng ngữ nghĩa (train only, eval off, noise_std=0 disable). Không HIGH bug.

## Findings

### Confounded test (fixed during build)
- **Test `noise_std=0 → train==eval`** ban đầu FAIL vì Dropout (hardcoded 0.2 trong
  `ArticleSetAttentionPooling`/`NewsTemporalEncoder` của EmbeddingBaseline) cũng làm train≠eval.
  → Fix: đếm lần gọi `torch.randn_like` (noise dùng randn, dropout dùng bernoulli) → tách biệt sạch.
  3 test property mới chính xác: noise_std=0 → 0 call; noise_std>0 train → >0 call; eval → 0 call.

### Inherited (accepted, documented in design)
- **Dropout hardcoded 0.2** trong news_pool/news_temporal (từ EmbeddingBaseline) — không override
  được qua subclass constructor. Accepted: không liên quan noise,Embedding baseline đã review.
- **Subclass duplicate forward** (~5 dòng) vì EmbeddingBaseline.forward không có hook inject.
  Accepted trade-off (LOW, đã note trong sibling LOW-10 pattern).
- **noise_std=0.1 cố định, chưa tune** — experiment 5 epoch, defer tuning nếu lift.

### Not a bug (verified)
- `model.eval()` trước test eval (train_latent_noise.py) → noise chắc chắn OFF cho test công bằng.
- `weight_decay=1e-5` (§3.E), grad clip 1.0, learning curves (§3.C) — đầy đủ.
- Isolation: import read-only từ sibling + src; output vào results/models gốc (§3.D).

## Tests
`pytest baselines/2026-07-11_latent_noise/test/ -v` → **7/7 pass**:
output shape, eval determinism, train stochasticity, randn-call-count (×3), backward smoke.
