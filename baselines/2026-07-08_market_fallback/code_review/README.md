# code_review/ — Market Fallback Baseline

Folder lưu adversarial code review (CLAUDE.md §5 + rule §3.F). File convention: `code_review_<YYYY-MM-DD>.md`.

## Trạng thái (08/07/2026)
- Chưa có file review. Code đã build + smoke-test + pytest pass NHƯNG chưa qua adversarial review → baseline CHƯA "done" theo checklist §3.F.

## Phạm vi review (sauchạy smoke pass)
| File | Trọng tâm |
|---|---|
| `extract_market_embeddings.py` | PCA leakage (fit train only), no-ticker-filter đúng, env transformers<5 |
| `dataset_embedding.py` | market date-match, broadcast shape, 7-tuple đúng, normalizer fit train only |
| `model_embedding.py` | gate deterministic đúng, market broadcast, shape, NaN guard, reuse đúng |
| `train_market_fallback.py` | 7-tuple unpacking, inverse_transform per-stock, §3.B format, learning curves |

## Format
```markdown
# Code Review — market_fallback (<date>)
## Summary
- N findings: H HIGH, M MEDIUM, L LOW
## Findings
### [HIGH] <bug> — <file:line>
**Vấn đề:** ...  **Kịch bản lỗi:** ...  **Fix:** ...
## Re-verify
- [ ] pytest pass  - [ ] smoke không regression
```
