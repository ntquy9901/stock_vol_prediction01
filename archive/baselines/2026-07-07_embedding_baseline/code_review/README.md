# code_review/ — Code Review Results

Folder lưu kết quả **adversarial code review** cho baseline này (theo `CLAUDE.md` mục 5 + rule 3.F).

## Quy ước

- **Tên file**: `code_review_<YYYY-MM-DD>.md` (vd `code_review_2026-07-07.md`). Mỗi lần review = 1 file mới, không ghi đè.
- **Khi nào review**: SAU khi code/ chạy được + test/ pass, TRƯỚC khi coi baseline "done". Lặp lại sau mỗi lần fix lớn.

## Phương pháp (adversarial — per CLAUDE.md mục 5)

1. **Cynical review** — giả định code có bug, tìm ẩn.
2. **Tìm ≥10 issues** (ngưỡng tối thiểu cho độ sâu), phân loại HIGH / MEDIUM / LOW.
3. **Fix tất cả HIGH/MEDIUM** — không ngoại lệ.
4. **Verify lại bằng test** sau khi fix.

## Phạm vi review (cho baseline embedding)

| File | Trọng tâm |
|---|---|
| `code/extract_embeddings.py` | PCA leakage (fit trên train only), xử lý empty cache, env transformers<5, batch OOM |
| `code/dataset_embedding.py` | Windowing chống leakage (date ≤ T), padding/mask đúng, normalizer fit trên train only, MAX_ARTICLES truncation |
| `code/model_embedding.py` | Shape correctness, 0-news NaN, permutation invariance, unused ParallelLSTMGNN fusion params, determinism |
| `code/train_embedding_baseline.py` | Inverse_transform đúng per-stock, metric DirAcc (sign of changes), checkpoint save/load, EarlyStopping API |

## Format file review

```markdown
# Code Review — <baseline> (<date>)
**Reviewer:** <agent/human>  **Scope:** <files>

## Summary
- <N> findings: <H> HIGH, <M> MEDIUM, <L> LOW

## Findings
### [HIGH] <viết tắt bug> — <file:line>
**Vấn đề:** ...
**Kịch bản lỗi:** <input/state → output sai>
**Fix:** ...

### [MEDIUM] ...

## Re-verify
- [ ] test/ pass sau fix
- [ ] smoke run không regression
```

## Trạng thái hiện tại (07/07/2026)

- Chưa có file review nào trong folder này.
- Code đã smoke-test pass nhưng CHƯA qua adversarial review → baseline CHƯA "done" theo checklist rule 3.F.
