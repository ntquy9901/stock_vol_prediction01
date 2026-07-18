# Code Review — Pure Market Baseline (2026-07-15)

**Tool:** `/code-review` (1 finder agent, correctness angles) + self-fix, per CLAUDE.md DoD.

## Findings (2) — cả 2 đã xử lý

| # | Finding | Verdict | Xử lý |
|---|---|---|---|
| 1 | `test_market_contribution_identical_across_stocks` tự tính lại logic broadcast thay vì gọi `model.forward()` thật — không bắt được bug thật trong `forward()` | CONFIRMED | Fixed: đổi sang `register_forward_pre_hook` trên `model.fusion` để bắt input THẬT của fusion layer trong `forward()`, verify 8 cột cuối (market slice) giống hệt nhau qua các mã |
| 2 | `_pad_articles` thiếu guard dim-mismatch (sibling `2026-07-08_market_fallback` có, bài học [HIGH-2]) — lỗi numpy broadcast-shape khó hiểu nếu cache lẫn dim khác nhau | CONFIRMED | Fixed: thêm `raise ValueError` rõ ràng trước khi gán, giống sibling |

## Final state

3/3 pytest pass (`pytest baselines/2026-07-15_pure_market_baseline/test/ -v`).
Smoke CLI (`train_pure_market.py --epochs 1 --smoke`) chạy end-to-end không lỗi (dummy market,
real price data 32 mã, train=864/val=164/test=164 sequences).
