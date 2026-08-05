# Code Review — 1-Day-Ahead Horizon Baseline (2026-08-01)

**Phương pháp:** tự review đối kháng, cùng cách 2 baseline horizon trước trong phiên này. Phạm vi:
2 file train mới (copy-modify cơ học từ sibling `2026-08-01_horizon22_baseline`, đã review).

## Kiểm tra trọng tâm — copy-modify cơ học có sót chỗ nào không

File tạo bằng `cp` + `sed` từ sibling h22 rồi sửa tay các chỗ sed không khớp (docstring nhiều
dòng). Đã grep lại toàn bộ 2 file + 2 test file tìm số "22" còn sót — chỉ còn các chỗ ĐÚNG Ý
(tham chiếu tới sibling h22/h10 trong docstring, hoặc `seq_length=22` — tham số KHÁC, không phải
horizon). Không còn `default=22` hay chuỗi "22-day" nào lẫn vào code logic của baseline này.

## Kiểm tra target shift + window count (không chỉ đọc code)

- `test_target_shift_h1.py`: xác nhận target = `parkinson_volatility[index+22]` (=`22+1-1`), KHÔNG
  PHẢI công thức của 3 horizon kia (`+26`, `+31`, `+43`). 4/4 pass.
- Window-count thật trên toàn bộ 32 mã: train 891→868, val/test 191→168 — **margin rộng nhất
  trong 4 horizon đã thử** (23 ngày tối thiểu/split, thấp hơn cả horizon-5). Đúng như dự đoán ở
  requirements.md §3, không phát sinh vấn đề.

## Findings

Không tìm thấy HIGH/MEDIUM. Diện thay đổi tối thiểu (2 file copy + đổi số + docstring).

## Acceptance Audit

| Tiêu chí | Trạng thái |
|---|---|
| Unit test target shift đúng 1 ngày | ✅ 4/4 pass |
| Real-data window-count | ✅ pass, margin rộng nhất trong 4 horizon |
| 2 script chạy 10 epoch thật | ⏳ chạy ngay sau review |
| pytest toàn bộ pass | ✅ 7/7 pass |

## Kết luận

0 HIGH/MEDIUM. Đủ điều kiện chạy training thật.
