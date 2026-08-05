# Code Review — 22-Day-Ahead Horizon Baseline (2026-08-01)

**Phương pháp:** tự review đối kháng, cùng cách các baseline horizon trước trong phiên này. Phạm
vi: 2 file train mới (copy-modify từ sibling `2026-08-01_horizon10_baseline`, đã review). Dataset/
model không sửa.

## Kiểm tra trọng tâm — rủi ro MỚI: window-count ở split ngắn

Đây là baseline horizon-cao nhất từng thử (44 ngày tối thiểu/split, so với 27 ở 5-ngày, 32 ở
10-ngày). Đã viết `test_every_common_stock_has_positive_windows_at_horizon_22` chạy trên DỮ LIỆU
THẬT của toàn bộ 32 mã (không chỉ 1-2 mã mẫu) — kết quả đo được:

| Split | min_length (ngày) | Window @ horizon=22 | Window @ horizon=5 (đối chiếu) |
|---|---:|---:|---:|
| Train | 891 | 847 | 864 |
| Val | 191 | 147 | 164 |
| Test | 191 | 147 | 164 |

Val/test giảm ~10% số window (164→147) so với horizon-5 — không đáng lo, còn dư nhiều. Không mã
nào rơi vào rủi ro 0-window đã lo ngại ở design.md §3.

## Kiểm tra target shift đúng 22 ngày (không chỉ đọc code)

`test_target_shift_h22.py`: xác nhận target = `parkinson_volatility[index+43]` (=`22+22-1`), KHÔNG
PHẢI `+26` (5-ngày) hay `+31` (10-ngày). Test chạy TRƯỚC khi viết 2 script train.

## Findings

Không tìm thấy HIGH/MEDIUM nào. 2 file train là copy gần như nguyên vẹn từ sibling horizon10 (đã
review), chỉ đổi số 10→22 ở default arg + docstring + tên output dir — diện thay đổi rất nhỏ, rủi
ro thấp.

## Acceptance Audit (requirements.md §5)

| Tiêu chí | Trạng thái |
|---|---|
| Unit test target shift đúng 22 ngày | ✅ 4/4 pass |
| Real-data test không mã nào 0 window | ✅ pass, số liệu đo được ở trên |
| 2 script chạy 10 epoch thật | ⏳ chạy ngay sau review |
| pytest toàn bộ pass | ✅ 7/7 pass |
| So sánh 5/10/22-ngày | ⏳ chờ kết quả training |

## Kết luận

0 HIGH/MEDIUM. Đủ điều kiện chạy training thật.
