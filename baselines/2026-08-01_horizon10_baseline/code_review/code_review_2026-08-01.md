# Code Review — 10-Day-Ahead Horizon Baseline (2026-08-01)

**Phương pháp:** tự review đối kháng (3 lớp: Blind Hunter, Edge Case Hunter, Acceptance Auditor),
cùng cách các baseline trước trong phiên này. Phạm vi: 2 file train mới. Dataset
(`MultiStockDatasetWithDualNews`) và model (`ParallelLSTMGNN`, `PerTickerGatedNewsBaseline`)
KHÔNG sửa (read-only import) — không review lại lần 2 (đã review ở baseline gốc).

## Kiểm tra trọng tâm: đúng công thức `forecast_horizon` không bị ignore

Rủi ro lớn nhất của baseline này (theo bản chất "chỉ đổi 1 tham số") là: nếu code lỡ QUÊN truyền
`forecast_horizon=args.forecast_horizon` vào `create_dual_news_dataloaders(...)`, script vẫn CHẠY
BÌNH THƯỜNG (không lỗi) nhưng ÂM THẦM dùng lại default 5 ngày — kết quả trông hợp lý nhưng SAI hoàn
toàn mục tiêu bài toán. Đã kiểm tra:
- Cả 2 file đều có `forecast_horizon=args.forecast_horizon` tường minh trong lời gọi
  `create_dual_news_dataloaders(...)` (không dựa vào default).
- `test_target_shift.py` (chạy TRƯỚC khi viết 2 script train, test-first) đã chứng minh bằng số
  liệu cụ thể (không chỉ đọc code): target đúng là `+10-1`, KHÔNG PHẢI `+5-1` — đây là bằng chứng
  mạnh hơn nhiều so với chỉ đọc lại dòng code.

## Findings

### Không tìm thấy HIGH nào

### LOW — Đã bỏ tính năng resume (`--resume_checkpoint`/`--resume_results_dir`) khỏi
`train_per_ticker_gate_h10.py` so với bản gốc 5-ngày
Quyết định có chủ đích (Simplicity Gate): baseline này chỉ cần 1 lần chạy 10 epoch để so sánh, không
cần train nối tiếp nhiều đợt như bản gốc (bản gốc từng train tới epoch 40 qua nhiều lần resume).
Nếu sau này cần train thêm epoch cho horizon 10, phải thêm lại tính năng này — ghi nhận rõ, không
phải thiếu sót ẩn.

### LOW — Rủi ro biên window-count cho mã có lịch sử ngắn (chưa xảy ra, cần theo dõi khi chạy thật)
`seq_length(22) + forecast_horizon(10) = 32` ngày tối thiểu mỗi split, so với 27 ngày ở bản 5-ngày.
Với các mã lịch sử ngắn (vd SSB, ~1.5 năm dữ liệu), split test/val có thể có `min_length` nhỏ — về
lý thuyết CÓ THỂ khiến 1 mã bị 0 window nếu split đó cực ngắn. Bản 5-ngày đã chạy thật thành công
với `common_stocks` hiện có (nên margin đủ), tăng thêm 5 ngày yêu cầu khó có khả năng phá vỡ điều
này, nhưng CHƯA chạy thật để xác nhận 100% — sẽ phát hiện ngay (crash rõ ràng, không âm thầm sai)
nếu xảy ra khi train thật.

## Acceptance Audit (đối chiếu requirements.md §4)

| Tiêu chí | Trạng thái |
|---|---|
| Unit test xác nhận target shift đúng 10 ngày, fail nếu dùng nhầm default 5 | ✅ `test_target_shift.py`, 5/5 pass |
| 2 script chạy 10 epoch thật, xuất đủ 6 metric | ✅ `results/har_only_h10_2026-08-01_090759/`, `results/per_ticker_gate_h10_2026-08-01_091853/` |
| pytest toàn bộ pass | ✅ 9/9 pass |
| Code review adversarial | ✅ file này |
| So sánh 5-ngày vs 10-ngày cho 2 kiến trúc | ✅ xem `docs/reports/2026-08-01_0928_summaryOfUpdate_report.md` — cả 4 metric xấu đi ở horizon 10-ngày cho cả 2 kiến trúc |

## Kết luận

Không có HIGH/MEDIUM cần fix. 2 LOW đã ghi nhận là quyết định có chủ đích/rủi ro thấp đã biết trước
— đủ điều kiện chạy training thật.
