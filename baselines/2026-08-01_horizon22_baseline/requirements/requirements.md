# Requirements — 22-Day-Ahead Volatility Forecast Baseline

## 1. Mục tiêu

Kiểm tra dự báo **22 ngày** (≈1 tháng giao dịch, khớp HAR "monthly" window) thay vì 5 hoặc 10
ngày — CLAUDE.md liệt kê 22-ngày là "Secondary Target". Câu hỏi: 22-ngày dễ/khó dự báo hơn 5 và
10-ngày ra sao, tin tức có vai trò khác không, và kiến trúc cần bao nhiêu epoch để hội tụ (bài học
từ 5-ngày hôm nay: epoch 10 chưa đủ, epoch ~20 mới là đỉnh).

**Phạm vi (kế thừa quyết định đã áp dụng cho horizon10, không hỏi lại vì cùng pattern):**
- Chỉ 2 kiến trúc: HAR-only (`ParallelLSTMGNN`) và per-ticker gated news (`PerTickerGatedNewsBaseline`).
- 5-ngày và 10-ngày giữ nguyên làm target/baseline hiện có — 22-ngày là baseline MỚI, không sửa
  file nào của 2 baseline horizon kia.

## 2. Vì sao vẫn chỉ là đổi 1 tham số

Giống hệt lý do đã nêu ở `2026-08-01_horizon10_baseline/requirements.md` §2 — `forecast_horizon`
là tham số constructor có sẵn, chỉ dùng ở `target_idx = i + seq_length(22) + forecast_horizon - 1`.
Với `forecast_horizon=22`: `target_idx = i + 43`.

## 3. Rủi ro MỚI so với horizon10 (cần kiểm tra thật, không chỉ suy luận)

`seq_length(22) + forecast_horizon(22) = 44` ngày tối thiểu mỗi split — cao hơn hẳn horizon10 (32
ngày) và horizon5 (27 ngày). Mã có lịch sử ngắn (vd SSB, ~1.5 năm dữ liệu, gia nhập VN30 gần đây)
có nguy cơ CAO HƠN bị 0 window ở 1 split (đặc biệt val/test, vốn chỉ 15% độ dài). Đây là điều
`test_target_shift_h10.py` không cần kiểm tra kỹ (32 ngày margin rộng hơn) nhưng ở đây BẮT BUỘC
phải test bằng dữ liệu thật, không chỉ synthetic — xem §5 acceptance criteria.

## 4. Bài học áp dụng từ thí nghiệm 5-ngày hôm nay (epoch 10 không đủ)

- Train 10 epoch trước (Training Policy mặc định), NHƯNG khi báo cáo, đọc kỹ xu hướng val
  loss/QLIKE/DirAcc giữa epoch 5→10 để đánh giá đã hội tụ hay chưa (giống cách đã làm với 5-ngày
  và 10-ngày) — KHÔNG tự ý train >10 epoch nếu chưa hỏi.
- Nếu xu hướng còn cải thiện rõ ở epoch 10 (như biến thể 5-ngày), báo cáo rõ + đề xuất train
  tiếp — không tự quyết định vượt cap khi chưa có xác nhận.

## 5. Acceptance criteria (go/no-go)

- [ ] Unit test xác nhận target shift đúng `i+43` (không phải `i+26` hay `i+31`).
- [ ] **Real-data test xác nhận KHÔNG mã nào trong `common_stocks` bị 0 window ở bất kỳ split nào**
      khi `forecast_horizon=22` — đây là rủi ro mới (§3), phải kiểm tra bằng dữ liệu thật của TOÀN
      BỘ 32 mã, không chỉ 1-2 mã mẫu.
- [ ] 2 script train chạy 10 epoch thật, xuất đủ 6 metric.
- [ ] pytest toàn bộ pass, code review 0 HIGH.
- [ ] So sánh 5/10/22-ngày cho cả 2 kiến trúc.

## 6. Out of scope

- Calendar-augmented variant ở horizon 22.
- Train >10 epoch mà chưa hỏi (dù biết từ bài học hôm nay có thể cần) — báo cáo xu hướng, hỏi trước.
- Multi-seed.
