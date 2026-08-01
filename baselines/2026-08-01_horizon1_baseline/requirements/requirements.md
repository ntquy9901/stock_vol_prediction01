# Requirements — 1-Day-Ahead Volatility Forecast Baseline

## 1. Mục tiêu

Kiểm tra dự báo **1 ngày** — mốc ngắn nhất trong "Secondary Targets: 1, 10, 22-day" liệt kê ở
CLAUDE.md. Hoàn thành bộ 4 mốc horizon (1/5/10/22-ngày) cùng pipeline so sánh. Giả thuyết: 1-ngày
dễ dự báo nhất (volatility có tính "persistence" cao — hôm nay tương quan mạnh với ngày mai), tiếp
nối xu hướng đơn điệu đã thấy ở 3 mốc trước (ngắn hơn = dễ hơn, hội tụ nhanh hơn).

**Phạm vi (kế thừa nguyên vẹn pattern horizon10/horizon22, không hỏi lại):**
- Chỉ 2 kiến trúc: HAR-only, per-ticker gated news.
- 5/10/22-ngày giữ nguyên — 1-ngày là baseline MỚI riêng.

## 2. Cơ chế — vẫn chỉ đổi 1 tham số

`target_idx = i + seq_length(22) + forecast_horizon(1) - 1 = i + 22`.

## 3. Rủi ro — THẤP HƠN mọi horizon trước

`seq_length(22) + forecast_horizon(1) = 23` ngày tối thiểu/split — thấp hơn cả horizon-5 (27
ngày), vốn đã chạy an toàn nhiều lần. Không cần lo ngại window-count như horizon-22 (44 ngày) —
vẫn viết test real-data để xác nhận (nhất quán quy trình), nhưng kỳ vọng pass ngay không có bất
ngờ.

## 4. Acceptance criteria

- [ ] Unit test xác nhận target = `i+22` (không phải `i+26/i+31/i+43` của 3 horizon kia).
- [ ] Real-data test: không mã nào 0 window (kỳ vọng margin rộng, §3).
- [ ] 2 script train 10 epoch thật, đủ 6 metric.
- [ ] pytest pass, code review 0 HIGH.
- [ ] Cập nhật Bảng B (báo cáo tổng hợp) thành đủ 4 mốc horizon.

## 5. Out of scope

- Calendar-augmented ở horizon này.
- Train >10 epoch mà chưa xác nhận xu hướng hội tụ (theo đúng nguyên tắc áp dụng cho 10/22-ngày).
