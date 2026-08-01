# Requirements — 10-Day-Ahead Volatility Forecast Baseline

## 1. Mục tiêu

Kiểm tra dự báo **10 ngày** thay vì **5 ngày** (target hiện tại, vẫn giữ nguyên làm primary — xem
CLAUDE.md "Primary Target: 5-day ahead... Secondary Targets: 1, 10, 22-day"). Câu hỏi: 10-ngày dễ
hay khó dự báo hơn 5-ngày, và tin tức có giúp ích nhiều/ít hơn ở horizon dài hơn không?

**Quyết định phạm vi (user, 2026-08-01, qua AskUserQuestion):**
- Chỉ chạy 2 kiến trúc: **HAR-only** (`ParallelLSTMGNN`, đối chứng không tin) và **per-ticker gated
  news** (`PerTickerGatedNewsBaseline`, best hiện tại QLIKE/R²) — KHÔNG chạy calendar-augmented hay
  toàn bộ 12+ biến thể lịch sử.
- **5-ngày vẫn là target chính** — 10-ngày là baseline MỚI riêng biệt, KHÔNG sửa bất kỳ file 5-ngày
  nào đang có.

## 2. Vì sao đây KHÔNG phải thay đổi kiến trúc

`forecast_horizon` đã là tham số constructor sẵn có (`int = 5`) trong toàn bộ pipeline dataset
đang dùng (`src/lstm_gat_hybrid/dataset_presplit.py`, `dataset_with_graph_method.py`,
`create_dual_news_dataloaders`) — CHỈ dùng ở 1 chỗ:
`target_idx = i + seq_length + forecast_horizon - 1`. Input window (`seq_length=22`), HAR
rolling-window (1/5/22 ngày — khái niệm KHÁC, không liên quan tới forecast horizon dù trùng số 5),
kiến trúc model, loss, và cả 6 metric đánh giá đều KHÔNG phụ thuộc giá trị horizon.

→ Việc "chuyển sang 10 ngày" chỉ là **đổi 1 tham số khi gọi hàm đã có**, KHÔNG viết dataset/model
mới. Phần việc chính của baseline này là quy trình bắt buộc (SDD, test, review, báo cáo), không
phải logic dự báo.

## 3. Input/Output

Giống hệt `per_ticker_news_gate_baseline`/`news_usefulness_ablation` (x_har, adj, x_news, y) —
CHỈ khác: `y[ticker]` = `parkinson_volatility` tại `i + 22 + 10 - 1` thay vì `i + 22 + 5 - 1`.

## 4. Acceptance criteria (go/no-go)

- [ ] Unit test xác nhận target shift đúng 10 ngày (không phải 5) — test phải FAIL nếu lỡ dùng
      default cũ (horizon=5), PASS khi truyền đúng horizon=10.
- [ ] 2 script train (`train_har_only_reference_h10.py`, `train_per_ticker_gate_h10.py`) chạy được
      10 epoch thật, xuất đủ 6 metric bắt buộc, format `results.json` nhất quán với các baseline
      trước.
- [ ] pytest toàn bộ pass.
- [ ] Code review adversarial — kiểm tra kỹ off-by-one `target_idx`, biên window-count khi mã có
      chuỗi ngắn.
- [ ] So sánh 5-ngày vs 10-ngày cho CẢ 2 kiến trúc (4 ô số liệu: HAR-only×{5,10}, gated-news×{5,10}).

**Go/no-go:** baseline thử nghiệm — mục tiêu là ĐO, không bắt buộc 10-ngày phải "thắng" 5-ngày.

## 5. Giả định / lưu ý

- Số lượng window giảm nhẹ ở mỗi split (mất thêm 5 ngày cuối mỗi split so với horizon=5) — chấp
  nhận được, không cần bù (dữ liệu còn hàng nghìn window).
- Dùng LẠI panel tin tức + graph + optimizer config y hệt bản 5-ngày (không đổi hyperparameter nào
  khác ngoài horizon) để phép so sánh 5-vs-10 cô lập đúng 1 biến.
- KHÔNG train >10 epoch (Training Policy) mà không có approval dựa trên kết quả 5/10 epoch.

## 6. Out of scope

- Calendar-augmented variant ở horizon 10 (có thể làm sau nếu 10-ngày cho tín hiệu thú vị).
- Đổi 5-ngày thành 10-ngày ở BẤT KỲ baseline cũ nào — không sửa file nào ngoài folder mới này.
- Multi-seed, hyperparameter tuning riêng cho horizon 10.
