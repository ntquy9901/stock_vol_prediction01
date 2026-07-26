# Requirements (Specify) — Per-Ticker Isolated-Gradient News Gate

**Baseline:** `2026-07-26_per_ticker_news_gate_baseline` · Theo SDD (CLAUDE.md §1.5).
**Depends on:** `2026-07-25_dual_group_news_embedding_baseline` (reuses model/dataset read-only),
rebuilt `dual_group_news_panel.parquet` (12-source rebuild, phiên trước).

## 1. Bối cảnh

User (2026-07-26, sau khi được giải thích công thức `gate_mlp` của `gated_crossattn` baseline)
yêu cầu chính xác: *"tôi chỉ muốn mạng học cổ phiếu nào nên áp dụng tin vì nó có lợi cho dự đoán
biến động cổ phiếu đó, còn lại là không áp dụng cho các cổ phiếu còn lại nếu áp dụng tin làm giảm
độ chính xác."*

4 cách trước đây đều KHÔNG cho ra 1 danh sách "mã nào cần tin" ổn định/đáng tin (xem
`[[project_selective_news_gate_finding]]` trong memory): 2 lần dùng EDA ngoài (model khác hẳn,
không transfer), 1 lần dùng ablation nội bộ (đúng hướng nhưng vẫn thua HAR-only), 1 lần dùng gate
học được NHƯNG dùng 1 MLP CHUNG cho cả 30 mã — nên gate đó học theo pattern ngành/giá (vd cụm
ngân hàng có `h_gnn` giống nhau → gate cao giống nhau) chứ không phải độ hữu ích thật của tin tức
(tương quan với ablation độc lập: Pearson r=0.13, sai chiều).

## 2. Giả thuyết

Nếu mỗi mã có **1 tham số gate RIÊNG, không chia sẻ trọng số với mã khác**, thì gradient của
tham số đó chỉ phụ thuộc vào sai số dự đoán CỦA CHÍNH MÃ ĐÓ (cô lập hoàn toàn, chứng minh được
bằng đạo hàm) — loại bỏ hoàn toàn khả năng gate học theo pattern ngành/giá chung như gate cũ.
Sau khi train, `sigmoid(theta_i)` gần 0 ⇒ tin tức làm HẠI dự đoán mã i; gần 1 ⇒ tin tức GIÚP mã i.

## 3. Mục tiêu / Output

1. `model_per_ticker_gate.py`: `PerTickerGatedNewsBaseline` — y hệt kiến trúc
   `DualGroupNewsBaseline` (HAR branch + `NewsFeatureLSTM`, concat fusion), CHỈ THÊM
   `self.gate_logits = nn.Parameter(torch.zeros(num_stocks))`, gate = `sigmoid(gate_logits)`
   nhân theo-mã vào `news_rep` TRƯỚC khi concat. Init tại 0 → sigmoid(0)=0.5 (trung lập, không
   thiên vị mở/đóng ban đầu).
2. `train_per_ticker_gate.py`: training loop trên panel đã rebuild, **debug output bắt buộc**
   (yêu cầu user 2026-07-26):
   - **In ra màn hình mỗi epoch**: bảng 30 mã sắp theo gate value (cao→thấp), để theo dõi quá
     trình học "mã nào tin tức giúp, mã nào tin tức hại" thay đổi qua từng epoch.
   - **Lưu file** `gate_history.json` (`{epoch: {ticker: gate_value}}`, ghi đè/append mỗi epoch)
     → xem lại toàn bộ quá trình học sau khi train xong, không chỉ epoch cuối.
   - **Learning curve mỗi 5 epoch** (rule gốc CLAUDE.md §3.C) — loss train/val, TÁI DÙNG
     `plot_learning_curves_with_analysis` (không viết lại).
   - **Gate-evolution plot mỗi 5 epoch** (bổ sung, cùng tinh thần §3.C nhưng cho gate thay vì
     loss) — 1 đường/mã, trục X=epoch, trục Y=gate value, để NHÌN THẤY quá trình học trực quan
     (không chỉ đọc số).
3. Tái dùng 100% `create_dual_news_dataloaders`, `evaluate_predictions`, `EarlyStopping` (import
   read-only từ sibling `2026-07-25_dual_group_news_embedding_baseline` + `src.lstm_gat_hybrid`).
4. Loss = MSE thuần (KHÔNG kết hợp QLIKE — cô lập đúng 1 biến thử nghiệm: per-ticker gate, so
   với baseline đối chứng `DualGroupNewsBaseline` không gate, CÙNG panel, CÙNG loss).
5. Optimizer: `gate_logits` dùng learning rate RIÊNG, cao hơn phần còn lại (`--gate_lr`, mặc định
   0.05 vs `--lr` mặc định 5e-3) — vì đây là 30 số vô hướng cần "di chuyển" đủ nhanh để quan sát
   được trong 10 epoch (cap training policy), không phải speculative tuning (xem design.md §3).

## 4. Cô lập (hard isolation, CLAUDE.md §3.F rule 3)

Import read-only từ `2026-07-25_dual_group_news_embedding_baseline`
(`model_dual_news.NewsFeatureLSTM`, `dataset_dual_news.create_dual_news_dataloaders`) và
`src.lstm_gat_hybrid` (`config`, `model_parallel.ParallelLSTMGNN`,
`train_parallel_enhanced.EarlyStopping/plot_learning_curves_with_analysis`). KHÔNG sửa file nào
của baseline khác.

## 5. Success criteria / Go-No-go

- [x] **Property test bắt buộc (core claim):** `∂loss/∂theta_i` của mã i KHÔNG đổi khi target của
      mã j≠i thay đổi (chứng minh gradient cô lập hoàn toàn — khác biệt cốt lõi so với gate cũ).
- [x] Forward/backward shape đúng, gate ∈ (0,1) qua sigmoid.
- [x] Debug print + `gate_history.json` + gate-evolution plot hoạt động đúng trong 1 run thật.
- [x] Train 10 epoch thật (cap theo Training policy), learning curve mỗi 5 epoch.
- [x] So sánh test DirAcc/R²/QLIKE/RMSE với `DualGroupNewsBaseline` không gate (cùng panel, cùng
      loss).

  **KẾT QUẢ THẬT (2026-07-26, 10 epoch, panel rebuilt):**
  | Metric | Dual-group (không gate) | Per-ticker gate (MỚI) | Diff | vs. record hiện có |
  |---|---|---|---|---|
  | Test DirAcc | 68.25% | **68.76%** | +0.51pp | vẫn thua HAR-only (69.98%) |
  | Test R² | 0.7124 | **0.7159** | +0.0035 | **v.rợt kỷ lục gated-crossattn (0.7157)** |
  | Test QLIKE | 0.5598 | **0.5497** | -0.0101 (tốt hơn) | **KỶ LỤC MỚI của project** (trước: 0.557) |
  | Test RMSE | 0.002651 | **0.002635** | -0.000016 (tốt hơn) | |

  **Đây là baseline THẮNG rõ ràng trên cả 4 chỉ số so với đối chứng cùng panel** — kết quả tích
  cực đầu tiên sau ~10 lần thử null liên tiếp (xem `[[project_null_result_pattern_and_sota_pivot]]`
  trong memory).

- [x] Đối chiếu gate học được (theo mã) với `delta_qlike` ablation độc lập đã có sẵn
      (`results/ablation_derived_ticker_classification.json`).

  **KẾT QUẢ: vẫn KHÔNG khớp** — Pearson r=0.1416 (p=0.44), Spearman ρ=0.0733 (p=0.69), gần như
  giống hệt gate cũ (r=0.13, sai chiều). Mean gate của nhóm NEWS_ON (ablation nói tin giúp,
  n=11) = 0.561 vs NEWS_OFF (ablation nói tin hại, n=21) = 0.572 — KHÔNG có sự khác biệt có ý
  nghĩa, thậm chí sai chiều nhẹ. **Kết luận quan trọng:** dù đã cô lập gradient hoàn toàn (loại
  bỏ giả thuyết "shared weight gây nhiễu"), pattern học được VẪN không khớp với "độ hữu ích tin
  tức" đo bằng ablation độc lập — cải thiện hiệu năng tổng thể (bảng trên) có thể đến từ 1 cơ chế
  khác (regularization/scaling hữu ích chung), KHÔNG phải vì model "phát hiện đúng" mã nào cần
  tin như kỳ vọng ban đầu.
- [x] **Lưu ý quan trọng: gate CHƯA hội tụ trong 10 epoch** — epoch 9→10 vẫn có mã dịch >0.1
      (MSN, GAS, BVH, SSI, PDR) — cần train thêm mới biết gate có ổn định ở đâu, kết quả hiện tại
      là snapshot giữa quá trình học, không phải điểm hội tụ cuối cùng.
- [x] pytest pass (12/12), self-adversarial code review trước khi coi "done". Xem
      `code_review/code_review_2026-07-26.md`.

## 6. Out of scope

- Không kết hợp QLIKE loss hay đồ thị có hướng (baseline hôm qua) — 1 baseline = 1 biến thử
  nghiệm (Simplicity Gate).
- Không thêm regularization cho gate (vd L2 trên theta để chống saturation) — chưa có bằng chứng
  cần; nếu debug log cho thấy gate bão hoà quá nhanh (dying gate), ghi nhận làm follow-up, không
  tự thêm speculative fix trước.
- Gate là hằng số theo mã (không đổi theo ngày/bài báo cụ thể) — đây là giới hạn CÓ CHỦ ĐÍCH
  (đúng yêu cầu user: "áp dụng hay không áp dụng" theo MÃ, không phải theo bài báo).
