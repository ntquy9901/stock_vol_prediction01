# Requirements (Specify) — News Usefulness Ablation

**Baseline:** `2026-07-25_news_usefulness_ablation` · Theo SDD (CLAUDE.md §1.5).
**Nguồn:** user yêu cầu 2026-07-25 — thay vì mượn ΔR² per-ticker từ HGB/XGBoost (đã chứng minh
2 lần liên tiếp không chuyển giao sang LSTM-GNN, xem `2026-07-25_selective_news_gate_baseline`
và `2026-07-25_top3_news_gate_baseline`), **tự đo mức hữu ích của tin tức cho từng mã BẰNG CHÍNH
kiến trúc LSTM-GNN đang dùng**.

## 1. Phương pháp

So sánh 2 model, **CÙNG data pipeline, CÙNG split, CÙNG 32 mã** (đảm bảo bằng cách dùng chung
`create_dual_news_dataloaders` — chỉ khác `news_panel_path=None` vs panel thật; x_har/adj/y
không phụ thuộc panel nên 2 lần gọi cho ra windows/split giống hệt nhau):

- **Model A (HAR-only):** `ParallelLSTMGNN` nguyên bản (dùng fusion riêng của nó, KHÔNG freeze) —
  train mới hôm nay trên đúng pipeline 32-mã của baseline dual-group (khác timestamp June cũ,
  vốn dùng config/augmentation khác, không so sánh trực tiếp được).
- **Model B (all-ON):** checkpoint đã có sẵn `models/dual_group_news_2026-07-25_071825/best.pt`
  (40 epoch, hội tụ, TẤT CẢ 32 mã bật tin) — **KHÔNG train lại**, chỉ eval.

Với mỗi mã: `delta_metric = metric(B) - metric(A)` trên test set. Dùng **QLIKE và MSE** (liên
tục, ít nhiễu) làm tiêu chí chính; DirAcc (nhị phân, đã biết rất nhiễu với ~163 điểm/mã) chỉ để
tham khảo phụ.

## 2. Danh sách ON/OFF (kết quả bước ablation)

**[Bug tự phát hiện + fix]** Lần chạy đầu dùng checkpoint all-ON **40-epoch** (đã hội tụ) so với
HAR-only chỉ **10-epoch** — chênh lệch budget train làm sai lệch kết luận (26/32 mã "ON" — quá
nhiều, nghi ngờ chỉ phản ánh model train lâu hơn tốt hơn, không phải do tin tức). Đã sửa: dùng
checkpoint all-ON **10-epoch** (`models/dual_group_news_2026-07-25_011719/best.pt`) khớp đúng
epoch với HAR-only reference.

**Kết quả sau khi sửa (epoch-matched, đáng tin hơn):**

**NEWS-ON (11 mã):** HDB, HPG, MWG, NVL, PDR, PLX, SSI, VHM, VJC, VPB, VRE
**NEWS-OFF (21 mã):** ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, MBB, MSN, POW, SAB, SHB, SSB, STB,
TCB, TPB, VCB, VIB, VIC, VNM

11/11 mã ON cải thiện cả QLIKE lẫn MSE (đồng thuận cao); chỉ 4/11 cũng cải thiện DirAcc (xác nhận
DirAcc nhiễu, đúng như quan sát ở 2 baseline trước — QLIKE/MSE là tiêu chí chính đáng tin hơn).

**Khác biệt đáng chú ý so với EDA HGB/XGBoost:** ACB và VIB (2 trong 3 mã "mạnh nhất" theo EDA
cũ) lại rơi vào nhóm OFF ở đây; PLX (âm rõ theo EDA cũ) lại là ON. Xác nhận thêm: 2 phương pháp
đo "mã nào hưởng lợi từ tin tức" cho ra kết quả khác nhau đáng kể — không có gì mâu thuẫn về mặt
logic (2 model family khác nhau đo tín hiệu khác nhau), nhưng cần nhớ đây vẫn là single-seed,
có thể nhiễu.

## 3. Baseline áp dụng kết quả

Baseline riêng `2026-07-25_ablation_derived_gate_baseline` (tái dùng 100% cơ chế
`SelectiveGateNewsBaseline`, chỉ đổi ticker list) sẽ dùng danh sách này.

## 4. Lưu ý trung thực (đã cảnh báo trước khi chạy)

Đây vẫn là **single-seed** (train 1 lần mỗi model) — delta từng mã có thể nhiễu do test set nhỏ
(~163 cửa sổ/mã). Không đủ thời gian/ngân sách để chạy multi-seed trong phiên này; nếu kết quả
cuối vẫn mơ hồ, cần multi-seed mới kết luận chắc chắn được.

## 5. Training policy

10 epoch cho Model A (HAR-only) — khớp policy CLAUDE.md và khớp epoch-budget của các baseline
so sánh khác hôm nay.
