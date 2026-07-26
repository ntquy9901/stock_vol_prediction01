# Requirements (Specify) — Selective News Gate Baseline

**Baseline:** `2026-07-25_selective_news_gate_baseline` · Theo SDD (CLAUDE.md §1.5).
**Nguồn:** `docs/suggestion/2026-07-25_professor_report.md` (EDA: HGB/XGBoost per-ticker ΔR² khi
thêm news, 4 horizon) + quyết định user (loại SHB) 2026-07-25.

## 1. Vấn đề đang giải quyết

10 baseline tích hợp tin tức trước đều áp dụng nhánh news **đồng loạt cho cả 30 mã VN30**, dù
EDA (per-ticker, HGB/XGBoost, t+5) cho thấy tin tức chỉ thực sự cải thiện R² ở **một tập con** mã.
Với các mã còn lại, nhánh news có thể chỉ thêm nhiễu (tham số tốn mà không có tín hiệu bù lại —
đúng hiện tượng "text collapse" đã ghi nhận ở 10 baseline trước).

**Ý tưởng:** thay vì học 1 gate chung cho mọi mã (như Market Fallback's `has_news` theo NGÀY),
dùng **mask cố định theo TỪNG MÃ** dựa trên domain knowledge (EDA) — tắt hẳn nhánh news cho các
mã đã biết trước là không có lợi.

## 2. Danh sách mã (từ EDA t+5, đã user xác nhận loại SHB)

Trích cột ΔR² tại **t+5** (không dùng trung bình 4 horizon — tránh lẫn tín hiệu t+1/t+10/t+22)
từ Phụ lục báo cáo EDA.

**NEWS-ON (22 mã, ΔR²@t+5 ≥ 0.01):**
ACB(1.416), MWG(0.537), VIB(0.428), TPB(0.424), SAB(0.267), VJC(0.234), MBB(0.168), POW(0.164),
TCB(0.149), MSN(0.126), HPG(0.080), VIC(0.077), SSB(0.074), BID(0.069), SSI(0.049), STB(0.042),
FPT(0.037), VCB(0.032), CTG(0.026), GVR(0.021), VNM(0.015), HDB(0.012)

**NEWS-OFF / bias=0 (10 mã):**
- SHB — loại theo yêu cầu user (dù ΔR²@t+5=+1.807 cao nhất, nghi vấn spurious/time-proxy giống
  hiện tượng đã ghi nhận ở t+22, cần kiểm chứng riêng trước khi tin dùng).
- GAS(-0.199), PLX(-0.038), NVL(-0.029), BVH(-0.016) — ΔR²@t+5 âm rõ.
- VHM(0.006), BCM(0.008), PDR(0.004) — ΔR²@t+5 dương nhưng quá nhỏ (nhiễu, không đáng tin).
- **VPB, VRE** — [phát hiện khi chạy thật lần đầu] pipeline train thực tế của project dùng
  **32 mã chung** (`_load_raw_stock_data`/`_split_raw_data_by_date`), nhiều hơn 2 mã so với 30 mã
  EDA đã phân tích. Không có bằng chứng ΔR² cho 2 mã này — mặc định OFF (an toàn, đúng tinh thần
  "chỉ bật news khi có bằng chứng dương").

**[NEEDS CLARIFICATION — đã quyết định mặc định]:** danh sách này đến từ model HGB/XGBoost
(tree-based, per-ticker, feature set price+news_adv_full), KHÔNG phải từ chính kiến trúc
LSTM-GNN đang dùng. Tín hiệu ΔR² dương ở đây KHÔNG đảm bảo cũng giúp ích trong nhánh
`NewsFeatureLSTM` — đây là giả thuyết hợp lý nhất hiện có (domain knowledge), cần bản thân kết
quả train mới xác nhận đúng/sai.

## 3. Kiến trúc — tái dùng gần như 100% baseline dual-group (07-25)

KHÔNG rebuild data/panel — dùng thẳng `data/features/dual_group_news_panel.parquet` (đã fix
leakage) đã có sẵn. KHÔNG sửa `2026-07-25_dual_group_news_embedding_baseline` (import read-only
`dataset_dual_news.py`). Chỉ thêm: mask cố định theo ticker, nhân vào `news_rep` SAU
`NewsFeatureLSTM`, TRƯỚC khi concat vào fusion — đảm bảo "0 tuyệt đối" (không phụ thuộc bias của
LSTM) cho 8 mã NEWS-OFF.

## 4. Success criteria (go/no-go)

- **Go:** model train ổn định 10 epoch, đủ 6 metric, mask hoạt động đúng (test riêng: xáo trộn
  `x_news` của 1 mã NEWS-OFF không đổi output của mã đó).
- **So sánh:** DirAcc/R²/QLIKE so với dual-group baseline KHÔNG mask (68.50-68.71% tuỳ epoch) và
  HAR-only (69.98%). Không cần vượt để coi "done" — đây là thí nghiệm kiểm chứng domain-knowledge
  từ EDA khác model, không phải production gate.
- **Đặc biệt chú ý per-stock DirAcc** (không chỉ DirAcc tổng thể) — nếu domain knowledge đúng, kỳ
  vọng 22 mã NEWS-ON cải thiện so với baseline không mask, trong khi 8 mã NEWS-OFF không đổi/nhẹ
  tốt hơn (do giảm nhiễu).

## 5. Training policy

10 epoch thử nghiệm đầu (CLAUDE.md). Chỉ train thêm (20/40) nếu user duyệt sau khi xem kết quả 10ep.
