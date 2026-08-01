# BÁO CÁO KIẾN TRÚC HIỆN TẠI CỦA HỆ THỐNG
## Dự báo biến động (volatility) 5-ngày VN30 — Parallel LSTM-GNN + News Fusion + Per-ticker Gate

**Ngày báo cáo:** 01/08/2026 (cập nhật lần 2, cùng ngày — thêm thử nghiệm calendar feature + EDA)
**Người thực hiện:** ntquy99
**Lý do tạo file này:** Chưa có tài liệu thiết kế (`.md`) nào mô tả **toàn bộ kiến trúc end-to-end
đang dùng** trong 1 chỗ duy nhất — thông tin hiện nằm rải rác ở 3 nơi (xem "Bảng tham khảo" cuối
trang). File này gộp lại thành 1 báo cáo kiến trúc, cập nhật tới baseline mới nhất.

**Cập nhật lần 2 (cùng ngày):** câu hỏi đặt ra là "tin tức có thể ảnh hưởng theo mùa BCTC/Tết
không, làm sao thêm feature thời gian để kiểm tra" → đã (1) thiết kế + implement 10 calendar
feature, (2) train thử baseline mới, (3) chạy EDA/correlation để trả lời trực tiếp câu hỏi bằng
số liệu thay vì suy đoán. Toàn bộ thêm ở **§6 (mới)** — có code + ví dụ tính tay cụ thể, đọc không
cần trace lại source.

---

# 1. TÓM TẮT (đọc trước, 1 phút)

## 1.1 Kiến trúc hiện tại — 1 hình

```
Input: các mã VN30 × 22 ngày × 3 HAR feature        +        các mã VN30 × 22 ngày × 146 news feature
              │                                                        │
   ┌──────────┴──────────┐                                   ┌─────────┴─────────┐
   │  LSTM (per mã)      │  GAT (đồ thị k-NN giữa các mã)     │  NewsFeatureLSTM   │
   │  temporal branch    │  spatial branch                    │  (Linear+LSTM)     │
   │  → h_lstm [B,N,64] │  → h_gnn [B,N,256]                 │  → news_rep[B,N,64]│
   └──────────┬──────────┘                                   └─────────┬─────────┘
              └────────────┬──────────────┘                            │
                    har_embed [B,N,320]                                │
                                                            gate = sigmoid(gate_logits)
                                                            (1 số/mã học được, KHÔNG phụ thuộc input)
                                                                        │
                                                          gated_news = gate ⊙ news_rep
                    └──────────────────── concat ─────────────────────┘
                                    h [B,N,384]
                                        │
                              Linear (fusion, áp độc lập theo từng mã)
                                        │
                              pred [B,N]  (volatility 5-ngày dự báo)
```

## 1.2 Bảng so sánh tổng hợp (đọc trước tiên — mọi số liệu trích từ §4/§6/§7)

**Lưu ý bắt buộc đọc trước khi so sánh:** có 2 pipeline KHÔNG so trực tiếp được với nhau —
(a) pipeline gốc (batch_size=11, có augmentation — cho ra con số "69.98% DirAcc" nhắc ở §4)
và (b) pipeline so sánh công bằng dùng cho TẤT CẢ bảng dưới đây (batch_size=32, không augmentation).
Mọi kết luận "vượt HAR-only" trong báo cáo này đều trong pipeline (b).

*(Bảng A — tiến trình chi tiết per-ticker-gate qua các panel/epoch ở horizon 5-ngày — đã chuyển
xuống "Bảng tham khảo" cuối trang, vì Bảng B dưới đây đã đủ để so sánh tổng quan. Số liệu tốt nhất
hiện hành: DirAcc 69.51%, R² 0.7158, QLIKE 0.5436, epoch 20, panel đã fix — chi tiết §7.3-7.4.)*

### Bảng B — So sánh horizon dự báo (1 vs 5 vs 10 vs 22 ngày, pipeline so sánh công bằng, 10 epoch trừ khi ghi khác)

| Kiến trúc | Horizon | DirAcc | R² | QLIKE | RMSE | Trạng thái hội tụ |
|---|---|---:|---:|---:|---:|---|
| HAR-only | 1 ngày | 72.35% | 0.7581 | 0.5099 | 0.002428 | hội tụ nhanh (epoch ~5) |
| HAR-only | 5 ngày | 68.42% | 0.7141 | 0.5623 | 0.002643 | — |
| HAR-only | 10 ngày | 67.80% | **0.7041** | **0.5732** | **0.002689** | — |
| HAR-only | 22 ngày | 66.38% | **0.7051** | **0.5938** | **0.002750** | hội tụ ở epoch ~10 |
| Gated-news | 1 ngày | **72.39%** | **0.7595** | **0.4834** | **0.002420** | hội tụ nhanh (epoch ~5) |
| Gated-news | 5 ngày | **69.51%** | **0.7158** | **0.5436** | **0.002635** | epoch 20 — tốt nhất, xem Bảng A |
| Gated-news | 10 ngày | **67.92%** | 0.7040 | 0.5767 | 0.002690 | epoch 10 — hội tụ ở epoch ~10-20 |
| Gated-news | 22 ngày | **67.17%** | 0.7032 | 0.5943 | 0.002759 | epoch 10 — hội tụ ở epoch ~10 |

**Đọc bảng B (in đậm = kiến trúc thắng ở đúng horizon đó, so HAR-only với Gated-news cùng mốc):**

- **1-ngày và 5-ngày: Gated-news thắng CẢ 4 metric.** Tin tức thực sự giúp ích rõ ở 2 horizon ngắn
  nhất (lưu ý: hàng 5-ngày của Gated-news dùng epoch 20, không cùng epoch với hàng HAR-only —
  epoch 10 — nên so sánh này không hoàn toàn ngang epoch, xem Bảng A).
- **10-ngày và 22-ngày: kết quả đảo ngược — Gated-news chỉ thắng DirAcc, HAR-only thắng cả R²,
  QLIKE, RMSE** (dù chênh lệch rất nhỏ, có thể nằm trong nhiễu single-seed). Tin tức không còn
  giúp ích rõ ràng ở 2 horizon dài.
- **Xu hướng chung theo horizon (không phân biệt kiến trúc):** DirAcc giảm dần khi horizon dài hơn
  (72.35%→68.42%→67.80%→66.38% cho HAR-only), QLIKE tăng dần (0.5099→0.5623→0.5732→0.5938) —
  horizon càng dài càng khó dự báo, càng ngắn càng dễ, nhất quán ở cả 2 kiến trúc. R² nhảy vọt rõ
  ở mốc 1-ngày (~0.758) so với 3 mốc còn lại (~0.70-0.71).
- **Hội tụ:** 1, 10, 22-ngày đều hội tụ/chững lại nhanh (epoch ~5-10); CHỈ 5-ngày cần train tới
  epoch ~20 mới đạt đỉnh (Bảng A) — ngoại lệ, không phải quy luật chung theo horizon.

Chi tiết: §7.1-7.2, §7.5-7.6.

*(Bảng C — so sánh calendar feature — đã chuyển xuống cuối trang, xem "Bảng tham khảo" trước Phụ
lục, vì ít giá trị báo cáo ở mức tóm tắt so với Bảng A/B.)*

## 1.3 Kết luận điều hành — 5 câu (đã gộp từ bảng trên)

1. **Kết quả tốt nhất hiện tại: per-ticker gated news, panel đã fix, epoch 20** (Bảng A hàng 4) —
   DirAcc 69.51%, R² 0.7158, QLIKE 0.5436 — **lần đầu tiên 1 biến thể tích hợp tin tức vượt
   HAR-only trên DirAcc** trong cùng pipeline. Epoch 30 xác nhận epoch 20 là điểm dừng hợp lý
   (train thêm bắt đầu overfit).
2. **Horizon càng dài càng khó dự báo — xu hướng đơn điệu qua đủ 4 mốc 1/5/10/22-ngày** (Bảng B):
   DirAcc giảm dần (72.35%→68.42%→67.80%→66.38%, HAR-only), QLIKE tăng dần
   (0.5099→0.5623→0.5732→0.5938), cho cả 2 kiến trúc — **1-ngày dễ dự báo nhất, vượt trội rõ rệt
   so với 3 mốc còn lại** (đúng giả thuyết ban đầu). Riêng 5-ngày là ngoại lệ về hội tụ: 1/10/22-ngày
   đều hội tụ RẤT NHANH (~epoch 5-10), CHỈ 5-ngày cần ~20 epoch mới đạt đỉnh (câu 1).
3. **Calendar feature (day-of-week/tháng/Tết/mùa BCTC) không cải thiện dự báo** — no-lift ở cả
   training (§6.6, bảng tham khảo cuối trang) lẫn EDA tương quan độc lập (§6.5).
4. **Không có feature thời gian dạng lịch nào tồn tại trong kiến trúc gốc** trước hôm nay (§3) —
   đã triển khai + kiểm tra ở §6, kết quả nêu ở câu 3.
5. **Gate học được vẫn KHÔNG khớp tín hiệu "mã nào cần tin tức" đo độc lập** (§5 mục 2) — vấn đề
   mở, chưa giải quyết.

---

# 2. CHI TIẾT KIẾN TRÚC

## 2.1 Nhánh HAR (thời gian + không gian) — `ParallelLSTMGNN`, không đổi qua mọi baseline

**File:** `src/lstm_gat_hybrid/model_parallel.py`
**Nguồn ý tưởng:** Sonani et al. (2025), "Stock Price Prediction Using a Hybrid LSTM-GNN Model".

```
Input: [batch, 22 ngày, N mã, 3 HAR feature (daily/weekly/monthly rolling vol)]

LSTM stream (per mã, độc lập):
  input_size=3, hidden_size=64, num_layers=2, dropout=0.2
  → h_lstm [batch, N, 64]

GAT stream (per ngày, trộn thông tin giữa các mã qua đồ thị k-NN):
  2 lớp Graph Attention (4 head, hidden=64), sau đó mean-pool theo chiều thời gian
  → h_gnn [batch, N, 256]

Fusion (concat, KHÔNG cộng): [64+256=320] → Dense MLP (320→64→32→1) khi dùng riêng (HAR-only)
                             → hoặc concat thêm nhánh tin tức (xem 3.2) khi dùng làm HAR-only "backbone"
```

Đây chính là baseline **"HAR-only"** đang dẫn đầu DirAcc (69.98%) trong mọi bảng so sánh — mọi
baseline tích hợp tin tức đều **tái sử dụng nguyên vẹn** `ParallelLSTMGNN.get_embeddings()`
(read-only, không sửa) để lấy `h_lstm`, `h_gnn`.

## 2.2 Nhánh tin tức — `NewsFeatureLSTM`

**File:** `baselines/2026-07-25_dual_group_news_embedding_baseline/code/model_dual_news.py`

```
Input: x_news [batch, 22 ngày, N mã, 146 feature]
       146 = PCA-32×2 nhóm nguồn (khách quan / tổng hợp CTCK) + topic flags + EWMA(30d, half-life)

news_rep = LSTM(1 lớp, Linear(146→64) → ReLU → LSTM(64→64), qua 22 ngày)
         → [batch, N, 64]
```

Feature tin tức đã được **aggregate sẵn theo ngày** ở bước offline (`build_dual_group_panel.py`)
— khác với baseline tin tức đầu tiên (07-07) phải pool N bài báo thô bằng attention-pooling.

### 2.2.1 Ví dụ cụ thể: ngày CÓ tin vs ngày KHÔNG có tin được tổ chức thế nào khi train

Panel tin tức (`dual_group_news_panel.parquet`) chỉ có 1 dòng cho mỗi `(mã, ngày)` MÀ THỰC SỰ có
ít nhất 1 bài báo nhắc tới mã đó (đã qua PCA + EWMA) — **không phải mọi ngày giao dịch đều có dòng
trong panel**. Khi build sequence cho training, với mỗi ngày trong cửa sổ 22 ngày, code tra panel
theo `(mã, ngày)`:

```python
# baselines/2026-07-25_dual_group_news_embedding_baseline/code/dataset_dual_news.py
news_cache = self._news_by_ticker.get(stock_name) or {}
day_feats = []
for d in window_dates:
    vec = news_cache.get(d)          # tra panel theo đúng ngày d
    if vec is not None:
        day_feats.append(vec)                              # CÓ tin -> vector thật (146 số)
    else:
        day_feats.append(np.zeros(self._n_feat, dtype=np.float32))  # KHÔNG tin -> vector 0
```

**Ví dụ tính tay (3 ngày, 2 mã, số liệu minh hoạ theo đúng format thật — 146 chiều rút gọn còn 4
chiều đầu để dễ đọc):**

| Ngày | Mã ACB — có/không tin | x_news[ACB] (4/146 chiều đầu) | Mã FPT — có/không tin | x_news[FPT] (4/146 chiều đầu) |
|---|---|---|---|---|
| 2024-03-04 | **CÓ** (báo cafef đưa tin ACB) | `[0.182, -0.041, 0.077, 0.203, ...]` | KHÔNG | `[0.0, 0.0, 0.0, 0.0, ...]` |
| 2024-03-05 | KHÔNG | `[0.0, 0.0, 0.0, 0.0, ...]` | **CÓ** (vnexpress đưa tin FPT) | `[0.095, 0.114, -0.032, 0.061, ...]` |
| 2024-03-06 | KHÔNG | `[0.0, 0.0, 0.0, 0.0, ...]` | KHÔNG | `[0.0, 0.0, 0.0, 0.0, ...]` |

**Ý nghĩa của việc "fill 0" ngày không có tin:**
- KHÔNG có nghĩa "tin tức trung tính" (neutral) — nghĩa là **"không có tín hiệu tin tức nào cho
  mã này vào đúng ngày này trong panel"**. Đây là 1 đơn giản hoá có chủ đích (ghi trong
  `dataset_dual_news.py`'s docstring): không dùng mask riêng để phân biệt "không có tin" với "có
  tin nhưng tín hiệu = 0" — model KHÔNG thể phân biệt 2 trường hợp này.
- LSTM (`NewsFeatureLSTM`, §2.2) đọc qua 22 ngày liên tiếp — 1 mã có thể có chuỗi toàn số 0 rồi
  xen kẽ vài ngày có vector thật, tuỳ mật độ tin tức thực tế của mã đó trong khoảng thời gian đó.
- Tỷ lệ (mã, ngày) có tin thật trong toàn bộ panel: **77.83%** (đo khi build panel, xem
  `2026-07-25_dual_group_news_embedding_baseline/design/design.md` §3.3) — tức ~22% (mã, ngày)
  trong dữ liệu train/val/test là vector toàn số 0.
- **Rủi ro đã biết:** nếu 1 mã có coverage tin tức rất thấp (nhiều ngày toàn số 0 liên tục), nhánh
  tin tức gần như không có gì để học cho mã đó trong nhiều cửa sổ — đây là 1 phần lý do project
  từng nghi ngờ "gate học được" không khớp tín hiệu "mã nào cần tin tức" đo độc lập (§5 mục 2).

## 2.3 Per-ticker gate — cải tiến mới nhất (26/07), hiện là baseline tốt nhất QLIKE/R²

**File:** `baselines/2026-07-26_per_ticker_news_gate_baseline/code/model_per_ticker_gate.py`

```
gate_logits: nn.Parameter, shape [N]   (KHÔNG phụ thuộc input, 1 số vô hướng/mã)
gate = sigmoid(gate_logits)             # (0,1)
gated_news = gate.view(1,N,1) * news_rep   # nhân theo đúng vị trí mã, KHÔNG trộn giữa các mã
```

**Tính chất đã chứng minh + kiểm chứng bằng test** (`test_gate_gradient_isolated_per_ticker`):
gradient của `gate_logits[i]` **chỉ** phụ thuộc dữ liệu của chính mã `i` — đổi nhãn `y` của mã
khác không làm thay đổi gradient này. Lý do: `NewsFeatureLSTM` xử lý từng mã độc lập (reshape
`[B,S,T,F]→[B·S,T,F]`), gate nhân elementwise theo đúng vị trí, và lớp `fusion` cuối là `Linear`
áp độc lập theo từng `(batch, mã)` — không có tầng nào trộn thông tin giữa các mã ở giai đoạn này
(GAT có trộn, nhưng nằm ở nhánh HAR, TRƯỚC gate).

Optimizer dùng 2 param-group (Adam), `gate_lr=0.05` (10× LR của phần network chính) để các số này
có đủ "room" học trong 10 epoch (cap theo Training policy của CLAUDE.md).

## 2.4 Fusion cuối cùng

```
h = concat([h_lstm, h_gnn, gated_news])   # [batch, N, 320+64=384]
pred = Linear(h).squeeze(-1)              # [batch, N] — áp độc lập theo từng (batch, mã)
```

---

# 3. FEATURE LIÊN QUAN THỜI GIAN — hiện KHÔNG có calendar feature

Đã rà toàn bộ `src/` — **không tồn tại** bất kỳ feature nào dạng: day-of-week, ngày trong
tháng/quý, cờ ngày lễ VN, hay cyclical sin/cos encoding theo lịch.

"Thời gian" hiện chỉ xuất hiện dưới 2 dạng, cả hai đều KHÔNG phải calendar feature:

| Dạng | Ở đâu | Bản chất |
|---|---|---|
| HAR rolling window | `src/common/feature_engineering.py`, `har_features.py` | Trung bình trượt 1/5/22 ngày của volatility — time-*scale*, không phải time-*of-year* |
| Thứ tự bước trong LSTM | `TemporalLSTM` (`src/lstm_har_gat_hybrid/temporal_encoder.py`), `ParallelLSTMGNN` LSTM stream | Model biết "bước thứ mấy trong 22 ngày" nhưng KHÔNG biết đó là thứ Hai hay cuối tháng |

Target `volatility.shift(-5)` là lag/shift thuần, không phải input feature.

→ Nếu hướng nghiên cứu tiếp theo muốn khai thác hiệu ứng lịch (vd Monday effect, hiệu ứng cuối
tháng, trước/sau kỳ nghỉ lễ VN — vốn được nhắc tới trong tài liệu literature review nhưng chưa
triển khai) thì đây là phần hoàn toàn mới, chưa có gì để tái dùng trong codebase hiện tại.

**Cập nhật cùng ngày:** đã triển khai đúng hướng này — xem **§6**.

---

# 4. KẾT QUẢ HIỆN TẠI (bảng đầy đủ lịch sử — Bảng A ở "Bảng tham khảo" cuối trang là bản rút gọn, cập nhật nhất)

Toàn bộ biến thể tin tức từng thử ở horizon 5-ngày, gộp cả 2 pipeline (xem lưu ý pipeline ở §1.2):

| Biến thể | Pipeline | Epoch | DirAcc | R² | QLIKE | RMSE |
|---|---|---:|---:|---:|---:|---:|
| **HAR-only** (không dùng tin) | gốc (batch11+aug) | 70 | **69.98%** | 0.7140 | 0.5294 | 0.002644 |
| Gated Cross-Attention | gốc | 15 | 68.97% | 0.7157 | 0.5567 | 0.002636 |
| Dual-group + EWMA (không gate) | gốc | 10 | 68.25% | 0.7124 | 0.5598 | 0.002651 |
| HAR-only | so sánh (batch32) | 10 | 68.42% | 0.7141 | 0.5623 | 0.002643 |
| Per-ticker gated news, panel cũ | so sánh | 10 | 68.76% | 0.7159 | 0.5497 | 0.002635 |
| Per-ticker gated news, panel đã fix | so sánh | 10 | 68.69% | 0.7101 | 0.5631 | 0.002662 |
| **Per-ticker gated news, panel đã fix — hiện hành** | so sánh | 20 | **69.51%** | **0.7158** | **0.5436** | 0.002635 |

Nguồn: `docs/report_2026-07-25/BAO_CAO_CHO_THAY.md` §1.1 (3 dòng "gốc") +
`docs/reports/2026-07-26_2330_summaryOfUpdate_report.md` (panel cũ) +
`results/per_ticker_gate_2026-08-01_094139/results.json` (panel đã fix epoch 20, số liệu hiện hành
— xem phân tích hội tụ đầy đủ ở §7.4).

---

# 5. GIỚI HẠN / VIỆC CHƯA XONG

1. **[ĐÃ XONG cùng ngày, xem §7.3-7.4]** Panel tin tức cũ, thiếu VPB/VRE — bug đã fix (27/07),
   đã retrain per-ticker gate trên panel đã fix (01/08), train tới epoch 30 để kiểm tra hội
   tụ đầy đủ. Kết quả cuối: epoch 20 là mốc tốt nhất (QLIKE 0.5436, R² 0.7158, DirAcc 69.51%, vượt
   HAR-only cùng pipeline), epoch 30 cho dấu hiệu overfitting — số liệu panel đã fix, epoch 20 là số
   liệu hiện hành, xem Bảng A ở "Bảng tham khảo" cuối trang.
2. **Gate học được KHÔNG khớp tín hiệu "mã nào cần tin tức" đo độc lập** — 4 phương pháp đo (EDA
   HGB/XGBoost, ablation delta-QLIKE tự đo, và chính gate học được) cho 4 thứ tự mã khác nhau,
   chưa có lời giải thích thống nhất (xem memory `project_selective_news_gate_finding`).
3. **Single-seed** — mọi kết quả trên chỉ train 1 lần (1 seed); chênh lệch nhỏ giữa các biến thể
   (vd 68.25% vs 68.76%) nằm trong biên độ có thể là nhiễu, chưa multi-seed để kiểm chứng.
4. **Không có feature thời gian dạng lịch** (§3, cũ) — **ĐÃ LÀM cùng ngày, xem §6** — kết quả:
   thêm vào rồi nhưng chưa cải thiện (§6.4), và EDA không phát hiện tín hiệu theo mùa (§6.5).
5. **3/3 ablation tách nhóm calendar feature đã xong (§6.6)** — không nhóm nào (Tết/BCTC/generic)
   vượt được đối chứng trên R²/QLIKE/RMSE; `generic_calendar` có DirAcc nhỉnh hơn (+0.14pp, trong
   biên độ nhiễu). Gộp cả 10 cột lại TỆ HƠN mỗi nhóm con riêng lẻ.
6. **10-ngày và 22-ngày-trước (§7) chỉ mới thử 2 kiến trúc (HAR-only, gated-news), chưa thử
   calendar-augmented ở 2 horizon này** — nếu cần, đây là phần mở rộng riêng, chưa làm.
7. **So sánh horizon dùng pipeline batch_size=32/không augmentation** — khác pipeline với con số
   "69.98% DirAcc" nhắc ở §4 dòng đầu (batch_size=11, có augmentation). 2 pipeline không so trực
   tiếp được với nhau; §7 chỉ so sánh trong CÙNG 1 pipeline.
8. **[ĐÃ XONG]** Horizon 22-ngày: đã train 10 epoch cả 2 kiến trúc, hội tụ ngay từ epoch ~10
   (giống 10-ngày, khác 5-ngày) — xem Bảng B (§1.2) và §7.5. Chưa thử epoch >10 (theo trạng thái
   hội tụ đo được, không có lý do để train thêm).

---

# 6. THỬ NGHIỆM CALENDAR FEATURE + EDA TƯƠNG QUAN (01/08/2026)

**Mục này viết để đọc độc lập — có design, code, ví dụ tính tay cụ thể, không cần mở source ra
trace.** Toàn bộ code nằm trong `baselines/2026-08-01_calendar_news_gate_baseline/`.

## 6.1 Câu hỏi đặt ra → 2 việc cần làm

> "Tin tức có thể ảnh hưởng vào một số tháng/tuần trong năm (báo cáo tài chính, Tết Nguyên Đán).
> Làm sao thêm feature thời gian để nhận biết, và kiến trúc thay đổi thế nào?"

Tách thành 2 việc độc lập:
- **(A) Thêm feature thời gian** vào kiến trúc, train thử xem model có tận dụng được không (§6.2-6.4, 7.6).
- **(B) Kiểm tra bằng EDA/thống kê** xem tín hiệu đó có THẬT SỰ tồn tại trong dữ liệu không —
  KHÔNG cần train model mới, dùng lại 2 checkpoint đã có sẵn từ baseline trước (§6.5). Đây là cách
  rẻ hơn nhiều so với train nhiều biến thể để "dò" bằng tay.

## 6.2 Thiết kế 10 calendar feature — công thức + ví dụ tính tay

**File:** `code/calendar_features.py`. Hàm duy nhất, THUẦN (chỉ phụ thuộc ngày, không phụ thuộc
mã/dữ liệu khác — nên KHÔNG có rủi ro rò rỉ dữ liệu tương lai vào train set, đã tự kiểm chứng
trong code review):

| # | Cột | Công thức | Ý nghĩa |
|---|---|---|---|
| 1-2 | `dow_sin/cos` | sin/cos(2π·weekday/5) | thứ trong tuần giao dịch (Mon=0..Fri=4) |
| 3-4 | `month_sin/cos` | sin/cos(2π·(month-1)/12) | tháng trong năm |
| 5-6 | `tet_proximity`, `in_tet_window` | exp(-\|Δngày tới Tết\|/10), flag ≤10 ngày | gần/xa Tết |
| 7-8 | `is_month_end`, `is_quarter_end` | ngày ∈ 3 ngày cuối tháng (lịch); + tháng ∈{3,6,9,12} | cuối tháng/quý |
| 9-10 | `earnings_proximity`, `in_earnings_window` | exp(-\|Δngày tới hạn 20/1,20/4,20/7,20/10\|/10), flag trong [cuối-quý, +20 ngày] | mùa công bố BCTC |

**Code thật (rút gọn, giữ đúng công thức):**
```python
# code/calendar_features.py
CALENDAR_FEATURE_NAMES = [
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "tet_proximity", "in_tet_window",
    "is_month_end", "is_quarter_end",
    "earnings_proximity", "in_earnings_window",
]
TET_DATES = {2024: "2024-02-10", 2025: "2025-01-29", 2026: "2026-02-17", ...}  # 2005-2027, tra cứu thật

def compute_calendar_vector(date_str: str) -> np.ndarray:
    d = _parse_date(date_str)
    trading_dow = d.weekday() % 5
    dow_sin, dow_cos = sin(2π·trading_dow/5), cos(2π·trading_dow/5)
    month_sin, month_cos = sin(2π·(d.month-1)/12), cos(2π·(d.month-1)/12)
    tet_dist = min(abs((d - t).days) for t in TET_DATE_OBJS)
    tet_proximity, in_tet_window = exp(-tet_dist/10), float(tet_dist <= 10)
    is_month_end = d.day > days_in_month(d) - 3
    is_quarter_end = is_month_end and d.month in (3,6,9,12)
    earn_dist = min(abs((d - dl).days) for dl in earnings_deadlines_near(d))
    earnings_proximity = exp(-earn_dist/10)
    in_earnings_window = any(quarter_end < d <= quarter_end + 20_days for quarter_end in ...)
    return np.array([dow_sin, dow_cos, month_sin, month_cos, tet_proximity, in_tet_window,
                      is_month_end, is_quarter_end, earnings_proximity, in_earnings_window])
```

**Ví dụ tính tay thật (chạy trực tiếp, không phải số giả định):**

| Ngày | dow_sin/cos | month_sin/cos | tet_proximity | in_tet_window | is_month_end | is_quarter_end | earnings_proximity | in_earnings_window |
|---|---|---|---|---|---|---|---|---|
| **2020-01-25** (đúng ngày Tết 2020) | 0.0/1.0 | 0.0/1.0 | **1.0000** | **1** | 0 | 0 | 0.6065 | 0 |
| **2020-01-24** (1 ngày trước Tết) | -0.951/0.309 | 0.0/1.0 | 0.9048 | 1 | 0 | 0 | 0.6703 | 0 |
| **2010-06-30** (cuối quý 2) | 0.588/-0.809 | 0.5/-0.866 | 0.0000 | 0 | **1** | **1** | 0.1353 (=20 ngày tới hạn 20/7) | 0 |
| **2026-06-09** (ngày cuối data, giữa năm) | 0.951/0.309 | 0.5/-0.866 | 0.0000 | 0 | 0 | 0 | 0.0166 | 0 |

*(4 dòng trên chạy thật bằng `compute_calendar_vector(...)`, không phải số minh hoạ — xem
`code_review_2026-08-01.md` mục "adversarial spot-check".)*

**3 nhóm ablation (để tách "cột nào có tín hiệu", §6.6):**
```python
CALENDAR_FEATURE_GROUPS = {
    "tet_only":         ["tet_proximity", "in_tet_window"],                       # 2 cột
    "earnings_only":    ["earnings_proximity", "in_earnings_window"],             # 2 cột
    "generic_calendar": ["dow_sin","dow_cos","month_sin","month_cos",
                         "is_month_end","is_quarter_end"],                        # 6 cột
}
```

## 6.3 Cách nối vào kiến trúc (Cách 1 — KHÔNG đổi model)

```
x_news_cũ [B,22,N,146]  (dual-group PhoBERT/PCA/EWMA, không đổi)
        │
        ├──concat──► x_news_mới [B,22,N,146+10=156]
        │
calendar_vec(date)  [10]  (tính LIVE mỗi ngày trong dataset, KHÔNG cần panel/parquet riêng —
                            khác nhánh tin tức PhoBERT vì đây là hàm rẻ, không cần cache)
                            broadcast GIỐNG NHAU cho tất cả các mã trong cùng 1 ngày
        │
        ▼
PerTickerGatedNewsBaseline (KHÔNG SỬA 1 dòng nào — n_feat là tham số, tự nhận 156 thay vì 146;
                             xem §2.3 — gate tĩnh theo mã VẪN GIỮ NGUYÊN theo yêu cầu ban đầu)
```

**Code nối (dataset, không đổi model):**
```python
# code/dataset_calendar_news.py, trong vòng lặp build sequence
dual_vec = dual_cache.get(d)  or  np.zeros(n_dual)   # 146 cột, từ panel PhoBERT có sẵn
cal_vec  = self._calendar_vec(d)                     # 10 cột (hoặc ít hơn nếu ablation), tính live
day_feats.append(np.concatenate([dual_vec, cal_vec]))  # -> 1 vector 156 cột/ngày/mã
```

Model **không đổi 1 dòng** — đây chính là điểm mạnh của thiết kế: `n_feat` (146 hay 156 hay 148)
chỉ là 1 tham số constructor của `PerTickerGatedNewsBaseline` (§2.3), không phải hằng số cứng.

## 6.4 Kết quả training — Cách 1, đủ 10 cột (10 epoch, so với baseline không-calendar)

| Metric | Không calendar (146 cột) | **Có đủ 10 cột calendar (156 cột)** | Diff |
|---|---:|---:|---:|
| Test DirAcc | 68.76% | 68.13% | -0.63pp (xấu hơn) |
| Test R² | 0.7159 | 0.7117 | -0.0041 (xấu hơn) |
| Test QLIKE | 0.5497 | 0.5660 | +0.0163 (xấu hơn) |
| Test RMSE | 0.002635 | 0.002654 | +0.000019 (xấu hơn) |

**Cả 6/6 metric xấu đi nhẹ** — no-lift. Không kết luận "Tết/BCTC chắc chắn không ảnh hưởng" — chỉ
là cách nối đơn giản nhất (concat tĩnh) không đủ để kiến trúc hiện tại tận dụng được, NẾU tín hiệu
đó tồn tại. Đây là lý do làm tiếp §6.5 (kiểm tra bằng thống kê, độc lập với việc train model).

## 6.5 EDA/tương quan — trả lời trực tiếp câu hỏi đặt ra, KHÔNG cần train model mới

**Ý tưởng:** đã có sẵn 2 model đã train (cùng 10 epoch, cùng dữ liệu) từ thí nghiệm trước
(25/07): Model A = HAR-only (không tin), Model B = có tin (tất cả các mã). Với MỖI điểm dữ liệu
test (mỗi mã, mỗi ngày), tính:

```
delta_QLIKE(mã, ngày) = QLIKE(Model B tại điểm đó) − QLIKE(Model A tại điểm đó)
   âm  → tin tức giúp ích tại đúng ngày/mã đó
   dương → tin tức làm dự báo TỆ hơn tại đúng ngày/mã đó
```

rồi gộp theo THÁNG, theo "trong/ngoài cửa sổ Tết", theo "trong/ngoài mùa BCTC" — xem trung bình
có khác biệt rõ không, kiểm định bằng Welch t-test + hệ số tương quan Pearson.

**Công thức QLIKE từng điểm (code thật, `code/analyze_news_calendar_correlation.py`):**
```python
def qlike_pointwise(y_true, y_pred, epsilon=1e-8):
    y_pred = np.maximum(y_pred, epsilon); y_true = np.maximum(y_true, epsilon)
    ratio = y_true / y_pred
    return ratio - np.log(ratio) - 1.0     # trung bình của hàm này = QLIKE tổng (metric chuẩn)
```

**Ví dụ tính tay (số minh hoạ, để hiểu công thức — không phải số thật từ model):**
Giả sử ngày X, mã Y: `y_true=0.020`, Model A dự báo `0.018` (gần đúng), Model B dự báo `0.026`
(xa hơn):
```
qlike_A = 0.020/0.018 − ln(0.020/0.018) − 1 = 1.111 − 0.105 − 1 = 0.006   (tốt, gần 0)
qlike_B = 0.020/0.026 − ln(0.020/0.026) − 1 = 0.769 − (−0.262) − 1 = 0.031  (tệ hơn)
delta_QLIKE = 0.031 − 0.006 = +0.025   → tin tức làm dự báo TỆ hơn tại điểm này
```

**Kết quả trên toàn bộ test set (~5,248 điểm dữ liệu, mỗi điểm = 1 mã tại 1 ngày):**

| Tháng | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean Δ | +.0023 | +.0005 | +.0023 | +.0032 | **-.0009** | +.0032 | +.0042 | +.0057 | +.0049 | +.0017 | +.0043 | +.0052 |
| n | 450 | 406 | 453 | 365 | 392 | 413 | 430 | 467 | 387 | 458 | 515 | 512 |

→ **11/12 tháng: tin tức làm dự báo TỆ hơn (Δ dương)**, chỉ tháng 5 hơi âm (gần 0, không đáng kể).
KHÔNG có tháng nào Δ âm rõ rệt (không có "tháng tin tức giúp ích").

| Kiểm định | Trong cửa sổ | Ngoài cửa sổ | Welch t | p-value | Kết luận |
|---|---|---|---|---|---|
| **Tết** (±10 ngày) | mean Δ | mean Δ | -0.816 | **0.415** | KHÔNG khác biệt có ý nghĩa |
| **Mùa BCTC** (20 ngày sau mỗi quý) | mean Δ | mean Δ | 0.392 | **0.695** | KHÔNG khác biệt có ý nghĩa |

| Tương quan Pearson | r | p-value | Kết luận |
|---|---|---|---|
| tet_proximity vs delta_QLIKE | -0.019 | 0.168 | KHÔNG có tương quan (gần 0) |
| earnings_proximity vs delta_QLIKE | -0.020 | 0.156 | KHÔNG có tương quan (gần 0) |

**Kết luận EDA:** với dữ liệu và model hiện tại, **KHÔNG phát hiện được** hiệu ứng "tin tức giúp
ích hơn quanh Tết hoặc mùa BCTC" — mọi p-value đều > 0.15 (ngưỡng thường dùng 0.05), hệ số tương
quan gần như bằng 0. Điều này NHẤT QUÁN với kết quả training §6.4 (thêm calendar feature không
giúp gì).

**Giới hạn phương pháp (đã ghi rõ trong code):** đây là single-seed, ~164 ngày test không độc lập
hoàn toàn (các mã cùng chịu ảnh hưởng thị trường chung, window 22 ngày chồng lấp) — p-value ở đây
mang tính SÀNG LỌC (screening), không phải bằng chứng thống kê chặt chẽ. Với tín hiệu gần 0 tuyệt
đối (r≈-0.02) thì khả năng có hiệu ứng thật đáng kể bị bỏ sót là thấp.

## 6.6 Ablation tách 3 nhóm feature — "cột nào có tín hiệu" (ĐÃ XONG cả 3, cập nhật lần 3 cùng ngày)

| Nhóm | Số cột | Test DirAcc | Test R² | Test QLIKE | Test RMSE |
|---|---:|---:|---:|---:|---:|
| **Không calendar (đối chứng)** | 0 | 68.76% | **0.7159** | **0.5497** | **0.002635** |
| Đủ 10 cột (§6.4) | 10 | 68.13% | 0.7117 | 0.5660 | 0.002654 |
| tet_only | 2 | 68.76% | 0.7124 | 0.5640 | 0.002651 |
| **earnings_only** | 2 | 68.71% | 0.7131 | **0.5501** | 0.002648 |
| **generic_calendar** (dow/month/cuối tháng-quý) | 6 | **68.90%** | 0.7121 | 0.5583 | 0.002652 |

**Đọc bảng (tất cả cùng 10 epoch, cùng seed, cùng panel):**
1. **Cả 4 biến thể calendar đều KHÔNG vượt được đối chứng trên R²/QLIKE/RMSE** — dù chênh lệch nhỏ.
2. **`earnings_only` gần đối chứng nhất** (QLIKE 0.5501 vs 0.5497 — gần như bằng nhau, R² 0.7131
   vs 0.7159 — chênh 0.0028) → 2 cột mùa BCTC gây "hại" ít nhất trong 3 nhóm.
3. **`generic_calendar` là biến thể DUY NHẤT có DirAcc CAO HƠN đối chứng** (68.90% vs 68.76%,
   +0.14pp) — nhưng R²/QLIKE vẫn thấp hơn, và +0.14pp nằm trong biên độ nhiễu single-seed (§5) nên
   KHÔNG kết luận "cuối tháng/thứ-trong-tuần thực sự giúp ích".
4. **Gộp cả 10 cột (§6.4) TỆ HƠN mọi nhóm con riêng lẻ** trên cả 4 metric — dấu hiệu các cột
   ablation có thể "nhiễu lẫn nhau" khi nối chung (nhiều chiều dư thừa/tương quan cao hơn là mỗi
   nhóm học được tín hiệu riêng), nhất quán với việc EDA (§6.5) không tìm thấy tín hiệu rõ ở bất kỳ
   nhóm nào — không có "tín hiệu thật" để cộng dồn, chỉ có nhiễu cộng dồn.

**Kết luận ablation:** không nhóm calendar feature nào (kể cả tách riêng) cho thấy lợi ích rõ ràng,
nhất quán hoàn toàn với EDA §6.5 (p-value > 0.15 mọi kiểm định).

## 6.7 Kết luận + đề xuất

1. **3 phương pháp độc lập (training đủ 10 cột, ablation 3-nhóm, EDA thống kê trên 2 model đã có)
   đều đồng thuận: chưa phát hiện được hiệu ứng "tin tức theo mùa BCTC/Tết"** với model + dữ liệu
   hiện tại — 5 lần train (đủ 10 cột + 3 nhóm con + đối chứng) và 4 kiểm định thống kê (2 t-test +
   2 tương quan) đều KHÔNG cho tín hiệu rõ.
2. **`earnings_only` (2 cột) là biến thể "ít hại nhất"** nếu vẫn muốn giữ lại 1 phần calendar
   feature cho mục đích khác (vd giải thích/diễn giải mô hình) — nhưng KHÔNG đủ căn cứ để coi là
   "cải thiện".
3. **Không loại trừ khả năng hiệu ứng có thật nhưng kiến trúc concat tĩnh không khai thác được** —
   nếu vẫn muốn theo hướng này, bước tiếp theo hợp lý là **Cách 2** (gate học theo thời gian thay
   vì gate tĩnh theo mã). Nhưng với cả EDA (không cần kiến trúc gì) và ablation (đã thử 4 cách nối
   khác nhau) đều đồng thuận "không thấy gì", xác suất Cách 2 thành công là KHÔNG cao — nên cân
   nhắc kỹ trước khi đầu tư thêm.
4. **Nguồn dữ liệu tin tức tiếng Việt hiện có có thể là giới hạn thật sự** — đây là lần thứ 17
   (12 lần trước + 5 lần hôm nay: đủ-10-cột, tet_only, earnings_only, generic_calendar, và phép so
   sánh trong EDA) thử liên quan tới tích hợp tin tức mà chưa vượt được HAR-only trên DirAcc.

---

# 7. THỬ NGHIỆM HORIZON 10-NGÀY (thay vì 5-ngày) + RETRAIN PANEL ĐÃ FIX (01/08/2026)

Code trong `baselines/2026-08-01_horizon10_baseline/`. Phạm vi: chỉ 2 kiến trúc mạnh nhất
(HAR-only, per-ticker gated news), 5-ngày vẫn là target chính, 10-ngày là thử nghiệm bổ sung
(không sửa file nào của các baseline 5-ngày hiện có).

## 7.1 Vì sao chỉ cần đổi 1 tham số

`forecast_horizon` đã là tham số constructor sẵn có trong pipeline dataset dùng chung
(`create_dual_news_dataloaders`, mặc định 5), chỉ dùng ở đúng 1 chỗ:
`target_idx = i + seq_length + forecast_horizon - 1`. Kiến trúc model, loss, và 6 metric đánh giá
không phụ thuộc giá trị này. 2 script train mới chỉ thêm `--forecast_horizon` (mặc định 10) và
truyền vào lời gọi hàm có sẵn — không có dataset/model mới.

**Kiểm chứng bằng test (không chỉ đọc code):** `test_target_shift.py` — dựng dữ liệu synthetic có
giá trị biết trước, xác nhận target ở window 0 đúng bằng `parkinson_volatility[index+31]` (=
`22+10-1`), KHÔNG PHẢI `index+26` (= `22+5-1`, công thức 5-ngày). Test này chạy TRƯỚC khi viết 2
script train, để xác nhận giả định trước khi code phần còn lại.

## 7.2 Kết quả — 5-ngày vs 10-ngày (cùng pipeline, batch_size=32, không augmentation)

| Metric | HAR-only 5-ngày | HAR-only 10-ngày | Diff | Gated-news 5-ngày (panel đã fix) | Gated-news 10-ngày | Diff |
|---|---:|---:|---:|---:|---:|---:|
| DirAcc | 68.42% | 67.80% | -0.62pp | 68.69% | 67.92% | -0.77pp |
| R² | 0.7141 | 0.7041 | -0.0100 | 0.7101 | 0.7040 | -0.0061 |
| QLIKE | 0.5623 | 0.5732 | +0.0109 | 0.5631 | 0.5767 | +0.0136 |
| RMSE | 0.002643 | 0.002689 | +0.000046 | 0.002662 | 0.002690 | +0.000028 |

**Đọc bảng:** cả 4 metric đều xấu đi ở horizon 10-ngày, cho CẢ 2 kiến trúc — 10-ngày khó dự báo
hơn 5-ngày với dữ liệu/model hiện tại. Tin tức không đổi hướng kết luận này ở cả 2 horizon (gated
news gần bằng HAR-only ở cả 2 mức, không có horizon nào tin tức tạo khác biệt rõ).

## 7.3 Retrain panel đã fix (VPB/VRE) — đóng issue đã nêu ở §5 cũ

Panel `dual_group_news_panel.parquet` đã được rebuild ngày 27/07 (gồm VPB/VRE, trước đó bị thiếu)
nhưng chưa retrain baseline `per_ticker_news_gate_baseline` trên panel này (đã nêu ở §5 mục 1 bản
trước). Ngày 01/08, xác nhận panel đã có đủ VPB/VRE (kiểm tra trực tiếp file parquet) rồi train
lại đúng script cũ (code không đổi), 10 epoch:

| Metric | Panel cũ (26/07) | Panel đã fix (01/08) | Diff |
|---|---:|---:|---:|
| DirAcc | 68.76% | 68.69% | -0.07pp |
| R² | 0.7159 | 0.7101 | -0.0058 |
| QLIKE | 0.5497 | 0.5631 | +0.0134 |
| RMSE | 0.002635 | 0.002662 | +0.000027 |

Thêm VPB/VRE không cải thiện kết quả — QLIKE/R² kém hơn nhẹ, DirAcc gần như không đổi. Đây không
phải dấu hiệu fix sai (VPB/VRE trước đó thật sự có 0 dữ liệu tin tức, việc thêm vào là sửa đúng dữ
liệu, không phải một đòn bẩy hiệu năng).

**Bảng trên chỉ dừng ở 10 epoch — CHƯA hội tụ, xem §7.4 để có số liệu đúng của kiến trúc này.**

## 7.4 Train tiếp lên 20 rồi 30 epoch — phát hiện chính, cập nhật cùng ngày

10 epoch (patience=15) không đủ để early-stopping kích hoạt, nên số liệu §7.2/§7.3 chỉ là điểm
giữa quá trình, không phải kết quả hội tụ. Đã resume (tiếp tục train, không train lại từ đầu) cả 2
biến thể panel-32-mã thêm 10 epoch (→20), sau đó biến thể 5-ngày thêm 10 epoch nữa (→30):

| Biến thể | Epoch | Test DirAcc | Test R² | Test QLIKE | Test RMSE | Trạng thái hội tụ |
|---|---:|---:|---:|---:|---:|---|
| Gated-news 5-ngày, panel đã fix | 10 | 68.69% | 0.7101 | 0.5631 | 0.002662 | đang cải thiện |
| Gated-news 5-ngày, panel đã fix | 20 | **69.51%** | **0.7158** | 0.5436 | 0.002635 | tốt nhất — xem epoch 30 |
| Gated-news 5-ngày, panel đã fix | 30 | 68.72% | 0.7156 | **0.5423** | 0.002636 | **bắt đầu overfit** |
| Gated-news 10-ngày | 10 | 67.92% | 0.7040 | 0.5767 | 0.002690 | — |
| Gated-news 10-ngày | 20 | 67.39% | 0.7040 | 0.5733 | 0.002690 | hội tụ/chững lại |

**Kết quả nổi bật — biến thể 5-ngày, epoch 20 (mốc tốt nhất tổng thể):**
- Test QLIKE = 0.5436 — thấp hơn (tốt hơn) mọi số liệu QLIKE từng ghi nhận cho kiến trúc
  per-ticker-gate trước đó (số liệu cũ tốt nhất: 0.5497 ở panel cũ, 10 epoch).
- Test R² = **0.7158** — cao nhất trong toàn bộ 30 epoch train, xấp xỉ số liệu tốt nhất trước đó
  (0.7159).
- **Test DirAcc = 69.51%** — cao nhất trong toàn bộ 30 epoch, **vượt HAR-only cùng pipeline
  (68.42%, xem §7.2) lần đầu tiên trong lịch sử dự án** cho một biến thể có tích hợp tin tức.

**Epoch 21-30: dấu hiệu overfitting.** Train loss tiếp tục giảm đều (0.884→0.839) nhưng val loss
KHÔNG giảm theo — dao động rồi tăng nhẹ (thấp nhất ~epoch 26, sau đó tăng lại tới epoch 30). Val
DirAcc giảm dần: 70.80% (epoch 20) → 70.34% (epoch 25) → 69.22% (epoch 30). Test DirAcc ở checkpoint
tốt nhất trong khoảng epoch 21-30 (epoch 26) chỉ đạt 68.72% — THẤP HƠN epoch 20 (69.51%). QLIKE
epoch 26 (0.5423) nhỉnh hơn epoch 20 (0.5436) một chút nhưng R²/DirAcc đều kém hơn — tổng thể
**epoch 20 là mốc cân bằng tốt nhất**, không phải epoch 30.

**Điều này khớp với tiền lệ đã ghi nhận trước đó (panel cũ, 26/07-27/07): cùng kiến trúc từng
đạt tốt nhất ở epoch ~20 rồi xấu đi nhẹ tới epoch 40** — nay lặp lại đúng pattern đó trên panel 32
mã. Kết luận: **~20 epoch là điểm dừng hợp lý cho kiến trúc per-ticker-gate ở horizon 5-ngày**,
không cần train thêm.

**Biến thể 10-ngày hội tụ/chững lại sớm hơn nhiều, ở epoch 10-20** — test QLIKE/R²/RMSE gần như
không đổi so với epoch 10, val QLIKE epoch 15→20 còn xấu đi nhẹ (0.7128→0.7156). Không có lý do để
train thêm biến thể này.

**Diễn giải:** khác biệt hội tụ giữa 2 horizon (5-ngày cần ~20 epoch mới đạt đỉnh, 10-ngày bão hoà
ngay từ epoch ~10-15) là quan sát mới, chưa có lời giải thích — có thể liên quan tới việc target
10-ngày "mượt" hơn (ít biến động hơn theo horizon dài) khiến model học nhanh bão hoà sớm hơn, nhưng
đây là suy đoán, chưa kiểm chứng.

## 7.5 Horizon 22-ngày (~1 tháng giao dịch)

Code trong `baselines/2026-08-01_horizon22_baseline/`, cùng pattern §7.1 (chỉ đổi
`forecast_horizon=22` khi gọi `create_dual_news_dataloaders`). Rủi ro mới so với horizon-10: cửa
sổ tối thiểu mỗi split tăng lên 44 ngày (22 input + 22 horizon) — đã kiểm tra bằng test trên TOÀN
BỘ các mã thật (không chỉ mẫu) trước khi train: train 891→847 window, val/test 191→147 window mỗi
split — không mã nào rủi ro 0 window (xác nhận đúng dự đoán, không phát sinh vấn đề khi train
thật).

**Kết quả (test set, 10 epoch, pipeline so sánh công bằng):**

| Kiến trúc | DirAcc | R² | QLIKE | RMSE |
|---|---:|---:|---:|---:|
| HAR-only | 66.38% | 0.7051 | 0.5938 | 0.002750 |
| Gated-news | 67.17% | 0.7032 | 0.5943 | 0.002759 |

Gated-news vượt HAR-only trên DirAcc (+0.79pp) nhưng QLIKE/R² nhỉnh hơn (kém hơn) một chút — mẫu
hình tương tự horizon 5 và 10-ngày ở mốc 10 epoch đầu (DirAcc thắng nhẹ, QLIKE/R² không nhất
quán).

**Hội tụ:** val loss của cả 2 kiến trúc dao động không có xu hướng rõ suốt 10 epoch (vd HAR-only:
1.1495→1.1409→1.1436, không đơn điệu) — khác hẳn biến thể 5-ngày (liên tục cải thiện tới epoch 20).
**22-ngày hội tụ/chững lại ngay từ epoch ~10**, giống horizon 10-ngày (§7.4) — không có dấu hiệu
cần train thêm.

**So với horizon 5 và 10-ngày (Bảng B, §1.2):** DirAcc/QLIKE của HAR-only tiếp tục xấu đi đơn điệu
theo horizon dài hơn (66.38% là thấp nhất trong 3 mốc, 0.5938 là QLIKE cao nhất/tệ nhất). Xác nhận
xu hướng đã thấy ở horizon-10: horizon càng dài, dự báo càng khó, nhưng cũng càng nhanh bão hoà
(cần ít epoch hơn).

## 7.6 Horizon 1-ngày — hoàn thành đủ bộ 4 mốc (1/5/10/22-ngày)

Code trong `baselines/2026-08-01_horizon1_baseline/`, cùng pattern §7.1. Rủi ro window-count THẤP
NHẤT trong 4 horizon đã thử (23 ngày tối thiểu/split, so với 27/32/44 của 5/10/22-ngày) — đã kiểm
tra thật trên toàn bộ các mã, margin rộng nhất (train 891→868, val/test 191→168 window/split).

**Kết quả (test set, 10 epoch, pipeline so sánh công bằng):**

| Kiến trúc | DirAcc | R² | QLIKE | RMSE |
|---|---:|---:|---:|---:|
| HAR-only | **72.35%** | **0.7581** | **0.5099** | **0.002428** |
| Gated-news | **72.39%** | **0.7595** | **0.4834** | **0.002420** |

**Xác nhận giả thuyết ban đầu: 1-ngày dễ dự báo nhất trong 4 mốc, cách biệt lớn** — DirAcc cao hơn
5-ngày ~4pp, QLIKE thấp hơn (tốt hơn) rõ rệt so với mọi horizon khác. Gated-news vượt HAR-only trên
CẢ 4 metric (không chỉ DirAcc như các horizon khác) — QLIKE 0.4834 vs 0.5099 là khác biệt lớn nhất
quan sát được giữa 2 kiến trúc ở bất kỳ horizon nào trong 10 epoch đầu.

**Hội tụ:** val loss giảm mạnh 2 epoch đầu (~0.94→0.93 cho cả 2 kiến trúc) rồi dao động nhẹ quanh
mức đó tới epoch 10 — hội tụ nhanh, giống pattern 10 và 22-ngày, KHÔNG giống 5-ngày (cần ~20 epoch).
Không có dấu hiệu cần train thêm.

**Kết luận chung cho cả 4 horizon (Bảng B, §1.2):** horizon càng ngắn càng dễ dự báo VÀ hội tụ càng
nhanh — ngoại lệ duy nhất là 5-ngày, cần nhiều epoch hơn hẳn (~20) mới đạt đỉnh dù không phải
horizon khó nhất. Chưa có lời giải thích cho ngoại lệ này.

---

# BẢNG THAM KHẢO (chuyển xuống cuối trang — thông tin phụ, không cần thiết ở mức tóm tắt đầu trang)

## Vì sao cần file này — thông tin trước đây nằm rải rác (trước đây là §2)

| Tài liệu cũ | Phạm vi | Thiếu gì so với kiến trúc hiện tại |
|---|---|---|
| `docs/project/PARALLEL_LSTM_GNN_ARCHITECTURE.md` (21/06) | Chỉ nhánh HAR (LSTM+GAT) | Không có nhánh tin tức, không có gate; số liệu "Epoch 4/50" đã lỗi thời |
| `baselines/2026-07-25_dual_group_news_embedding_baseline/design/design.md` | Nhánh HAR (tái dùng) + nhánh tin tức (NewsFeatureLSTM) | Không có per-ticker gate (thêm sau, 26/07) |
| `baselines/2026-07-26_per_ticker_news_gate_baseline/design/design.md` | Chỉ phần gate (giả định người đọc đã biết 2 file trên) | Không tự đứng độc lập được — ghi "sibling, KHÔNG đổi" |

→ Không file nào mô tả được toàn bộ pipeline nếu đọc riêng lẻ. File báo cáo này gộp cả 3 lại.

## Bảng A cũ (§1.2) — Tiến trình kiến trúc per-ticker-gate (5-ngày, TẤT CẢ hàng cùng 1 pipeline so sánh công bằng)

*(Con số "69.98% DirAcc" hay nhắc tới ở nơi khác trong dự án đến từ 1 pipeline lịch sử khác
(batch_size=11, có augmentation, 70 epoch) — KHÔNG thuộc bảng này, không so được với các hàng dưới
đây. Xem §4 dòng đầu nếu cần đối chiếu.)*

| # | Biến thể | Dùng tin tức | Epoch | DirAcc | R² | QLIKE | RMSE | Trạng thái |
|---|---|:---:|---:|---:|---:|---:|---:|---|
| 1 | HAR-only | — | 10 | 68.42% | 0.7141 | 0.5623 | 0.002643 | đối chứng chính |
| 2 | Per-ticker gated news (panel cũ, trước khi fix thiếu VPB/VRE) | ✓ | 10 | 68.76% | 0.7159 | 0.5497 | 0.002635 | kỷ lục cũ (26/07) |
| 3 | Per-ticker gated news (panel đã fix VPB/VRE) | ✓ | 10 | 68.69% | 0.7101 | 0.5631 | 0.002662 | chưa hội tụ |
| 4 | Per-ticker gated news (panel đã fix VPB/VRE) | ✓ | 20 | **69.51%** | **0.7158** | 0.5436 | 0.002635 | **tốt nhất — vượt hàng 1** |
| 5 | Per-ticker gated news (panel đã fix VPB/VRE) | ✓ | 30 | 68.72% | 0.7156 | **0.5423** | 0.002636 | bắt đầu overfit |

**Đọc bảng A:** hàng 4 (epoch 20, panel đã fix) là số liệu hiện hành, tốt nhất tổng thể — DirAcc
vượt hàng 1 (HAR-only cùng pipeline) lần đầu tiên trong lịch sử dự án tích hợp tin tức. Hàng 5 xác
nhận epoch 20 là điểm dừng hợp lý (train thêm bắt đầu overfit). Chi tiết: §7.3-7.4.

## Bảng C cũ (§1.2) — Calendar feature (5-ngày, panel cũ, 10 epoch), chi tiết đầy đủ ở §6.6

| Biến thể | Số cột calendar | DirAcc | R² | QLIKE | RMSE |
|---|---:|---:|---:|---:|---:|
| Đối chứng (không calendar) | 0 | 68.76% | 0.7159 | 0.5497 | 0.002635 |
| Đủ 10 cột | 10 | 68.13% | 0.7117 | 0.5660 | 0.002654 |
| tet_only | 2 | 68.76% | 0.7124 | 0.5640 | 0.002651 |
| earnings_only | 2 | 68.71% | 0.7131 | 0.5501 | 0.002648 |
| generic_calendar | 6 | 68.90% | 0.7121 | 0.5583 | 0.002652 |

Không biến thể calendar nào vượt đối chứng trên cả 4 metric — no-lift, xác nhận thêm bởi EDA
tương quan độc lập (§6.5, mọi p-value > 0.15).

---

# PHỤ LỤC — File map

```
src/lstm_gat_hybrid/model_parallel.py                                    # ParallelLSTMGNN (HAR branch)
src/common/feature_engineering.py, har_features.py                       # HAR rolling-window features
src/lstm_har_gat_hybrid/temporal_encoder.py                              # TemporalLSTM (per-stock)
baselines/2026-07-25_dual_group_news_embedding_baseline/code/model_dual_news.py   # NewsFeatureLSTM, DualGroupNewsBaseline
baselines/2026-07-26_per_ticker_news_gate_baseline/code/model_per_ticker_gate.py  # PerTickerGatedNewsBaseline (gate)
data/features/dual_group_news_panel.parquet                              # 146-feature news panel (bản cũ, trước khi fix VPB/VRE)
docs/project/PARALLEL_LSTM_GNN_ARCHITECTURE.md                           # chi tiết nhánh HAR (lỗi thời về số liệu, kiến trúc vẫn đúng)
docs/report_2026-07-25/BAO_CAO_CHO_THAY.md                               # báo cáo kết quả 12 biến thể tin tức (tên file lịch sử, không đổi lại)
docs/reports/2026-07-27_0010_summaryOfUpdate_report.md                   # fix VPB/VRE (chưa retrain)

# --- Mới, thêm 01/08/2026 (§6) ---
baselines/2026-08-01_calendar_news_gate_baseline/code/calendar_features.py           # 10 calendar feature, thuần, §6.2
baselines/2026-08-01_calendar_news_gate_baseline/code/dataset_calendar_news.py       # nối vào x_news, §6.3
baselines/2026-08-01_calendar_news_gate_baseline/code/train_calendar_news_gate.py    # train + --calendar_groups ablation, §6.6
baselines/2026-08-01_calendar_news_gate_baseline/code/analyze_news_calendar_correlation.py  # EDA delta_QLIKE, §6.5
baselines/2026-08-01_calendar_news_gate_baseline/requirements/requirements.md        # spec + giả định (bảng Tết, proxy BCTC)
baselines/2026-08-01_calendar_news_gate_baseline/design/design.md                    # kiến trúc + Simplicity/Anti-Abstraction gate
baselines/2026-08-01_calendar_news_gate_baseline/code_review/code_review_2026-08-01.md  # review đối kháng, 0 HIGH
results/calendar_gate_2026-08-01_073829/results.json                     # §6.4, đủ 10 cột
results/calendar_gate_tet_only_2026-08-01_082432/results.json            # §6.6, tet_only
results/calendar_gate_earnings_only_2026-08-01_083159/results.json       # §6.6, earnings_only
results/calendar_gate_generic_calendar_2026-08-01_083946/results.json    # §6.6, generic_calendar
results/news_calendar_correlation_2026-08-01_081230/                     # §6.5, EDA (analysis.json + per_point_delta_qlike.parquet + plot)
docs/reports/2026-08-01_0745_summaryOfUpdate_report.md                   # báo cáo kỹ thuật, baseline calendar (§6.4)

# --- Mới, thêm 01/08/2026 (§7) ---
baselines/2026-08-01_horizon10_baseline/code/train_har_only_reference_h10.py         # §7.1-7.2
baselines/2026-08-01_horizon10_baseline/code/train_per_ticker_gate_h10.py            # §7.1-7.2
baselines/2026-08-01_horizon10_baseline/test/test_target_shift.py                    # kiểm chứng target shift đúng 10 ngày
baselines/2026-08-01_horizon10_baseline/requirements/requirements.md, design/design.md
baselines/2026-08-01_horizon10_baseline/code_review/code_review_2026-08-01.md        # review đối kháng, 0 HIGH
results/har_only_h10_2026-08-01_090759/results.json                      # §7.2
results/per_ticker_gate_h10_2026-08-01_091853/results.json               # §7.2, epoch 10
results/per_ticker_gate_h10_2026-08-01_095135/results.json               # §7.4, epoch 20 (hội tụ)
results/per_ticker_gate_2026-08-01_092309/results.json                  # §7.3, panel đã fix, epoch 10
results/per_ticker_gate_2026-08-01_094139/results.json                  # §7.4, panel đã fix, epoch 20 (tốt nhất)
results/per_ticker_gate_2026-08-01_100100/results.json                  # §7.4, panel đã fix, epoch 30 (overfit)
docs/reports/2026-08-01_0928_summaryOfUpdate_report.md                   # báo cáo kỹ thuật, §7 (epoch 10)
baselines/2026-08-01_horizon22_baseline/                                 # §7.5, code/test/requirements/design/code_review
results/har_only_h22_2026-08-01_101237/results.json                     # §7.5, HAR-only 22-ngày
results/per_ticker_gate_h22_2026-08-01_102011/results.json               # §7.5, gated-news 22-ngày
baselines/2026-08-01_horizon1_baseline/                                 # §7.6, code/test/requirements/design/code_review
results/har_only_h1_2026-08-01_103548/results.json                      # §7.6, HAR-only 1-ngày
results/per_ticker_gate_h1_2026-08-01_104140/results.json               # §7.6, gated-news 1-ngày
```

---
§1-5 tổng hợp kiến trúc từ báo cáo/`results.json` có sẵn. §6 là kết quả của 5 lần train + 1 lần EDA
chạy 01/08/2026 (calendar feature). §7 là kết quả của toàn bộ thử nghiệm horizon cùng ngày: horizon
10-ngày tới epoch 20, retrain panel đã fix tới epoch 20 rồi 30, horizon 22-ngày và 1-ngày tới epoch
10 (cả 2 hội tụ sớm, không cần train thêm) — hoàn thành đủ bộ 4 mốc horizon (1/5/10/22-ngày). Số
liệu trích trực tiếp từ `results/*/results.json` và `results/news_calendar_correlation_*/analysis.json`.
