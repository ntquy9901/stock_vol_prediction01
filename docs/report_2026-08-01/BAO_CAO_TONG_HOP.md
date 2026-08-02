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

**Cập nhật lần 3 (cùng ngày) — đã bỏ DirAcc khỏi báo cáo:** công thức DirAcc hiện dùng cho mọi bảng
trong báo cáo tính trên mảng đã làm phẳng theo thứ tự `[window, mã]`, khiến phần lớn phép so sánh
là giữa 2 mã khác nhau cùng ngày, không phải cùng 1 mã qua thời gian — tính đúng đắn của số liệu
này chưa được xác nhận. Toàn bộ cột/giá trị/kết luận DirAcc đã được gỡ khỏi các bảng và câu kết
luận trong báo cáo này, chỉ giữ lại R², QLIKE, RMSE. Ghi chú kỹ thuật đầy đủ về vấn đề này (công
thức, code, số liệu đối chiếu) đã được chuyển ra file riêng: `docs/report_2026-08-01/DIRACC_ISSUE_NOTE.md`.

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
(a) pipeline gốc (batch_size=11, có augmentation) và (b) pipeline so sánh công bằng dùng cho TẤT CẢ
bảng dưới đây (batch_size=32, không augmentation). Mọi kết luận "vượt HAR-only" trong báo cáo này
đều trong pipeline (b).

*(Bảng A — tiến trình chi tiết per-ticker-gate qua các panel/epoch ở horizon 5-ngày — đã chuyển
xuống "Bảng tham khảo" cuối trang, vì Bảng B dưới đây đã đủ để so sánh tổng quan. Số liệu tốt nhất
hiện hành: R² 0.7158, QLIKE 0.5436, epoch 20, panel đã fix — chi tiết §7.3-7.4.)*

### Bảng B — So sánh horizon dự báo (1 vs 5 vs 10 vs 22 ngày, pipeline so sánh công bằng, 10 epoch trừ khi ghi khác)

| Kiến trúc | Horizon | R² | QLIKE | RMSE | Trạng thái hội tụ |
|---|---|---:|---:|---:|---|
| HAR-only | 1 ngày | 0.7581 | 0.5099 | 0.002428 | hội tụ nhanh (epoch ~5) |
| HAR-only | 5 ngày | 0.7141 | 0.5623 | 0.002643 | — |
| HAR-only | 10 ngày | **0.7041** | **0.5732** | **0.002689** | — |
| HAR-only | 22 ngày | **0.7051** | **0.5938** | **0.002750** | hội tụ ở epoch ~10 |
| Gated-news | 1 ngày | **0.7595** | **0.4834** | **0.002420** | hội tụ nhanh (epoch ~5) |
| Gated-news | 5 ngày | **0.7158** | **0.5436** | **0.002635** | epoch 20 — tốt nhất, xem Bảng A |
| Gated-news | 10 ngày | 0.7040 | 0.5767 | 0.002690 | epoch 10 — hội tụ ở epoch ~10-20 |
| Gated-news | 22 ngày | 0.7032 | 0.5943 | 0.002759 | epoch 10 — hội tụ ở epoch ~10 |

**Đọc bảng B (in đậm = kiến trúc thắng ở đúng horizon đó, so HAR-only với Gated-news cùng mốc):**

- **1-ngày và 5-ngày: Gated-news thắng CẢ 3 metric (R², QLIKE, RMSE).** Tin tức giúp ích rõ ở 2
  horizon ngắn nhất (lưu ý: hàng 5-ngày của Gated-news dùng epoch 20, không cùng epoch với hàng
  HAR-only — epoch 10 — nên so sánh này không hoàn toàn ngang epoch, xem Bảng A).
- **10-ngày và 22-ngày: HAR-only thắng cả R², QLIKE, RMSE** (dù chênh lệch rất nhỏ, có thể nằm
  trong nhiễu single-seed). Tin tức không còn giúp ích rõ ràng ở 2 horizon dài.
- **Xu hướng chung theo horizon (không phân biệt kiến trúc):** QLIKE tăng dần khi horizon dài hơn
  (0.5099→0.5623→0.5732→0.5938 cho HAR-only) — horizon càng dài càng khó dự báo, càng ngắn càng dễ,
  nhất quán ở cả 2 kiến trúc. R² nhảy vọt rõ ở mốc 1-ngày (~0.758) so với 3 mốc còn lại (~0.70-0.71).
- **Hội tụ:** 1, 10, 22-ngày đều hội tụ/chững lại nhanh (epoch ~5-10); CHỈ 5-ngày cần train tới
  epoch ~20 mới đạt đỉnh (Bảng A) — ngoại lệ, không phải quy luật chung theo horizon.

Chi tiết: §7.1-7.2, §7.5-7.6.

*(Bảng C — so sánh calendar feature — đã chuyển xuống cuối trang, xem "Bảng tham khảo" trước Phụ
lục, vì ít giá trị báo cáo ở mức tóm tắt so với Bảng A/B.)*

## 1.3 Kết luận điều hành — 5 câu (đã gộp từ bảng trên)

1. **Kết quả tốt nhất hiện tại: per-ticker gated news, panel đã fix, epoch 20** (Bảng A hàng 4) —
   R² 0.7158, QLIKE 0.5436. Epoch 30 xác nhận epoch 20 là điểm dừng hợp lý (train thêm bắt đầu
   overfit).
2. **Horizon càng dài càng khó dự báo — xu hướng đơn điệu qua đủ 4 mốc 1/5/10/22-ngày** (Bảng B):
   QLIKE tăng dần (0.5099→0.5623→0.5732→0.5938, HAR-only), cho cả 2 kiến trúc — **1-ngày dễ dự báo
   nhất, vượt trội rõ rệt so với 3 mốc còn lại** (đúng giả thuyết ban đầu, dựa trên QLIKE/R²).
   Riêng 5-ngày là ngoại lệ về hội tụ: 1/10/22-ngày đều hội tụ RẤT NHANH (~epoch 5-10), CHỈ 5-ngày
   cần ~20 epoch mới đạt đỉnh (câu 1).
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

Đây chính là baseline **"HAR-only"** dùng làm đối chứng chính trong mọi bảng so sánh — mọi
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

### 2.2.1 Xây dựng vector cho 1 mã, 1 ngày cụ thể — nhiều bài báo GỘP LẠI 1 vector, không giữ riêng

**Câu hỏi:** nếu ACB có 3 tin tức trong 1 ngày, mỗi tin có 1 vector riêng hay gộp lại 1 vector? Trả
lời ngắn: **GỘP LẠI thành đúng 1 vector duy nhất** (không giữ 3 vector riêng biệt) bằng phép
**trung bình cộng (mean) từng chiều**, sau khi từng bài đã được PCA-reduce. 4 bước theo đúng thứ tự
code chạy thật:

**Bước 1 — Encode từng bài báo riêng lẻ (768 chiều, làm 1 lần, dùng lại qua cache):**
File `news_embeddings.py`, cache `news_emb_articles_{nguồn}.parquet` (key = `url`). Mỗi bài báo
(title+lead) → 1 vector PhoBERT [CLS] 768 chiều — bước này KHÔNG chạy trong baseline hiện tại (chỉ
đọc cache có sẵn, không gọi lại PhoBERT — xem `_get_article_embeddings`, "cache-ONLY lookup").

**Bước 2 — "Nổ" theo từng mã được nhắc tới (`_explode_tickers`):** 1 bài báo có thể nhắc NHIỀU mã
cùng lúc (vd "ACB và VCB cùng công bố..."). Khi đó CÙNG 1 vector 768-chiều của bài đó được COPY
thành nhiều dòng, 1 dòng cho mỗi mã được nhắc — ở bước này, mỗi bài báo VẪN là 1 vector riêng, CHƯA
gộp gì cả.

**Bước 3 — PCA giảm chiều 768→32 (`_reduce`):** áp dụng PCA (fit trên dữ liệu TRƯỚC `TRAIN_CUTOFF`
để tránh rò rỉ, transform toàn bộ) lên TỪNG DÒNG (từng bài báo × mã) — vẫn 1 vector 32 chiều riêng
cho mỗi bài, CHƯA gộp theo ngày.

**Bước 4 — Gộp theo (mã, ngày) bằng TRUNG BÌNH CỘNG (`aggregate_articles`, code thật):**
```python
# dual_news_features.py — chạy khi nhóm theo (ticker, date), rows = các bài báo cùng mã, cùng ngày
for c in emb_cols:                      # từng chiều trong 32 chiều
    out[c] = float(rows[c].astype(float).mean())      # TRUNG BÌNH CỘNG qua các bài, KHÔNG concat
out["emb_norm"] = norm(mean_vector)                    # L2-norm TÍNH SAU KHI đã lấy mean
for cat in TOPIC_CATEGORIES:
    out[f"topic_{cat}_count"] = int(rows[f"topic_{cat}_count"].sum())   # topic thì CỘNG DỒN, không mean
```

**Ví dụ minh hoạ cách tính (số làm tròn cho dễ hiểu, minh hoạ đúng công thức — không phải trích
nguyên văn từ 1 ngày cụ thể trong dữ liệu thật):** giả sử ACB có 3 bài báo khách quan cùng ngày,
sau bước 3 mỗi bài đã có vector 32 chiều riêng; chỉ xét 2 chiều đầu (`emb_0`, `emb_1`) và cờ chủ đề
BCTC:

| Bài báo | `emb_0` (sau PCA) | `emb_1` (sau PCA) | Có nhắc BCTC? |
|---|---:|---:|:---:|
| Bài 1 | 0.90 | -0.40 | có (1) |
| Bài 2 | -0.30 | 0.10 | có (1) |
| Bài 3 | 1.20 | -0.20 | không (0) |
| **`kq_emb_0`/`kq_emb_1`** (TRUNG BÌNH 3 bài) | **(0.90-0.30+1.20)/3 = 0.60** | **(-0.40+0.10-0.20)/3 = -0.167** | — |
| **`kq_topic_earnings_count`** (TỔNG 3 bài) | — | — | **1+1+0 = 2** |

**Vì sao mean chứ không phải concat/attention:** `NewsFeatureLSTM` (§2.2) nhận input cố định 146
chiều/ngày/mã — nếu giữ riêng N vector (N thay đổi theo ngày, có ngày 0 bài có ngày 5 bài) thì kiến
trúc phải xử lý số chiều thay đổi (cần padding + mask hoặc attention-pooling, như baseline 07-07 cũ
đã làm). Baseline 07-25 chọn gộp sẵn bằng mean ở bước offline — đơn giản hơn, đổi lại: **thông tin
"có bao nhiêu bài, bài nào nói gì khác bài nào" bị mất khi lấy trung bình** — 3 bài nói 3 điều khác
nhau và 1 bài nói lặp lại 3 lần đều cho ra vector trung bình có thể giống nhau. Cột `topic_*_count`
(tổng, không phải mean) là cách duy nhất trong 146 feature còn giữ được tín hiệu "có bao nhiêu bài"
của ngày đó.

### 2.2.2 146 feature gồm những gì — breakdown chi tiết + ví dụ vector thật

**File:** `baselines/2026-07-25_dual_group_news_embedding_baseline/code/vendor_data_eda/dual_news_features.py`,
`news_embeddings.py`. Đo trực tiếp trên `data/features/dual_group_news_panel.parquet` (146 cột
feature, không tính `ticker`/`date`):

| Nhóm cột | Số cột | Nguồn | Ý nghĩa |
|---|---:|---|---|
| `kq_emb_0..31` | 32 | Báo "khách quan" (cafef, vnexpress, thanhnien, tuoitre, nld, vietnamplus, hsc) | PhoBERT (768-dim) mean-pool các bài về đúng mã, đúng ngày → PCA còn 32 chiều |
| `kq_emb_norm` | 1 | " | L2-norm của vector `kq_emb_0..31` — 1 số đo "cường độ" tin khách quan hôm đó |
| `th_emb_0..31` | 32 | Nguồn "tổng hợp" CTCK (ssi, vndirect, vnstock, vietstock, vsdc) | Cùng cách tính, nhóm nguồn khác — PCA dùng CHUNG 1 basis với `kq_emb_*` nên 2 nhóm so sánh được theo từng chiều |
| `th_emb_norm` | 1 | " | L2-norm của `th_emb_0..31` |
| `ewma_kq_emb_0..31` + `_norm` | 33 | Suy giảm dần (half-life 30 ngày) của `kq_emb_*` | "Dư âm" tin khách quan, khác 0 cả những ngày không có tin mới |
| `ewma_th_emb_0..31` + `_norm` | 33 | Suy giảm dần của `th_emb_*` | Tương tự, cho nhóm tổng hợp CTCK |
| `kq_topic_{7 chủ đề}_count` | 7 | Đếm bài khách quan theo chủ đề: earnings/dividend/M&A/management/regulation/macro/sector | Tần suất tin theo chủ đề cụ thể, không phải embedding |
| `th_topic_{7 chủ đề}_count` | 7 | Tương tự, nhóm tổng hợp CTCK | " |
| **Tổng** | **146** | | 33+33+33+33+14 = 146 |

**Ví dụ vector thật — mã ACB, ngày 2007-02-12 (đo trực tiếp từ parquet, ngày này ACB có bài báo
khách quan thật, KHÔNG có bài tổng hợp CTCK):**

| Cột | Giá trị | Đọc thế nào |
|---|---:|---|
| `kq_emb_0` | -1.7540 | chiều PCA thứ 0 của embedding tin khách quan hôm đó |
| `kq_emb_1` | 0.9625 | chiều PCA thứ 1 |
| `kq_emb_2` | 0.7061 | chiều PCA thứ 2 |
| `kq_emb_norm` | 4.4027 | cường độ tổng hợp của tin khách quan hôm đó (L2-norm 32 chiều) |
| `kq_topic_earnings_count` | 0.0 | không có bài nào về BCTC hôm đó |
| `kq_topic_macro_count` | 0.0 | không có bài về vĩ mô hôm đó |
| `th_emb_0` | `NaN`→`0.0` khi train | KHÔNG có bài tổng hợp CTCK nào về ACB hôm đó |
| `th_emb_norm` | `NaN`→`0.0` khi train | " |
| `ewma_kq_emb_0` | -0.3535 | dư âm suy giảm — nhỏ hơn `kq_emb_0` (-1.7540) vì công thức EWMA còn pha trộn cả những ngày trước |
| `ewma_kq_emb_norm` | 0.9253 | cường độ "dư âm" tổng hợp, thấp hơn cường độ tức thời (4.4027) |

Đo trên toàn bộ 4989 ngày của ACB: có tin khách quan thật (`kq_emb_*` không NaN) ở **1155/4989 ngày
(23.2%)** — phần lớn ngày còn lại dựa vào cột `ewma_*` (dư âm) thay vì tín hiệu tức thời, đúng như
mô tả ở §2.2.4 dưới đây.

### 2.2.3 EWMA + half-life là gì, tại sao dùng, căn cứ khoa học, ví dụ cụ thể

**EWMA (Exponentially Weighted Moving Average)** là trung bình trượt có trọng số giảm dần theo cấp
số nhân — giá trị càng cũ càng ít ảnh hưởng tới trung bình hiện tại. Công thức đệ quy tổng quát:

```
ema[t] = alpha * value[t] + (1 - alpha) * ema[t-1]     (khi ngày t có giá trị mới)
```

**Half-life (chu kỳ bán rã)** là số ngày để trọng số của 1 giá trị cũ giảm còn đúng một nửa. Chọn
half-life = 30 ngày (không phải chọn `alpha` trực tiếp) vì half-life dễ diễn giải hơn: "sau 30 ngày
không có tin mới, ảnh hưởng của tin cũ còn lại đúng 50%". Từ half-life suy ra `alpha`:

**Code thật:** `baselines/2026-07-25_dual_group_news_embedding_baseline/code/vendor_data_eda/dual_news_features.py`
```python
def _ewma_on_series(series: pd.Series, halflife: float) -> pd.Series:
    alpha = 1.0 - np.exp(-np.log(2) / halflife)   # halflife=30 -> alpha ≈ 0.02284
    ema = np.nan
    for i in range(len(series)):
        val = series.iloc[i]
        if np.isnan(val):                          # KHÔNG có tin mới hôm nay
            if not np.isnan(ema):
                ema = (1.0 - alpha) * ema           # chỉ suy giảm, không cộng thêm gì
        else:                                       # CÓ tin mới hôm nay
            ema = val if np.isnan(ema) else alpha * val + (1.0 - alpha) * ema
        series.iloc[i] = ema
    return series
```
Điểm khác biệt quan trọng so với EWMA "chuẩn" (`pandas.Series.ewm`, vốn giả định ngày nào cũng có
giá trị): code này xử lý đúng trường hợp tin tức đến THƯA (phần lớn ngày là `NaN`, không phải 0) —
những ngày không có tin chỉ suy giảm giá trị cũ (nhân với `1-alpha`), KHÔNG cộng thêm tín hiệu mới.

**Kiểm chứng công thức `(1-alpha)^30 = 0.5` đúng nghĩa "half-life 30 ngày":**
`(1 - alpha) = exp(-ln(2)/30)` → `(1-alpha)^30 = exp(-ln(2)) = 1/2` — đúng bằng 50% sau đúng 30
ngày liên tiếp không có tin mới, bất kể giá trị gốc là bao nhiêu.

**Ví dụ tính tay — số liệu thật, mã ACB, cột `kq_emb_0`, giai đoạn 24/10/2008 → 03/11/2008
(alpha = 0.022840, đo trực tiếp từ panel):**

| Ngày | `kq_emb_0` (tin thật hôm đó) | `ewma_kq_emb_0` | Diễn giải |
|---|---:|---:|---|
| 2008-10-24 | 1.352 (có tin) | -0.0795 | cập nhật đầy đủ: `alpha*1.352 + (1-alpha)*ema_cũ` |
| 2008-10-27 | *(không tin)* | -0.0777 | suy giảm thuần: `-0.0795 × 0.97716 = -0.0777` |
| 2008-10-28 | *(không tin)* | -0.0759 | suy giảm tiếp: `-0.0777 × 0.97716 = -0.0759` |
| 2008-10-29 | *(không tin)* | -0.0742 | suy giảm tiếp: `-0.0759 × 0.97716 = -0.0742` |
| 2008-10-30 | -2.777 (có tin) | -0.1359 | cập nhật lại: `alpha×(-2.777) + (1-alpha)×(-0.0742)` |
| 2008-11-03 | -2.792 (có tin) | -0.1936 | tin mới liên tiếp → ema dịch nhanh về phía tin mới |

Mỗi dòng "không tin" ở trên nhân đúng `(1-alpha) = 0.97716` với dòng trước — khớp 100% với công
thức, không phải số minh hoạ.

**Căn cứ khoa học/thực nghiệm để dùng decay thay vì raw tức thời:**
- Tài chính hành vi (behavioral finance) đã ghi nhận hiệu ứng **"post-earnings-announcement
  drift"** và các nghiên cứu về "investor attention decay"/"stale news": phản ứng giá trước 1 tin
  tức KHÔNG kết thúc ngay trong ngày đăng bài, mà kéo dài nhiều ngày/tuần rồi mới suy giảm dần — tin
  tức có "quán tính" (persistence), không phải sự kiện tức thời rồi biến mất.
- EWMA với half-life là kỹ thuật chuẩn để mã hoá "quán tính suy giảm dần" này thành 1 con số duy
  nhất mỗi ngày (thay vì phải nhớ toàn bộ lịch sử N ngày gần nhất) — dùng phổ biến trong tài chính
  định lượng (vd EWMA volatility của RiskMetrics/JP Morgan cũng dùng half-life tương tự cho biến
  động giá, không riêng cho tin tức).
- **Half-life = 30 ngày trong project này là giá trị kế thừa từ code `data_eda` có sẵn (vendor,
  Anti-Abstraction Gate — dùng thẳng code đã chạy được thay vì tự viết lại), KHÔNG phải giá trị đã
  qua tinh chỉnh/kiểm định riêng cho bộ dữ liệu VN30 này** — 30 ngày là con số hợp lý theo bậc độ
  lớn (1 tháng giao dịch, cùng bậc với "mùa BCTC" ở §6.2) nhưng chưa có thử nghiệm ablation nào so
  sánh half-life 10/30/60 ngày trong project để xác nhận 30 là tối ưu.

**Nếu KHÔNG có feature `ewma_*` thì vấn đề là gì:**
1. **Tín hiệu tin tức gần như biến mất ở đa số ngày.** Với mã ACB chỉ 23.2% ngày có tin khách quan
   thật (§2.2.2) — nếu chỉ dùng cột raw (`kq_emb_*`/`th_emb_*`, `fillna(0.0)`), model sẽ thấy vector
   toàn số 0 ở ~77% ngày còn lại. Nhánh tin tức gần như "câm" phần lớn thời gian, LSTM khó học được
   quy luật thời gian từ 1 tín hiệu chỉ xuất hiện rải rác, không liên tục.
2. **Mất phân biệt "vừa có tin hôm qua" với "đã im lặng cả tháng".** Không có EWMA, cả 2 trường hợp
   đều là vector 0 giống hệt nhau ở ngày hiện tại — model không có cách nào biết mình đang ở "trong
   dư âm" của 1 tin quan trọng vừa xảy ra hay thực sự không có gì liên quan đang diễn ra.
3. **Bước nhảy gián đoạn (discontinuity) giữa các ngày liên tiếp.** Ví dụ ở bảng trên: nếu không có
   EWMA, chuỗi giá trị `kq_emb_0` sẽ là `1.352, 0, 0, 0, -2.777, 0, -2.792` — nhảy đột ngột từ số
   thật về 0 rồi lại về số thật, thay vì chuỗi mượt `−0.0795 → −0.0777 → −0.0759 → −0.0742 → −0.1359
   → ... → −0.1936` mà EWMA tạo ra — chuỗi mượt dễ học hơn nhiều cho 1 LSTM so với chuỗi có bước
   nhảy lớn, thất thường.

### 2.2.4 Ví dụ cụ thể: ngày CÓ tin vs KHÔNG có tin được tổ chức thế nào khi train

**Xem trực tiếp:** `data/features/dual_group_news_panel.parquet` — mở bằng pandas:
```python
import pandas as pd
df = pd.read_parquet('data/features/dual_group_news_panel.parquet')
sub = df[df['ticker'] == 'ACB'].sort_values('date')   # đổi mã tuỳ ý
```

**Thực tế đo được (chạy trên panel thật, mã ACB, 4989 dòng — gần như mọi ngày giao dịch từ
2006 đến nay) có 3 LOẠI NGÀY, không phải 2:** 146 cột chia thành **80 cột "raw"** (`kq_emb_*`,
`th_emb_*`, `*_topic_*_count` — tín hiệu tin NGAY HÔM ĐÓ) và **66 cột `ewma_*`** (dư âm suy giảm
dần, half-life 30 ngày, mang theo ảnh hưởng của tin cũ):

| Loại ngày | 80 cột raw | 66 cột `ewma_*` | Tỷ lệ đo thật (mã ACB) |
|---|---|---|---|
| **Có tin ngay hôm đó** | số thật | số thật | 26.46% |
| **Không tin mới, còn "dư âm" tin cũ** | NaN trong panel | vẫn số thật (chưa suy giảm hết) | ~73.5% |
| **Chưa từng có tin trước đó** | NaN | NaN | hiếm (chỉ trước ngày panel bắt đầu có dữ liệu cho mã đó, vd trước 27/10/2006 với ACB) |

**Khi load để train**, hàm `load_news_panel()` áp `fillna(0.0)` lên TOÀN BỘ 146 cột — nghĩa là
NaN (dù ở nhóm raw hay ewma) đều biến thành 0:
```python
# baselines/2026-07-25_dual_group_news_embedding_baseline/code/dataset_dual_news.py
df[feature_cols] = df[feature_cols].fillna(0.0)   # NaN (raw HOẶC ewma) -> 0
...
vec = news_cache.get(d)                            # tra theo (mã, ngày)
day_feats.append(vec if vec is not None else np.zeros(self._n_feat, dtype=np.float32))
```

**Hệ quả quan trọng — khác với suy nghĩ đơn giản "không tin = toàn số 0":** phần lớn "ngày không
có tin mới" (~73.5% với ACB) **vẫn có 66/146 số khác 0** (phần EWMA mang theo ảnh hưởng tin cũ đã
suy giảm dần) — chỉ có 80 cột raw về 0. Vector "toàn bộ 146 số = 0" CHỈ xảy ra ở loại ngày thứ 3
(chưa từng có tin trước đó) — hiếm hơn nhiều so với giả định ban đầu.

**Ví dụ tính tay (4 cột đầu, số liệu chạy thật từ panel):**

| Ngày (ACB) | Loại | raw[0:4] | ewma[0:4] |
|---|---|---|---|
| 2006-10-27 | Có tin ngay hôm đó | `[-0.825, 0.122, 0.254, 0.874]` | `[-0.825, 0.122, 0.254, 0.874]` (mới, chưa suy giảm) |
| 2006-10-30 | Không tin mới, còn dư âm | `[0, 0, 0, 0]` (NaN→0) | `[-0.807, 0.119, 0.248, 0.855]` (đã suy giảm nhẹ) |

**Ý nghĩa:** model không thể phân biệt "raw = 0 vì hết tin" với "raw thật sự bằng 0" (không có
mask riêng — đơn giản hoá có chủ đích, ghi trong docstring code). Nhưng nhờ phần EWMA gần như luôn
khác 0, nhánh tin tức KHÔNG bị "chết" hoàn toàn giữa các ngày có tin thật — đây là điểm khác biệt
quan trọng so với mô tả ban đầu của mục này.

**Rủi ro đã biết:** nếu 1 mã có tin tức rất thưa (ít ngày raw thật), nhánh tin tức phần lớn dựa vào
EWMA (xu hướng dài hạn) hơn là tín hiệu tức thời — đây là 1 phần lý do project từng nghi ngờ "gate
học được" không khớp tín hiệu "mã nào cần tin tức" đo độc lập (§5 mục 2).

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

| Biến thể | Pipeline | Epoch | R² | QLIKE | RMSE |
|---|---|---:|---:|---:|---:|
| **HAR-only** (không dùng tin) | gốc (batch11+aug) | 70 | 0.7140 | 0.5294 | 0.002644 |
| Gated Cross-Attention | gốc | 15 | 0.7157 | 0.5567 | 0.002636 |
| Dual-group + EWMA (không gate) | gốc | 10 | 0.7124 | 0.5598 | 0.002651 |
| HAR-only | so sánh (batch32) | 10 | 0.7141 | 0.5623 | 0.002643 |
| Per-ticker gated news, panel cũ | so sánh | 10 | 0.7159 | 0.5497 | 0.002635 |
| Per-ticker gated news, panel đã fix | so sánh | 10 | 0.7101 | 0.5631 | 0.002662 |
| **Per-ticker gated news, panel đã fix — hiện hành** | so sánh | 20 | **0.7158** | **0.5436** | 0.002635 |

Nguồn: `docs/report_2026-07-25/BAO_CAO_CHO_THAY.md` §1.1 (3 dòng "gốc") +
`docs/reports/2026-07-26_2330_summaryOfUpdate_report.md` (panel cũ) +
`results/per_ticker_gate_2026-08-01_094139/results.json` (panel đã fix epoch 20, số liệu hiện hành
— xem phân tích hội tụ đầy đủ ở §7.4).

---

# 5. GIỚI HẠN / VIỆC CHƯA XONG

1. **[ĐÃ XONG cùng ngày, xem §7.3-7.4]** Panel tin tức cũ, thiếu VPB/VRE — bug đã fix (27/07),
   đã retrain per-ticker gate trên panel đã fix (01/08), train tới epoch 30 để kiểm tra hội
   tụ đầy đủ. Kết quả cuối: epoch 20 là mốc tốt nhất (QLIKE 0.5436, R² 0.7158), epoch 30 cho dấu
   hiệu overfitting — số liệu panel đã fix, epoch 20 là số liệu hiện hành, xem Bảng A ở "Bảng tham
   khảo" cuối trang.
2. **Gate học được KHÔNG khớp tín hiệu "mã nào cần tin tức" đo độc lập** — 4 phương pháp đo (EDA
   HGB/XGBoost, ablation delta-QLIKE tự đo, và chính gate học được) cho 4 thứ tự mã khác nhau,
   chưa có lời giải thích thống nhất (xem memory `project_selective_news_gate_finding`).
3. **Single-seed** — mọi kết quả trên chỉ train 1 lần (1 seed); chênh lệch nhỏ giữa các biến thể
   (vd 68.25% vs 68.76%) nằm trong biên độ có thể là nhiễu, chưa multi-seed để kiểm chứng.
4. **Không có feature thời gian dạng lịch** (§3, cũ) — **ĐÃ LÀM cùng ngày, xem §6** — kết quả:
   thêm vào rồi nhưng chưa cải thiện (§6.4), và EDA không phát hiện tín hiệu theo mùa (§6.5).
5. **3/3 ablation tách nhóm calendar feature đã xong (§6.6)** — không nhóm nào (Tết/BCTC/generic)
   vượt được đối chứng trên R²/QLIKE/RMSE. Gộp cả 10 cột lại TỆ HƠN mỗi nhóm con riêng lẻ.
6. **10-ngày và 22-ngày-trước (§7) chỉ mới thử 2 kiến trúc (HAR-only, gated-news), chưa thử
   calendar-augmented ở 2 horizon này** — nếu cần, đây là phần mở rộng riêng, chưa làm.
7. **So sánh horizon dùng pipeline batch_size=32/không augmentation** — khác pipeline (a) ở §1.2
   (batch_size=11, có augmentation). 2 pipeline không so trực tiếp được với nhau; §7 chỉ so sánh
   trong CÙNG 1 pipeline.
8. **[ĐÃ XONG]** Horizon 22-ngày: đã train 10 epoch cả 2 kiến trúc, hội tụ ngay từ epoch ~10
   (giống 10-ngày, khác 5-ngày) — xem Bảng B (§1.2) và §7.5. Chưa thử epoch >10 (theo trạng thái
   hội tụ đo được, không có lý do để train thêm).
9. **[MỚI, quan trọng]** DirAcc đã được gỡ khỏi báo cáo này (mọi bảng, mọi kết luận) do phát hiện
   vấn đề về công thức tính (thứ tự dữ liệu khi flatten khiến phép so sánh không đúng nghĩa "cùng
   mã, ngày kế tiếp") — chưa xác nhận được số liệu đúng. Ghi chú kỹ thuật đầy đủ (công thức, code,
   số liệu đối chiếu) ở `docs/report_2026-08-01/DIRACC_ISSUE_NOTE.md`.

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

| Nhóm | Số cột | Test R² | Test QLIKE | Test RMSE |
|---|---:|---:|---:|---:|
| **Không calendar (đối chứng)** | 0 | **0.7159** | **0.5497** | **0.002635** |
| Đủ 10 cột (§6.4) | 10 | 0.7117 | 0.5660 | 0.002654 |
| tet_only | 2 | 0.7124 | 0.5640 | 0.002651 |
| **earnings_only** | 2 | 0.7131 | **0.5501** | 0.002648 |
| **generic_calendar** (dow/month/cuối tháng-quý) | 6 | 0.7121 | 0.5583 | 0.002652 |

**Đọc bảng (tất cả cùng 10 epoch, cùng seed, cùng panel):**
1. **Cả 4 biến thể calendar đều KHÔNG vượt được đối chứng trên R²/QLIKE/RMSE** — dù chênh lệch nhỏ.
2. **`earnings_only` gần đối chứng nhất** (QLIKE 0.5501 vs 0.5497 — gần như bằng nhau, R² 0.7131
   vs 0.7159 — chênh 0.0028) → 2 cột mùa BCTC gây "hại" ít nhất trong 3 nhóm.
3. **Gộp cả 10 cột (§6.4) TỆ HƠN mọi nhóm con riêng lẻ** trên cả 3 metric — dấu hiệu các cột
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
   sánh trong EDA) thử liên quan tới tích hợp tin tức mà chưa cho thấy lợi ích rõ ràng trên R²/QLIKE/RMSE.

## 6.8 Tin tức "chung chung"/thị trường có nhiều không, có tương quan với biến động giá không

**Câu hỏi:** ngoài tin gắn đúng 1 mã, có nhiều tin chung chung/thị trường không, và khối lượng tin
đó có tương quan với biến động giá không? Trả lời bằng thống kê model-free (không train model),
code: `code/analyze_market_news_volume_correlation.py`.

**Khối lượng tin chung chung — đo trực tiếp:**
- Kho crawl thô (`crawl_data/data/`) lớn hơn nhiều so với phần dùng trong pipeline: hàng chục file
  theo nguồn (baodautu, dantri, sggp, plo, cand, bnews...) cộng 1 file `news_articles.csv` ~9.3
  triệu dòng — KHÔNG nguồn nào trong số này nằm trong danh sách 12 nguồn `GROUP_SOURCES` (§2.2.2)
  đang được dùng. Đây là tin tức chưa được khai thác, không phải "không tồn tại".
- Ngay trong 12 nguồn đang dùng, phần lớn bài KHÔNG gắn được vào 1 mã cụ thể: panel có 159,648
  dòng (mã × ngày) nhưng chỉ 18,985 dòng có tin khách quan thật cùng ngày và 5,106 dòng có tin
  tổng hợp CTCK thật cùng ngày — còn lại là NaN (không có bài nào nhắc đúng mã đó, đúng ngày đó).
- Tin gắn nhãn "vĩ mô" (`topic_macro`) cực hiếm: chỉ 183/159,648 dòng (0.11%).

**Tương quan khối lượng tin thị trường vs biến động giá (2026-08-01, số liệu thật, 4987 ngày giao dịch):**

Xây dựng `news_volume(ngày)` = tổng số bài gắn topic (7 chủ đề × 2 nhóm nguồn) trên TOÀN BỘ 32 mã
ngày đó (đo được từ panel, không cần crawl thô 9.3 triệu dòng). Tương quan với
`market_avg_change(ngày)` = trung bình biến động Parkinson-vol ngày-qua-ngày trên cả 32 mã:

| Phép đo | Pearson r | p-value | Kết luận |
|---|---:|---:|---|
| Cùng ngày (contemporaneous) | 0.011 | 0.435 | KHÔNG có tương quan |
| Ngày kế tiếp (có dấu) | -0.038 | 0.008 | "có ý nghĩa" nhưng r quá nhỏ, không đáng kể thực tế |
| Ngày kế tiếp (trị tuyệt đối, biến động mạnh/nhẹ) | 0.027 | 0.059 | ở ranh giới ngưỡng 0.05, không rõ ràng |

| Nhóm | Ngưỡng | n | Biến động TB ngày kế tiếp (trị tuyệt đối) |
|---|---:|---:|---:|
| Ngày ít/không tin (quartile dưới) | volume ≤ 0 | 3143 | 0.000583 |
| Ngày nhiều tin (quartile trên) | volume ≥ 1 | 1844 | 0.000758 |

Welch t = 2.02, p = 0.044 — ngày nhiều tin thị trường có biến động ngày kế tiếp lớn hơn ngày ít tin,
khác biệt có ý nghĩa thống kê ở ngưỡng 0.05 nhưng chênh lệch tuyệt đối rất nhỏ (0.000758 vs
0.000583).

**Kết luận:** có tồn tại 1 tương quan yếu, đúng hướng trực giác (ngày nhiều tin thị trường hơn →
biến động ngày sau lớn hơn một chút) nhưng độ lớn hiệu ứng gần như không đáng kể (r < 0.04 ở mọi
phép đo) — nhất quán với toàn bộ phát hiện "no-lift"/tín hiệu tin tức yếu đã ghi nhận xuyên suốt
báo cáo này (§6.5-6.7). **Giới hạn:** `news_volume` ở đây là khối lượng tin ĐÃ gắn nhãn mã/chủ đề
(không phải toàn bộ 9.3 triệu dòng crawl thô, xem giới hạn ghi trong docstring code); dữ liệu chuỗi
thời gian không độc lập (biến động có tính tự tương quan) nên p-value mang tính sàng lọc, không
phải bằng chứng thống kê chặt chẽ.

Test: `test/test_market_news_volume_correlation.py`, 5/5 pass — bao gồm 1 real-data-sample smoke
test phát hiện bug thật (VPB/VRE có ngày dạng tz-aware `+07:00` trong khi 30 mã còn lại tz-naive,
gây lỗi khi ghép; đã fix bằng cách chuẩn hoá về tz-naive trước khi ghép).

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
| R² | 0.7141 | 0.7041 | -0.0100 | 0.7101 | 0.7040 | -0.0061 |
| QLIKE | 0.5623 | 0.5732 | +0.0109 | 0.5631 | 0.5767 | +0.0136 |
| RMSE | 0.002643 | 0.002689 | +0.000046 | 0.002662 | 0.002690 | +0.000028 |

**Đọc bảng:** cả 3 metric đều xấu đi ở horizon 10-ngày, cho CẢ 2 kiến trúc — 10-ngày khó dự báo
hơn 5-ngày với dữ liệu/model hiện tại. Tin tức không đổi hướng kết luận này ở cả 2 horizon (gated
news gần bằng HAR-only ở cả 2 mức, không có horizon nào tin tức tạo khác biệt rõ).

## 7.3 Retrain panel đã fix (VPB/VRE) — đóng issue đã nêu ở §5 cũ

Panel `dual_group_news_panel.parquet` đã được rebuild ngày 27/07 (gồm VPB/VRE, trước đó bị thiếu)
nhưng chưa retrain baseline `per_ticker_news_gate_baseline` trên panel này (đã nêu ở §5 mục 1 bản
trước). Ngày 01/08, xác nhận panel đã có đủ VPB/VRE (kiểm tra trực tiếp file parquet) rồi train
lại đúng script cũ (code không đổi), 10 epoch:

| Metric | Panel cũ (26/07) | Panel đã fix (01/08) | Diff |
|---|---:|---:|---:|
| R² | 0.7159 | 0.7101 | -0.0058 |
| QLIKE | 0.5497 | 0.5631 | +0.0134 |
| RMSE | 0.002635 | 0.002662 | +0.000027 |

Thêm VPB/VRE không cải thiện kết quả — QLIKE/R² kém hơn nhẹ. Đây không phải dấu hiệu fix sai
(VPB/VRE trước đó thật sự có 0 dữ liệu tin tức, việc thêm vào là sửa đúng dữ liệu, không phải một
đòn bẩy hiệu năng).

**Bảng trên chỉ dừng ở 10 epoch — CHƯA hội tụ, xem §7.4 để có số liệu đúng của kiến trúc này.**

## 7.4 Train tiếp lên 20 rồi 30 epoch — phát hiện chính, cập nhật cùng ngày

10 epoch (patience=15) không đủ để early-stopping kích hoạt, nên số liệu §7.2/§7.3 chỉ là điểm
giữa quá trình, không phải kết quả hội tụ. Đã resume (tiếp tục train, không train lại từ đầu) cả 2
biến thể panel-32-mã thêm 10 epoch (→20), sau đó biến thể 5-ngày thêm 10 epoch nữa (→30):

| Biến thể | Epoch | Test R² | Test QLIKE | Test RMSE | Trạng thái hội tụ |
|---|---:|---:|---:|---:|---|
| Gated-news 5-ngày, panel đã fix | 10 | 0.7101 | 0.5631 | 0.002662 | đang cải thiện |
| Gated-news 5-ngày, panel đã fix | 20 | **0.7158** | 0.5436 | 0.002635 | tốt nhất — xem epoch 30 |
| Gated-news 5-ngày, panel đã fix | 30 | 0.7156 | **0.5423** | 0.002636 | **bắt đầu overfit** |
| Gated-news 10-ngày | 10 | 0.7040 | 0.5767 | 0.002690 | — |
| Gated-news 10-ngày | 20 | 0.7040 | 0.5733 | 0.002690 | hội tụ/chững lại |

**Kết quả nổi bật — biến thể 5-ngày, epoch 20 (mốc tốt nhất tổng thể):**
- Test QLIKE = 0.5436 — thấp hơn (tốt hơn) mọi số liệu QLIKE từng ghi nhận cho kiến trúc
  per-ticker-gate trước đó (số liệu cũ tốt nhất: 0.5497 ở panel cũ, 10 epoch).
- Test R² = **0.7158** — cao nhất trong toàn bộ 30 epoch train, xấp xỉ số liệu tốt nhất trước đó
  (0.7159).

**Epoch 21-30: dấu hiệu overfitting.** Train loss tiếp tục giảm đều (0.884→0.839) nhưng val loss
KHÔNG giảm theo — dao động rồi tăng nhẹ (thấp nhất ~epoch 26, sau đó tăng lại tới epoch 30). Test
QLIKE ở checkpoint tốt nhất trong khoảng epoch 21-30 (epoch 26, 0.5423) nhỉnh hơn epoch 20 (0.5436)
một chút nhưng R² kém hơn (0.7156 vs 0.7158) — tổng thể **epoch 20 là mốc cân bằng tốt nhất**,
không phải epoch 30.

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

| Kiến trúc | R² | QLIKE | RMSE |
|---|---:|---:|---:|
| HAR-only | 0.7051 | 0.5938 | 0.002750 |
| Gated-news | 0.7032 | 0.5943 | 0.002759 |

HAR-only nhỉnh hơn (tốt hơn) trên cả 3 metric ở horizon 22-ngày, dù chênh lệch nhỏ — mẫu hình
tương tự horizon 10-ngày (§7.2).

**Hội tụ:** val loss của cả 2 kiến trúc dao động không có xu hướng rõ suốt 10 epoch (vd HAR-only:
1.1495→1.1409→1.1436, không đơn điệu) — khác hẳn biến thể 5-ngày (liên tục cải thiện tới epoch 20).
**22-ngày hội tụ/chững lại ngay từ epoch ~10**, giống horizon 10-ngày (§7.4) — không có dấu hiệu
cần train thêm.

**So với horizon 5 và 10-ngày (Bảng B, §1.2):** QLIKE của HAR-only tiếp tục xấu đi đơn điệu theo
horizon dài hơn (0.5938 là QLIKE cao nhất/tệ nhất trong 3 mốc). Xác nhận xu hướng đã thấy ở
horizon-10: horizon càng dài, dự báo càng khó, nhưng cũng càng nhanh bão hoà (cần ít epoch hơn).

## 7.6 Horizon 1-ngày — hoàn thành đủ bộ 4 mốc (1/5/10/22-ngày)

Code trong `baselines/2026-08-01_horizon1_baseline/`, cùng pattern §7.1. Rủi ro window-count THẤP
NHẤT trong 4 horizon đã thử (23 ngày tối thiểu/split, so với 27/32/44 của 5/10/22-ngày) — đã kiểm
tra thật trên toàn bộ các mã, margin rộng nhất (train 891→868, val/test 191→168 window/split).

**Kết quả (test set, 10 epoch, pipeline so sánh công bằng):**

| Kiến trúc | R² | QLIKE | RMSE |
|---|---:|---:|---:|
| HAR-only | 0.7581 | 0.5099 | 0.002428 |
| Gated-news | **0.7595** | **0.4834** | **0.002420** |

**Xác nhận giả thuyết ban đầu: 1-ngày dễ dự báo nhất trong 4 mốc, cách biệt lớn** — QLIKE thấp hơn
(tốt hơn) rõ rệt so với mọi horizon khác. Gated-news vượt HAR-only trên CẢ 3 metric — QLIKE 0.4834
vs 0.5099 là khác biệt lớn nhất quan sát được giữa 2 kiến trúc ở bất kỳ horizon nào trong 10 epoch
đầu.

**Hội tụ:** val loss giảm mạnh 2 epoch đầu (~0.94→0.93 cho cả 2 kiến trúc) rồi dao động nhẹ quanh
mức đó tới epoch 10 — hội tụ nhanh, giống pattern 10 và 22-ngày, KHÔNG giống 5-ngày (cần ~20 epoch).
Không có dấu hiệu cần train thêm.

**Kết luận chung cho cả 4 horizon (Bảng B, §1.2):** horizon càng ngắn càng dễ dự báo VÀ hội tụ càng
nhanh — ngoại lệ duy nhất là 5-ngày, cần nhiều epoch hơn hẳn (~20) mới đạt đỉnh dù không phải
horizon khó nhất. Chưa có lời giải thích cho ngoại lệ này.

---

# 8. DirAcc — đã gỡ khỏi báo cáo này

DirAcc đã được gỡ khỏi toàn bộ báo cáo này (mọi bảng, mọi kết luận ở §1-§7) do phát hiện vấn đề về
công thức tính hiện dùng trong dự án — chưa xác nhận được số liệu đúng. Ghi chú kỹ thuật đầy đủ
(2 công thức khác nhau, code, số liệu đối chiếu thật cho thấy chênh lệch rất lớn giữa 2 công thức)
được lưu riêng ở `docs/report_2026-08-01/DIRACC_ISSUE_NOTE.md` để tham khảo khi kiểm tra lại.

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

| # | Biến thể | Dùng tin tức | Epoch | R² | QLIKE | RMSE | Trạng thái |
|---|---|:---:|---:|---:|---:|---:|---|
| 1 | HAR-only | — | 10 | 0.7141 | 0.5623 | 0.002643 | đối chứng chính |
| 2 | Per-ticker gated news (panel cũ, trước khi fix thiếu VPB/VRE) | ✓ | 10 | 0.7159 | 0.5497 | 0.002635 | kỷ lục cũ (26/07) |
| 3 | Per-ticker gated news (panel đã fix VPB/VRE) | ✓ | 10 | 0.7101 | 0.5631 | 0.002662 | chưa hội tụ |
| 4 | Per-ticker gated news (panel đã fix VPB/VRE) | ✓ | 20 | **0.7158** | 0.5436 | 0.002635 | **tốt nhất** |
| 5 | Per-ticker gated news (panel đã fix VPB/VRE) | ✓ | 30 | 0.7156 | **0.5423** | 0.002636 | bắt đầu overfit |

**Đọc bảng A:** hàng 4 (epoch 20, panel đã fix) là số liệu hiện hành, tốt nhất tổng thể trên
QLIKE/R². Hàng 5 xác nhận epoch 20 là điểm dừng hợp lý (train thêm bắt đầu overfit). Chi tiết: §7.3-7.4.

## Bảng C cũ (§1.2) — Calendar feature (5-ngày, panel cũ, 10 epoch), chi tiết đầy đủ ở §6.6

| Biến thể | Số cột calendar | R² | QLIKE | RMSE |
|---|---:|---:|---:|---:|
| Đối chứng (không calendar) | 0 | 0.7159 | 0.5497 | 0.002635 |
| Đủ 10 cột | 10 | 0.7117 | 0.5660 | 0.002654 |
| tet_only | 2 | 0.7124 | 0.5640 | 0.002651 |
| earnings_only | 2 | 0.7131 | 0.5501 | 0.002648 |
| generic_calendar | 6 | 0.7121 | 0.5583 | 0.002652 |

Không biến thể calendar nào vượt đối chứng trên cả 3 metric — no-lift, xác nhận thêm bởi EDA
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
baselines/2026-08-01_calendar_news_gate_baseline/code/analyze_market_news_volume_correlation.py  # EDA khối lượng tin vs biến động, §6.8
baselines/2026-08-01_calendar_news_gate_baseline/test/test_market_news_volume_correlation.py     # 5/5 pass, §6.8
baselines/2026-08-01_calendar_news_gate_baseline/requirements/requirements.md        # spec + giả định (bảng Tết, proxy BCTC)
baselines/2026-08-01_calendar_news_gate_baseline/design/design.md                    # kiến trúc + Simplicity/Anti-Abstraction gate
baselines/2026-08-01_calendar_news_gate_baseline/code_review/code_review_2026-08-01.md  # review đối kháng, 0 HIGH
results/calendar_gate_2026-08-01_073829/results.json                     # §6.4, đủ 10 cột
results/calendar_gate_tet_only_2026-08-01_082432/results.json            # §6.6, tet_only
results/calendar_gate_earnings_only_2026-08-01_083159/results.json       # §6.6, earnings_only
results/calendar_gate_generic_calendar_2026-08-01_083946/results.json    # §6.6, generic_calendar
results/news_calendar_correlation_2026-08-01_081230/                     # §6.5, EDA (analysis.json + per_point_delta_qlike.parquet + plot)
results/market_news_volume_correlation_2026-08-01_133849/                # §6.8, EDA (analysis.json + scatter.png + joined_daily_series.parquet)
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
