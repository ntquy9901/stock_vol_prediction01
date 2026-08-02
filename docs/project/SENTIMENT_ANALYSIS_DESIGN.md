# TÀI LIỆU THIẾT KẾ — TÍCH HỢP SENTIMENT ANALYSIS
## Mở rộng Parallel LSTM-GNN cho dự báo biến động VN30

**Ngày:** 04/07/2026
**Tác giả:** ntquy99
**Dự án:** Dự báo biến động cổ phiếu VN30 (5-day ahead volatility)
**Mục đích:** Tài liệu thiết kế kiến trúc — trình bày thầy/cô
**Trạng thái:** Thiết kế hoàn chỉnh, chưa triển khai code

---

## 1. TỔNG QUAN

### 1.1 Mục tiêu
Tích hợp đặc trưng sentiment từ tin tức tài chính vào mô hình **Parallel LSTM-GNN** hiện tại nhằm cải thiện độ chính xác dự báo biến động 5-day ahead cho 30 cổ phiếu VN30.

### 1.2 Bối cảnh & động lực
| Hạng mục | Hiện tại |
|----------|----------|
| Model tốt nhất | Parallel LSTM-GNN (k-NN graph, k=8) |
| Dir Acc | **69.98%** (vượt target 55%) |
| R² | **0.714** (vượt target 0.50) |
| QLIKE | 0.529 (gần target 0.50) |
| Input features | 3 HAR features (daily/weekly/monthly volatility) |
| Điểm yếu | QLIKE chưa < 0.50; Dir Acc sát ngưỡng 70% |

**Giả thuyết:** Tin tức tài chính mang tín hiệu dẫn dắt thị trường chưa được khai thác → thêm luồng sentiment có thể cải thiện Dir Acc vượt 70% và giảm QLIKE.

### 1.3 Nguyên tắc thiết kế (theo CLAUDE.md)
1. **Simplicity First** — bắt đầu bằng phương án đơn giản nhất (concat), đo, rồi mới phức tạp hóa.
2. **Chống data leakage** — tách thời gian, windowing đúng (chỉ dùng tin đã biết trong input window).
3. **Xử lý dữ liệu thiếu là tính năng, không phải lỗi** — model phải hoạt động tốt cả khi "mù" tin tức.
4. **Surgical** — không phá kiến trúc LSTM-GNN hiện tại, chỉ mở rộng input.

---

## 2. PHÂN TÍCH DỮ LIỆU TIN TỨC THỰC TẾ

### 2.1 Nguồn dữ liệu (3 file tại `data/raw/news/`)

| File | Số dòng | Phạm vi năm | Đóng góp |
|------|---------|-------------|----------|
| `data.csv` | 2.336 | 2020–2026 | tin gần đây + 2026 |
| `data_2021_2025.csv` | 5.030 | 2021–2025 | ✅ **Lấp gap test/val 2021-2025** |
| `data_archive.csv` | 8.495 | 2008–2023 | ✅ Lấp 2017-2019 + 2014-2015 |
| **Gộp (dedup)** | **~12.212 unique** | 2008–2026 | — |

### 2.2 Cấu trúc dữ liệu
```
id, title, source, date, pdf_url, pdf_filename, downloaded_at
```

**Đặc điểm quan trọng (phản ánh giả định ban đầu):**
- ⚠️ **Chỉ có `title`** — không có body bài viết (`pdf_filename` rỗng, PDF chưa download).
- ⚠️ **Không có cột ticker** — phải extract từ title (vd: "HND: Hưởng lợi...", "TCB và MBBank").
- ⚠️ **`date` format DD/MM/YYYY + nhiễu** — có dòng năm 2 chữ số cần làm sạch.
- ✅ Nguồn đa dạng: báo cáo phân tích từ các CTCK (KBSV, VCBS, FPTS, PHS, SSI...), Vietstock, Tổng cục Thống kê.

### 2.3 Phân bố thời gian & khoảng trống (gaps)

```
Năm   Số bài  Trạng thái     Biểu đồ
2008    181   OK          ███████
2009    203   OK          ████████
2010    376   OK          ███████████████
2011    713   OK          █████████████████████████████
2012    387   OK          ████████████████
2013    496   OK          ███████████████████
2014    501   OK          ████████████████████
2015    450   OK          ██████████████████
2016     18   LOW ▏       ← nhỏ (trong TRAIN)
2017    369   OK          ███████████████        ← ĐÃ LẤP
2018    307   OK          ████████████           ← ĐÃ LẤP
2019    453   OK          ██████████████████     ← ĐÃ LẤP
2020    936   OK          █████████████████████████████████████
2021   1176   OK          █████████████████████████████████████████████████
2022   1103   OK          ██████████████████████████████████████████████
2023   1067   OK          ███████████████████████████
2024   1190   OK          ████████████████████████████   ← ĐÃ LẤP
2025   1186   OK          ████████████████████████████   ← ĐÃ LẤP
2026   1100   OK          ███████████████████████████
```

### 2.4 Coverage ánh xạ vào train/val/test
(Theo split trong `BAO_CAO_CHO_THAY.md`: Train 2006-2020 / Val 2020-2021 / Test 2021-2026)

| Split | Giai đoạn | Số bài | Coverage | Đánh giá |
|-------|-----------|--------|----------|----------|
| **Train** (70%) | 2006–2020 | 5.390 | 2008-2019 ✅ (2016 thấp), 2020 ✅ | ✅ Tốt |
| **Val** (15%) | 2020–2021 | 2.112 | 2020 ✅, 2021 ✅ | ✅ Tốt |
| **Test** (15%) | 2021–2026 | **6.822** | **2021-2026全覆盖** ✅ | ✅✅ Rất tốt |

> **Cập nhật tích cực (04/07):** sau nhiều lần crawl, **test set giờ có 6.822 bài, toàn bộ 2021-2026 đã có tin** ✅. Rào cản "test mù tin" **đã giải quyết**. Chỉ còn gap nhỏ 2016 (18 bài) + 2006-2007.

### 2.5 Vấn đề dữ liệu cốt lõi

1. **Trùng lặp:** 3 file gộp ~16.000 dòng nhưng chỉ **~12.212 id duy nhất** → script dedup xử lý.
2. **Gap nhỏ còn lại:** chỉ 2016 (18 bài) + 2006-2007 → sentiment neutral ở vài ngày này (không ảnh hưởng).
3. **Title-only:** giới hạn độ tin cậy của feature phức tạp (độ sốc, tin nóng).
4. **Sparse per-stock:** chỉ **~16% bài** match mã VN30 cụ thể (2.063/12.212) → **1.851 stock-ngày có tin** / ~140k stock-ngày tổng.
5. **⚠️ SSB/min_length truncation** (xem memory `project-data-folders-split-issue`): model thực dùng 1.299 dòng đầu mỗi stock (theo vị trí), không phải cửa sổ "2021-2026" như BAO_CAO mô tả. Chưa reconcile.

---

## 3. KIẾN TRÚC TỔNG THỂ

Pipeline 6 bước, mỗi bước là một script riêng, idempotent, có cache trung gian → dễ debug và tái chạy từng bước.

```
data/raw/news/{news,data,data_archive}.csv
        │
        ▼  [1] DEDUP + LÀM SẠCH DATE
        │      - Merge 3 file theo id, bỏ 3.859 dòng trùng
        │      - Parse DD/MM/YYYY, sửa năm 2 chữ số, loại row sai
        │      → data/interim/news_clean.csv
        │
        ▼  [2] EXTRACT TICKER + TAG
        │      - Regex ^([A-Z]{2,5})[:\-] + match vocab VN30 trong title
        │      - Gán ticker_tags[] (1 bài có thể có nhiều mã)
        │      - Bài không match → bỏ hoặc nhóm "market-wide"
        │      → data/interim/news_tagged.csv
        │
        ▼  [3] SCORE SENTIMENT (NLP model)
        │      - sentiment_score ∈ [-1, 1], confidence_score ∈ [0, 1]
        │      → data/interim/news_scored.csv
        │
        ▼  [4] AGGREGATE per TICKER per DAY
        │      - Group by (ticker, date): mean(score), count, ...
        │      → data/interim/sentiment_daily/{TICKER}.csv
        │
        ▼  [5] ALIGN NGÀY GIAO DỊCH + XỬ LÝ THIẾU
        │      - Left join vào danh sách ngày giao dịch của từng stock
        │      - Fill gap: sentiment_1d=0, news_count_1d=0
        │      → data/sentiment_baseline/{TICKER}_sentiment.csv
        │
        ▼  [6] INTEGRATE vào DATASET CLASS
               - Load sentiment file, merge với HAR features theo date
               - Input tensor: [batch, seq_len, 30 stocks, 3 HAR + 2 sentiment = 5 features]
               - (Model LSTM-GNN giữ nguyên kiến trúc)
```

---

## 4. CHI TIẾT PIPELINE

### Bước 1 — Dedup & làm sạch date
- Merge 3 file, drop duplicate theo `id`.
- Parse date: chuẩn hóa về `YYYY-MM-DD`; nếu năm 2 chữ số → cộng 2000.
- Loại row có date không parse được hoặc ngoài khoảng 2006-2026.

### Bước 2 — Extract ticker
- **Vocab:** 30 mã VN30 (từ `data/processed/processing_summary.csv`).
- **Chiến lược 2 lớp:**
  1. Regex tiền tố: `^([A-Z]{2,5})\s*[:\-]` (vd "HND:", "GMD -").
  2. Fuzzy match: tìm mã VN30 xuất hiện trong title dạng từ độc lập (regex `\b(TICKER)\b`).
- Một bài gán cho **tất cả mã được nhắc** → edge "co-occurrence" tự nhiên.
- Bài không match mã nào → đánh tag `MARKET_WIDE` (dùng cho feature vĩ mô, tùy chọn).

### Bước 3 — Score sentiment
- Đầu vào: `title` (text ngắn tiếng Việt).
- Đầu ra: `sentiment_score` ∈ [-1, +1], `confidence_score` ∈ [0, 1].
- Model: xem **Mục 8** (đề xuất PhoBERT pretrained làm chính, LLM cross-check).

### Bước 4 — Aggregate per ticker per day
Một (ticker, ngày) có thể có nhiều bài → aggregate:
```
sentiment_score_1d  = mean(score các bài trong ngày)
news_count_1d       = số bài trong ngày
confidence_1d       = mean(confidence)
```

### Bước 5 — Align ngày giao dịch + xử lý thiếu (TRỌNG TÂM — xem Mục 5)
- Lấy danh sách ngày giao dịch từ `data/processed/{TICKER}_processed.csv`.
- Left join sentiment theo `date` (tin đăng ngày cuối tuần/lễ → map sang ngày giao dịch kế tiếp, xem Mục 9).
- Tính rolling 3d/5d với `min_periods=1`.
- Fill mọi gap bằng neutral + flag (không để NaN).

### Bước 6 — Integrate vào dataset
- Mở rộng `MultiStockDatasetWithGraphMethod`: load thêm sentiment file, concat vào feature dim.
- Input shape: `(batch, seq_len=22, 30 stocks, 5 features)` (3 HAR + 2 sentiment).
- **Không đổi** kiến trúc LSTM-GNN, chỉ tăng input dim.

---

## 5. XỬ LÝ DỮ LIỆU THIẾU (TRỌNG TÂM)

### 5.1 Nguyên tắc
> **Thiếu phải là TÍN HIỆU (informative), không phải NaN (destructive).**

Baseline daily phân biệt 2 trạng thái:
1. **Có tin hôm nay** (`news_count_1d > 0`) → `sentiment_1d` mang tín hiệu.
2. **Không có tin hôm nay** (`news_count_1d = 0`, `sentiment_1d = 0`) → neutral.

> Thiết kế đầy đủ có thể thêm `news_coverage_flag` để tách "yên tĩnh" vs "gap era" (xem Mục 6.2). Baseline gộp làm một cho đơn giản.

### 5.2 Ma trận chiến lược xử lý

| Tình huống | sentiment_1d | news_count_1d |
|-----------|--------------|---------------|
| Có tin trong ngày | mean(score) | > 0 |
| Không có tin / gap | 0 (neutral) | 0 |

**Quy tắc cứng:** không bao giờ để NaN trong feature file (LSTM-GNN không xử lý NaN).

### 5.3 Kỹ thuật: Missing Modality Learning (Modality Dropout)

Áp dụng kỹ thuật multimodal learning hợp lệ (Acar et al., *Multimodal learning with missing modality*) để model **không bị phụ thuộc quá mức** vào sentiment:

```python
# Trong training loop:
if training and random() < 0.25:        # modality dropout p=0.25
    x_sentiment[:]  = 0.0               # zero out feature sentiment
    x_news_count[:] = 0                 # zero out count → giống trạng thái "không tin"
# Ở inference: bỏ qua — test set tự sentiment=0, count=0
```

**Tác dụng:** model học pathway "không sentiment → dồn trọng số vào HAR + GAT". Khi test set mù tin (đúng như thực tế 2021-2025), model không bị sập.

**Lưu ý trung thực:** Vì dữ liệu sentiment **đã cực thưa sẵn** (đa số stock-day đã `news_count_1d=0`), model đã tự học phần nào robustness. Modality dropout là **regularizer phụ** (rẻ, vài dòng), không phải giải pháp chính —— không kỳ vọng nó cải thiện Dir Acc test.

### 5.4 Kiến trúc thay thế — DEFER (không làm giai đoạn này)

| Kỹ thuật | Lý do defer |
|----------|-------------|
| **Dynamic GAT + co-occurrence news edges** | (a) Đổi kiến trúc graph builder (hiện = volatility correlation 22-day); (b) **Topology mismatch**: train có news edge (2008-2020), test không có (2021-2025) → distribution shift; (c) Phải thêm DropEdge regularize. Để v2 khi có tin test + simple concat đã chứng minh có tín hiệu |
| **Gated fusion** (`g = σ(W·news_count_1d)` nhân sentiment) | Explicit, dễ interpret hơn dropout. Nhưng chỉ thêm nếu simple concat + dropout chưa đủ (luật Simplicity) |

---

## 6. LƯỢC ĐỒ ĐẶC TRƯNG (FEATURES)

Thiết kế **2 lớp** tách bạch: per-article (trung gian) → per-stock-day (đưa vào model).

### 6.1 Lớp per-article — `data/interim/news_scored.csv`

| Cột | Kiểu | Ý nghĩa | Nguồn fill |
|-----|------|---------|------------|
| `id` | str | ID gốc | giữ |
| `date` | date | Ngày xuất bản (đã sạch) | parse |
| `ticker_tags` | list[str] | Các mã VN30 được nhắc | NLP step 2 |
| `sentiment_score` | float [-1,1] | Sắc thái tài chính | NLP step 3 |
| `confidence_score` | float [0,1] | Độ tin cậy model | NLP step 3 |

### 6.2 Lớp per-stock-day — `data/sentiment_baseline/{TICKER}_sentiment.csv`

Thiết kế **daily đơn giản** (quyết định sau thảo luận): sentiment + count của **ngày đó** cho mã đó, không rolling, không flag. Lý do: bản thân HAR đã là rolling, và LSTM nhận cửa sổ 22 ngày nên tự học temporal pattern từ 22 giá trị daily — không cần pre-roll sentiment.

| Cột | Kiểu | Ý nghĩa | Khi thiếu |
|-----|------|---------|-----------|
| `date` | date | Ngày giao dịch | — |
| `sentiment_1d` | float [-1,1] | Sentiment trung bình **ngày đó** cho mã đó | 0 (neutral) |
| `news_count_1d` | int | Số bài tin **ngày đó** cho mã đó | 0 |

**Input model: 3 HAR + 2 sentiment = 5 feature/stock.**

**Đã bỏ (so với thiết kế đầu):**
- ~~`sentiment_5d`, `sentiment_3d` (rolling)~~ — LSTM tự aggregate từ 22 ngày daily.
- ~~`sentiment_confidence`~~ — chỉ có khi dùng NLP model (PhoBERT/LLM); baseline lexicon không có.
- ~~`news_coverage_flag`~~ — flag phân biệt "ngày yên tĩnh" vs "gap era"; bỏ cho đơn giản (baseline). Khi nào cần lại: nếu gap period dài (toàn 0) làm model học pattern sai.

> **Lưu trữ:** route **B** — file sentiment riêng trong `data/sentiment_baseline/`, left-join trong dataset class. Giữ `data/processed/*_processed.csv` (chỉ `date, parkinson_volatility`) sạch, dễ tái tạo sentiment độc lập.

### 6.3 Baseline implementation (ĐÃ BUILD — `src/sentiment_baseline/`)

Baseline cô lập 10 epoch đã hiện thực để test nhanh xem sentiment có tín hiệu không:

| File | Vai trò |
|------|---------|
| `lexicon.py` | Scorer thô tiếng Việt (pos/neg keyword) → `score(title) ∈ [-1,1]`. Baseline-only, dễ swap |
| `phobert_scorer.py` | **Scorer thay thế** dùng HuggingFace transformer (XLM-RoBERTa multilingual / PhoBERT head, lazy-load). Chạy qua `--scorer phobert`. Cùng scale `[-1,1]` → downstream không đổi. **Đã code, chưa run** (đợi 2 training hiện tại xong) |
| `process_news_to_sentiment.py` | Đọc `data/raw/news/`, dedup, parse date, extract ticker, score (`--scorer lexicon\|phobert`), aggregate per stock/day → `{out_dir}/{TICKER}_sentiment.csv` |
| `dataset_sentiment.py` | Subclass `MultiStockDatasetWithPreSplitData` + copy dataloader fn, gắn sentiment (5 feature). **Không sửa file cũ** |
| `train_sentiment_baseline.py` | Import `train_epoch`/`validate`/`EarlyStopping` từ trainer gốc. Hỗ trợ `--epochs`, `--resume_from` (train tiếp), `--sentiment_dir`. Output `results/sentiment_baseline_*` |

**Cô lập bảo đảm:** không sửa `src/lstm_gat_hybrid/`, `data/processed/`, hay kết quả cũ. Verify bằng `git status`.

**Scorer scale:** sentiment features được scale về cỡ HAR (~1e-3) trong subclass vì `VolatilityNormalizer` fit scalar global (xem memory `project-data-folders-split-issue`).

### 6.4 Kết quả thử nghiệm ban đầu (cập nhật 04/07/2026)

Baseline cô lập đã chạy (lexicon scorer, k-NN graph, 5 feature = 3 HAR + 2 sentiment). So sánh với HAR-only baseline:

| Cấu hình | Data | Epochs | Dir Acc | R² | QLIKE | RMSE |
|----------|------|--------|---------|------|-------|------|
| HAR-only baseline (tham chiếu) | — | 70 | **69.98%** | 0.714 | 0.529 | 0.002644 |
| Sentiment + lexicon | 9.334 bài | 10 | 68.17% | 0.714 | 0.561 | 0.002645 |
| Sentiment + lexicon | 9.334 bài | 15 | 68.21% | 0.714 | 0.558 | 0.002643 |
| Sentiment + lexicon | 9.334 bài | 20 | 68.57% | 0.713 | 0.546 | 0.002647 |
| Sentiment + lexicon | **11.025 bài (mới)** | 10 | 67.96% | 0.713 | 0.576 | 0.002647 |

**Quan sát trung thực:**
1. ✅ **Model khỏe:** R² (0.714) và RMSE (~0.002643-45) gần như **identical** với HAR baseline → sentiment không phá kiến trúc, pipeline end-to-end chạy đúng.
2. ⚠️ **DirAcc test tăng chậm**: 68.17% (10 ep) → 68.21% (15 ep) → **68.57% (20 ep)**; QLIKE cải thiện đều (0.561 → 0.546). Model vẫn học nhưng đang bão hòa, mỗi lượt +5 epoch chỉ thêm ~0.3-0.4% DirAcc.
3. 🔴 **KHÔNG thể kết luận** "sentiment làm DirAcc giảm" từ 68.2% vs 69.98% —— vì **10-15 epoch vs 70 epoch (không công bằng)**. Cần baseline HAR-only chạy cùng 10/15/20 epoch để so sánh công bằng.
4. 📉 **Lexicon thô + sparse**: chỉ 15-16% tin map được mã VN30 cụ thể, đa số stock-day `sentiment_1d=0` → tín hiệu yếu là **đúng kỳ vọng** với scorer thô này. **Review chất lượng lexicon** (đối chiếu cột `news_titles`): bắt ĐÚNG case rõ — "Khuyến nghị BÁN"→âm, "tăng trưởng/lợi nhuận"→dương (57% stock-ngày có tin ra non-zero) — nhưng MISS 43% (title chung chung như "Báo cáo phân tích cổ phiếu") và **lỗi phủ định** ("Lợi nhuận KHÔNG tăng trưởng" bị point +). → giới hạn cố hữu của keyword-matching, động lực chính để chuyển sang phobert.
5. 📉 **Tin thêm KHÔNG giúp (với lexicon)**: new-data 10-ep DirAcc **67.96%** ≈ old-data 68.17% (thậm chí hơi thấp hơn), QLIKE hơi tệ hơn (0.576 vs 0.561). → **Bottleneck là chất lượng scorer (lexicon thô), không phải lượng data** —— thêm tin chỉ thêm nhiễu. Đây là động lực chuyển sang scorer phobert.

**Tiếp theo (roadmap thực nghiệm):**
- Chạy **baseline HAR-only tại 10/15/20 epoch** (cùng setup, bỏ sentiment) → mới có so sánh công bằng để kết luận sentiment có tín hiệu không.
- Chạy **scorer phobert** (transformer) —— chất lượng sentiment tốt hơn lexicon → có thể ra tín hiệu rõ hơn.
- Nếu cả 2 đều không cải thiện Dir Acc test → sentiment (với data title-only hiện tại) không đủ tín hiệu → cần body bài viết hoặc crawldata giàu hơn.

---

## 7. TÍCH HỢP VỚI PARALLEL LSTM-GNN

Tiếp cận **3 phase tăng dần** (luật Simplicity — làm đơn giản trước). Sentiment đi từ "trộn thẳng vào input" → "gated riêng" → "nhánh độc lập + cross-attention".

Kiến trúc gốc Parallel LSTM-GNN (giữ nguyên ở mọi phase): **LSTM stream** (2 layers, hidden 64, per-stock temporal) + **GAT stream** (2 layers, 4 heads × 64 = 256 dim, dynamic k-NN graph, mean-pool 22 days) → **concat [64+256=320]** → **MLP 320→64→32→1**.

---

### Phase 1 — Simple Concatenation (ĐÃ BUILD baseline)

Sentiment **concat thẳng vào input** cùng HAR — không tách nhánh. Cả LSTM và GAT stream đều "thấy" sentiment.

```
INPUT TENSOR: [batch, seq=22, stocks=30, features=5]
   features = [har_daily, har_weekly, har_monthly,  sentiment_1d, news_count_1d]
                \____________ HAR (3) ___________/   \____ sentiment (2) ____/
                                  (gộp, không tách)
                                         │
                ┌────────────────────────┴────────────────────────┐
                ▼                                                 ▼
   ┌─────────────────────────┐                       ┌──────────────────────────┐
   │  LSTM STREAM (temporal)  │                       │  GAT STREAM (spatial)     │
   │                          │                       │                           │
   │  mỗi stock độc lập:      │                       │  mỗi timestep: 30 stocks  │
   │  22 days × 5 feat        │                       │  × 5 feat                 │
   │  → LSTM 2L, hidden=64    │                       │  → dynamic k-NN graph     │
   │  → last hidden state     │                       │    (xây/lấy từ 22-day win)│
   │  → [batch, 30, 64]       │                       │  → GAT 2L, 4 heads × 64   │
   │                          │                       │    = 256 dim              │
   │                          │                       │  → mean-pool qua 22 days  │
   │                          │                       │  → [batch, 30, 256]       │
   └────────────┬─────────────┘                       └─────────────┬────────────┘
                │                                                   │
                └──────────────────── CONCAT ───────────────────────┘
                                       │
                              [batch, 30, 64+256 = 320]
                                       │
                              MLP  320 → 64 → 32 → 1
                                       │
                              [batch, 30, 1]   ← dự báo biến động 5-day
```
- **Thay đổi vs model gốc:** chỉ `config.num_features_per_stock` 3 → 5. Code model KHÔNG đụng.
- **+ (tùy chọn) modality dropout** trong training loop.
- **Đo:** Dir Acc, QLIKE có cải thiện không. Baseline 10 epoch chạy trong `src/sentiment_baseline/`.

---

### Phase 2 — Gated Fusion (chỉ nếu Phase 1 có tín hiệu)

Tách HAR và sentiment ở input. HAR đi nhánh LSTM+GAT như cũ. Sentiment đi **nhánh riêng**, được **gate** (`σ(news_count_1d)`) điều khiển —— không tin thì gate≈0 → sentiment bị tắt.

```
INPUT tách theo loại feature:
   HAR (3 feat):        [batch, 22, 30, 3]  ───────────────────────────┐
   SENTIMENT (2 feat):  [batch, 22, 30, 2]  ──────────────┐             │
                                                         │             │
                                 ┌───────────────────────┘             │
                                 ▼                                     │
                  ┌────────────────────────────┐                       │
                  │  SENTIMENT BRANCH (riêng)   │                       │
                  │                             │                       │
                  │  22 days × 2 sentiment      │       ┌───────────────┴──────────────┐
                  │  → encoder nhỏ              │       │  HAR BRANCH (= Phase 1        │
                  │    (Linear hoặc 1-L LSTM)   │       │   chạy trên 3 HAR)            │
                  │  → mean-pool 22 days        │       │                               │
                  │  → [batch, 30, s]           │       │  LSTM(64) + GAT(256)          │
                  └──────────────┬──────────────┘       │  → concat [320]               │
                                 │                      │  → [batch, 30, 320]           │
                                 ▼                      └───────────────┬───────────────┘
                  GATE:  g = σ( W · news_count_1d )                      │
                         [batch, 30, 1]  ∈ (0,1)                        │
                         (g≈0 khi không có tin → sentiment OFF)         │
                                 │                                       │
                                 ▼                                       │
                  g ⊙ sentiment_emb   (element-wise gating)             │
                                 │                                       │
                                 └───────────── CONCAT ──────────────────┘
                                                │
                                       [batch, 30, 320 + s]
                                                │
                                       MLP  (320+s) → 64 → 1
                                                │
                                       [batch, 30, 1]
```
- **Lý do:** HAR branch học pattern biến động; sentiment branch chỉ đóng góp khi CÓ tin (gate mở). Tránh sentiment nhiễu (toàn 0) làm hỏng HAR.
- **Code:** thêm sentiment encoder + gate layer (~30 dòng) trong model. Dataset phải tách feature HAR vs sentiment ở `__getitem__`.

---

### Phase 3 — Late Fusion + Cross-Attention (MSGCA, SOTA 2025) — chỉ khi đã crawl đủ tin test

**3 nhánh hoàn toàn độc lập**, mỗi nhánh 1 encoder riêng. Fusion bằng **cross-attention có gate** (MSGCA) — HAR branches "hỏi" sentiment branch (Q←HAR, K,V←sentiment) để học stock/ngày nào nên chú ý tin nào.

```
                    ┌──────────────────┬───────────────────────┬────────────────────────┐
                    │ TEMPORAL (LSTM)   │ SPATIAL (GAT)          │ SENTIMENT (Transformer │
                    │                   │                        │  / self-attn encoder)  │
                    │ input: HAR        │ input: HAR             │ input: sentiment       │
                    │ per-stock         │ cross-stock + graph    │ per-stock              │
                    │ 22d × 3           │ 30 stocks, dynamic     │ 22d × 2                │
                    │                   │ k-NN graph             │                        │
                    │ → LSTM 2L, h=64   │ → GAT 2L, 4×64=256     │ → encoder              │
                    │ → [30, 64]        │ → [30, 256]            │ → [30, s]              │
                    └────────┬──────────┴───────────┬───────────┴────────────┬───────────┘
                             │                      │                        │
                             └── CROSS-ATTENTION (MSGCA, gated) ────────────┘
                                   • Q từ nhánh HAR (temporal+spatial)
                                   • K, V từ nhánh SENTIMENT
                                   • multi-head: học "stock/ngày nào attend tin nào"
                                   • gate σ(·): kiểm soát cường độ đóng góp của sentiment
                                              │
                                     FUSION LAYER (concat + MLP)
                                              │
                                     [batch, 30, 1]  dự báo
```
- **Lý do:** kiến trúc SOTA cho multi-modal (FNSPID, MSGCA 2025). Sentiment được học chọn lọc per-stock, per-day thay vì ảnh hưởng đồng loạt.
- **Khi nào:** chỉ sau khi (a) đã crawl đủ tin cho test, (b) Phase 1/2 đã chứng minh sentiment có tín hiệu.
- **Code:** model mới (~150-200 dòng), cross-attention layer + 3 encoders. Phức tạp nhất.

---

### So sánh 3 phase

| Phase | Sentiment vào đâu | Code đổi | Complexity | Khi nào dùng |
|-------|------------------|----------|-----------|--------------|
| **1. Concat** | input (cùng HAR) | 1 dòng config | ⭐ Thấp | ✅ Baseline (đã build) |
| **2. Gated** | nhánh riêng + gate | ~30 dòng | ⭐⭐ Trung bình | Phase 1 có tín hiệu |
| **3. MSGCA** | nhánh riêng + cross-attn | ~200 dòng, model mới | ⭐⭐⭐ Cao | Đủ tin test + Phase 2 ok |

---

## 8. LỰA CHỌN MODEL NLP SENTIMENT

| Phương án | Ưu | Nhược | Khuyến nghị |
|-----------|-----|-------|-------------|
| **A. PhoBERT sentiment pretrained (HuggingFace)** | Reproducible, không API cost, chuẩn research VN | Cần chọn head tốt; domain tài chính VN hạn chế | ✅ **Làm chính** |
| **B. LLM scoring (DeepSeek / local LLM MCP)** | Chính xác jargon tài chính, trả score+confidence, không cần label | Cost/reproducibility, chậm | ✅ **Cross-check** trên sample 200 dòng |
| **C. Lexicon tài chính VN** | Đơn giản nhất, không GPU | Ít chính xác | Baseline nhanh (tùy chọn) |

**Quyết định:** **A làm chính, B cross-check** trên mẫu 200 dòng để validate A. **Không tự fine-tune** (thiếu labeled data → over-engineering). Dataset nhỏ (~4.500 bài) → cost LLM chấp nhận được.

---

## 9. CHỐNG DATA LEAKAGE

### 9.1 Alignment theo `date` + windowing (không dùng T+1)

**Vì sao KHÔNG dùng T+1:**
- Dữ liệu chỉ có `date` (KHÔNG có timestamp giờ) → không xác định được tin ra trước/sau giờ đóng cửa → T+1 chính xác không tính được.
- Với dự báo **5-day-ahead** (input đến ngày T, target T+5): tin ngày T (dù cuối ngày) vẫn đã biết trước khi dự báo T+5 → dùng `date` gốc **không gây lookahead**.
- T+1 chỉ thực sự cần cho dự báo same-day / 1-day → không áp dụng cho project này.

**Leakage được chống bằng windowing (đã đủ):**
```
Quy tắc: trong snapshot kết thúc ngày T (input days [i, T]),
         chỉ đưa tin có date ≤ T vào feature sentiment.
         → Không bao giờ dùng tin tương lai làm input.
```

**Xử lý ngày cuối tuần/lễ (chi tiết trong code, không thành cột):**
- Tin đăng ngày không giao dịch → map sang ngày giao dịch kế tiếp (trong bước left-join lịch).
- Tin ngày giao dịch → giữ nguyên `date`.

**Đây là lỗi cùng kiểu** với data leakage (`random_split`) đã được khắc phục trong project —— phải tránh bằng mọi giá.

### 9.2 Temporal split giữ nguyên
- Split 70/15/15 theo thời gian, KHÔNG random.
- Sentiment file được tạo **sau khi split** (HAR features tính riêng per split) → không leakage từ rolling.

---

## 10. GIỚI HẠN & RỦI RO (đánh giá trung thực)

| # | Giới hạn | Tác động | Giảm nhẹ |
|---|----------|----------|----------|
| 1 | **Test set 2021-2025 không có tin** | Sentiment không cải thiện được metric test (chỉ 2026 có tin) | **Crawl thêm 2021-2025** (ưu tiên cao nhất) |
| 2 | **Chỉ có title, không body** | Feature phức tạp (độ sốc, tin nóng) ít tin cậy | Giữ feature tối giản; download PDF sau |
| 3 | **Sparse per-stock** (nhiều bài vĩ mô) | Số tin per-stock/day thực tế thấp | Thêm `market-wide` sentiment làm fallback |
| 4 | **Trùng lặp 3.859 dòng** | Bloat data | Dedup bước 1 |
| 5 | **Model NLP chưa validate trên domain VN tài chính** | Sentiment có thể nhiễu | Cross-check LLM + confidence filter |
| 6 | **Gap còn lại (2024-2025, 2016 thấp, 2006-2007)** | Sentiment neutral ở các khoảng này | neutral fill (gap nhỏ) |

**Kỳ vọng thực tế (trung thực):**
- Nếu **không crawl thêm tin test**: sentiment chủ yếu giúp **train/val interpretation**, **không nâng metric test**.
- Nếu **crawl đủ tin 2021-2026**: kỳ vọng +2-5% Dir Acc, -5-10% QLIKE (theo nghiên cứu SOTA trong `project-context.md`).

---

## 11. LỘ TRÌNH TRIỂN KHAI

| Phase | Việc | Verify | Thời gian |
|-------|------|--------|-----------|
| **0 — Data** | Crawl tin 2021-2025 + 2016-2019 | Coverage > 80% mọi năm | (phụ thuộc nguồn) |
| **1 — Pipeline NLP** | Bước 1-5 (dedup → sentiment file) | Sentiment file cho 30 mã, không NaN | 1-2 tuần |
| **2 — Integrate Phase 1** | Concat 6 features + modality dropout | Val Dir Acc cải thiện | 3-5 ngày |
| **3 — Đánh giá** | So sánh vs baseline, ablation | Bảng metric 6 tiêu chí | 2-3 ngày |
| **4 — (Tùy chọn) Phase 2/3** | Gated fusion / MSGCA | Có tín hiệu từ Phase 1 | 1-2 tuần |

**Điểm quyết định (go/no-go):** sau Phase 2, nếu val Dir Acc không cải thiện → sentiment không có tín hiệu, dừng, không đầu tư Phase 3.

---

## 12. PHỤ LỤC — ĐÁNH GIÁ CÁC ĐỀ XUẤT THAY THẾ

### 12.1 Đề xuất 10 cột từ chuyên gia (đánh giá chọn lọc)

| Cột đề xuất | Verdict | Lý do |
|-------------|---------|-------|
| `ticker_tags` | ✅ Adopt | Bắt buộc — không có cột ticker |
| `sector_tags` | ⏸️ Defer | Đổi kiến trúc GAT (graph hiện = volatility corr) |
| `sentiment_label` | ❌ Drop | Redundant với `sentiment_score` |
| `sentiment_score` | ✅ Adopt | Core feature |
| `volatility_impact_score` | ⏸️ Defer | **Circular**: NLP đoán đúng thứ model dự báo; khó validate |
| `event_category` | ⏸️ Defer | Categorical cần embedding, phình feature dim |
| `is_breaking_news` | ❌ Drop | Không suy ra đáng tin từ title ngắn |
| `confidence_score` | ✅ Adopt | Filter noise + feature |
| `trading_date_applied` | ❌ Drop | Không có timestamp; 5-day horizon không cần T+1; windowing đã chống leakage (Mục 9.1) |
| `source_reliability_weight` | ⏸️ Defer | Chủ quan, khó justify |

**Kết luận:** adopt **3/10** (ticker_tags, sentiment_score, confidence_score), drop 1 (`trading_date_applied` — không có timestamp, không cần cho 5-day), defer 6 (lý do: speculative, circular, hoặc đổi kiến trúc —— vi phạm luật Simplicity).

### 12.2 Kỹ thuật Missing Modality Learning
- ✅ Kỹ thuật **hợp lệ**, adopt phần rẻ (modality dropout vài dòng).
- ⚠️ **Sửa timeline:** tin thực tế ở 2008-2015 + 2020 (train/val), KHÔNG phải 2021+ → áp dụng dropout trên toàn train, không phải "chỉ 2021-2026".
- ⚠️ Mua **robustness** (không bị sập khi thiếu), **không mua test lift**.

### 12.3 Dynamic GAT + co-occurrence edges
- ⏸️ **Defer** — topology mismatch train/test + đổi kiến trúc. Để v2 khi có tin test.

---

## 13. KẾT LUẬN

1. **Kiến trúc đề xuất:** pipeline 6 bước + 2 feature sentiment daily (`sentiment_1d`, `news_count_1d`) + tích hợp simple concat (5 feature input). Baseline 10 epoch đã build trong `src/sentiment_baseline/`.
2. **Giải quyết dữ liệu thiếu:** neutral fill (`sentiment_1d=0` khi không tin) + (tùy chọn) modality dropout → model hoạt động cả khi mù tin.
3. **Chống leakage:** windowing (date ≤ cuối input window) + temporal split.
4. **Trung thực về giới hạn:** test set 2021-2025 chưa có tin → **crawl thêm là ưu tiên số 1**, quan trọng hơn mọi kỹ thuật.
5. **Đường đi:** Phase 0 (crawl) → Phase 1 (pipeline) → Phase 2 (concat) → go/no-go → Phase 3 (chỉ nếu có tín hiệu + tin test).

**Thông điệp cho thầy/cô:** Thiết kế tuân thủ nguyên tắc simplicity-first và chống data leakage; lựa chọn feature đã được sàng lọc kỹ (chỉ giữ các feature trace được đến nhu cầu cụ thể); đánh giá trung thực giới hạn dữ liệu hiện tại và điều kiện (crawl thêm tin) để sentiment thực sự phát huy tác dụng.

---

**Tài liệu liên quan:**
- Báo cáo model hiện tại: `docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` (báo cáo model cũ
  `docs/report_2026-06-27/BAO_CAO_CHO_THAY.md` đã archive 2026-08-02, xem
  `archive/docs_reports_legacy/report_2026-06-27/`)
- Kiến trúc LSTM-GNN: `docs/project/LSTM_GAT_ARCHITECTURE.md`
- Nghiên cứu SOTA fusion: `_bmad-output/planning-artifacts/research/technical-sentiment-volatility-fusion-sota-2026-06-29.md`
- Project context: `project-context.md` (mục Sentiment Analysis Integration)
- **Mở rộng Phase 3+ (góp ý thầy, 05/07):** `SENTIMENT_LATENT_SPACE_TECHNIQUES.md` (kỹ thuật latent space / VIB chống data thưa) · `SENTIMENT_NEWS_EMBEDDING_ARCHITECTURE.md` (dùng news embedding vector 768-dim thay sentiment score)

**Phiên bản:** 1.0 — 04/07/2026
