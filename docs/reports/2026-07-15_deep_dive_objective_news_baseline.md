# Deep-dive: Kiến trúc, tổ chức dữ liệu và nguyên nhân accuracy thấp — Objective News Baseline

**Baseline:** `baselines/2026-07-15_objective_news_baseline/` · Ngày review: 15/07/2026
**Kết quả cần giải thích:** Test DirAcc **67.87%** (10 epoch) — thấp nhất trong tất cả biến thể
news đã thử (HAR-only 69.98% > latent-noise 69.33% > embedding-baseline 68.76% > **cái này 67.87%**).

---

## 1. Kiến trúc tổng thể (end-to-end)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RAW DATA (READ-ONLY, ngoài repo)                                             │
│  D:/bmad-projects/crawl_data/data/objective/                                 │
│  ├─ vietstock_records.csv   (670 dòng, company_code CÓ SẴN)                  │
│  ├─ vsdc_records.csv        (6 dòng,   company_code CÓ SẴN)                  │
│  └─ news_unenriched_{vnexpress,tuoitre,thanhnien,vietnamplus,nld}.csv        │
│       (~400 dòng, company_code RỖNG — tin phổ thông)                        │
└──────────────────────────────┬────────────────────────────────────────────┘
                                │  extract_objective_embeddings.py
                                │  1. match ticker: company_code trực tiếp
                                │     HOẶC ticker-regex HOẶC brand-alias (NAME_ALIASES)
                                │  2. drop nếu thiếu ngày / ngày tương lai (leakage guard)
                                │  3. PhoBERT encode (frozen, vinai/phobert-base, 768-d)
                                │  4. PCA 768→64 (fit 1 lần trên 205 record TRƯỚC 2020-01-01)
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ data/objective_embedding/{TICKER}_emb.npz   (24/30 mã có ít nhất 1 ngày;     │
│   {date_str: [n_articles, 64]}               6/30 mã RỖNG hoàn toàn)         │
│ + _manifest.json (incremental) + _pca.pkl (persisted, không refit)          │
└──────────────────────────────┬────────────────────────────────────────────┘
                                │  create_embedding_dataloaders()
                                │  (dataset_embedding.py, KHÔNG sửa — chỉ trỏ --emb_dir mới)
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MultiStockDatasetWithEmbedding  (subclass MultiStockDatasetWithPreSplitData) │
│  → xem chi tiết mục 3 (tổ chức train/val/test)                              │
│  Output 1 sample:                                                            │
│   x_har [22, 32, 3]   adj [32,32]   x_emb [22,32,10,64]   mask [22,32,10]    │
│                        y [32]  (Parkinson volatility, 5 ngày sau cửa sổ)     │
└──────────────────────────────┬────────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ EmbeddingBaseline (model_embedding.py) — KHÔNG sửa, tái dùng nguyên bản      │
│                                                                               │
│  x_har,adj ──► ParallelLSTMGNN.get_embeddings() ──► h_lstm[B,32,64]          │
│                (per-stock LSTM 1L)                   h_gnn [B,32,256]        │
│                (GAT 4 head×64, mean theo 22 ngày)         │                  │
│                                                            │                  │
│  x_emb,mask ──► ArticleSetAttentionPooling ──► daily[B,32,22,64]             │
│                 (attention theo query học được;                             │
│                  ngày KHÔNG có tin → no_news_token cố định)                  │
│                     │                                                        │
│                     ▼                                                       │
│                NewsTemporalEncoder (LSTM 1L, 22 bước) ──► news_rep[B,32,64]  │
│                                                            │                  │
│         h_lstm[64] ⊕ h_gnn[256] ⊕ news_rep[64] = concat[384] ──► fusion MLP  │
│                                            384→64→32→1 (ReLU+Dropout 0.5)     │
│                                                            │                  │
│                                                     ŷ [B, 32]  (volatility)  │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    train_embedding_baseline.py (sibling, KHÔNG sửa)
                    MSE loss, Adam lr=5e-3, wd=1e-5, ReduceLROnPlateau,
                    grad_clip=1.0, 10 epoch → evaluate_predictions (6 metric)
```

---

## 2. Kiến trúc model chi tiết (dims thật, đọc từ `LSTMGATConfig`)

| Thành phần | Nguồn | Dim | Ghi chú |
|---|---|---|---|
| HAR-LSTM (`h_lstm`) | `ParallelLSTMGNN.get_embeddings` (`model_parallel.py:210-216`) | 64 | 1 LSTM/stock, lấy `h_n[-1]` — cuối chuỗi 22 ngày |
| HAR-GAT (`h_gnn`) | `model_parallel.py:218-226` | 4 head × 64 = **256** | GAT áp cho từng ngày trong 22 ngày, rồi **mean-pool theo thời gian** |
| News pool (`daily`) | `ArticleSetAttentionPooling` (`model_embedding.py:25-60`) | 64 | Attention có trọng số học được (`query`), ngày rỗng → `no_news_token` |
| News temporal (`news_rep`) | `NewsTemporalEncoder` (`model_embedding.py:63-80`) | 64 | 1-layer LSTM qua 22 ngày, lấy state cuối |
| Fusion input | `torch.cat([h_lstm, h_gnn, news_rep])` | 64+256+64=**384** | |
| Fusion MLP | `model_embedding.py:110-114` | 384→64→32→1 | Dropout 0.5 mỗi lớp |

**Nhánh HAR chiếm 320/384 = 83% chiều fusion input; nhánh news chỉ 64/384 = 17%.** Nhưng vấn đề
không phải tỷ lệ chiều, mà là **chất lượng tín hiệu bên trong 64 chiều đó** — xem mục 4.

---

## 3. Tổ chức dữ liệu Train / Validation / Test

### 3.1 Nguồn giá + tách theo THỜI GIAN TRƯỚC KHI sinh feature (chống leak)

Log thật từ lần chạy cuối (`train_embedding_baseline.py --emb_dir data/objective_embedding`):

```
[_load_raw_stock_data] Successfully loaded 32 stocks     (data/processed/*.csv, KHÔNG chỉ 30 VN30)
[_load_raw_stock_data] Total outliers removed: 2096       (n_std=3, theo từng mã riêng lẻ)

[_split_raw_data_by_date] Chronological split by DATE index:
  Min length across stocks: 1273 (ngày giao dịch — mã ngắn nhất quyết định)
  Train: [0, 891)      → 891 ngày   (70%)
  Val:   [891, 1082)   → 191 ngày   (15%)
  Test:  [1082, 1273)  → 191 ngày   (15%)
```

Cơ chế (`src/lstm_gat_hybrid/dataset_with_graph_method.py`):
1. `_load_raw_stock_data()` (dòng 551) — load OHLCV thô từng mã, loại outlier **trước khi** cắt
   split (theo n_std=3, tính riêng mỗi mã).
2. `_split_raw_data_by_date()` (dòng 624) — cắt theo **chỉ số ngày** (không phải random), tỷ lệ
   70/15/15 dựa trên `min_length` (mã ngắn nhất trong 32 mã) → đảm bảo mọi mã dùng CÙNG một mốc
   ngày cắt (không lệch pha giữa các mã).
3. `_generate_har_for_split()` (dòng 684) — **HAR (daily/weekly/monthly rolling mean) được tính
   RIÊNG cho từng split** (train/val/test) — KHÔNG tính trên toàn bộ chuỗi rồi cắt, để rolling
   window monthly (22 ngày) không "nhìn" sang dữ liệu tương lai (val/test) khi tính train.

### 3.2 Cửa sổ trượt (sequence) + gắn embedding

`MultiStockDatasetWithEmbedding._create_sequences()` (`dataset_embedding.py:94-179`):
- `seq_length=22` (1 tháng giao dịch), `forecast_horizon=5` → mỗi sample dự báo volatility
  **5 ngày sau khi kết thúc cửa sổ 22 ngày** (target index = `i + 22 + 5 - 1`).
- Với mỗi vị trí cửa sổ `i`, với MỖI mã: lấy 22 ngày HAR (`x_har`), rồi với MỖI ngày trong 22
  ngày đó, tra cache embedding theo **khớp ngày CHÍNH XÁC** (`_norm_date`, dòng 29-35) —
  KHÔNG nội suy/lan truyền từ ngày gần nhất. Không khớp → vector 0 + mask 0 (dòng 148-159).
- Đồ thị (`adj`) dựng theo `graph_method=knn` — k-NN trên tương quan volatility 22 ngày, tính
  lại cho MỖI cửa sổ (không cố định toàn cục).

**Số sequence thực tế** (log dòng `[emb] sequences -`):
```
train=864, val=164, test=164
```
(864 = 891 ngày − 22 − 5, tương tự cho val/test — đúng công thức `min_length_split − seq − horizon`)

### 3.3 Coverage tin tức thực đo được trong TỪNG split (bằng chứng trực tiếp cho mục 4)

`_create_sequences()` đếm `_matched_cells / _total_cells` (mỗi cell = 1 tổ hợp
cửa-sổ×mã×ngày-trong-cửa-sổ) và IN RA LOG — đây là số liệu **đo trên chính run cuối cùng**,
không phải ước lượng:

```
[emb] date-match coverage: 1693/608256 cells (0.28%)   ← TRAIN
[emb] date-match coverage: 246/115456  cells (0.21%)   ← VAL
[emb] date-match coverage: 261/115456  cells (0.23%)   ← TEST
```

So sánh: nhánh tin tức hiện tại (báo cáo phân tích, baseline 2026-07-07) có coverage ước tính
~5.5% (từ dữ liệu đã lớn hơn ~20 lần). Nói cách khác: **>99.7% các cell trong TẤT CẢ 3 split đều
là "không có tin"** — xem mục 4 để hiểu hệ quả.

### 3.4 Chuẩn hoá (normalize)

`create_embedding_dataloaders()` (`dataset_embedding.py:250-268`): `VolatilityNormalizer`
(StandardScaler wrapper) fit **CHỈ trên train** cho từng mã (cả HAR feature lẫn target `y`), rồi
áp dụng (transform, không fit lại) cho val/test — đúng nguyên tắc chống leak đã ghi trong
CLAUDE.md §10. Embedding tin tức (`x_emb`) **KHÔNG được normalize** (PCA đã đưa về scale hợp lý,
comment `dataset_embedding.py:42`).

---

## 4. Vì sao accuracy thấp — phân tích nguyên nhân

### 4.1 Nguyên nhân chính: coverage cực thấp làm nhánh news gần như hằng số

`ArticleSetAttentionPooling.forward()` (`model_embedding.py:50-60`):
```python
has_news = (mask.sum(-1, keepdim=True) > 0).to(daily.dtype)
daily = has_news * daily + (1 - has_news) * self.no_news_token
```
Khi `has_news=0` (>99.7% trường hợp, theo số đo mục 3.3), `daily` = **đúng một vector hằng số học
được** (`no_news_token`, 64 chiều) — GIỐNG HỆT NHAU cho mọi mã, mọi ngày, mọi cửa sổ không có tin.
Sau đó `NewsTemporalEncoder` chạy LSTM qua 22 bước — nếu cả 22 ngày đều là `no_news_token` (rất
phổ biến vì coverage <0.3%), LSTM chỉ đang lặp lại đúng 1 input suốt 22 bước → `news_rep` gần như
**một hằng số duy nhất cho toàn bộ dataset** (sai khác nhỏ tuỳ epoch/khởi tạo, không tuỳ ngày/mã).

**Hệ quả:** nhánh news lẽ ra phải "im lặng" (đóng góp ~0 vào fusion) để không cản trở nhánh HAR,
nhưng nó vẫn là **384 → 64 → 32 → 1 MLP có 384 chiều input, học từ đầu (không pretrain)**, tốn
tham số + gradient để học cách "bỏ qua" input gần-hằng-số này, mà với chỉ 10 epoch (theo Training
policy CLAUDE.md, thử nghiệm ban đầu) mô hình **chưa đủ thời gian hội tụ về trạng thái "bỏ qua
sạch"** — dẫn tới nhiễu dư thừa so với chạy HAR-only (không có nhánh news, không có tham số để
học "bỏ qua" gì cả).

### 4.2 PCA fit trên mẫu rất nhỏ (205 record) — nguy cơ basis không đại diện

Log: `PCA 768->64 (fit on 205 train records, explained var: 0.938)`. 205 record train-period gần
như toàn bộ đến từ `vietstock_records.csv` lịch sử (2005-2019, thông báo cổ tức/phát hành — văn
phong lặp lại rất nhiều: "Trả cổ tức đợt X/YYYY bằng tiền, Z đồng/CP"). PCA 64 chiều fit trên tập
mẫu **lặp lại mẫu câu cao, đa dạng chủ đề thấp** này rất có thể học ra các trục chính chủ yếu phân
biệt "loại sự kiện" (cổ tức vs phát hành vs ĐHCĐ) — không nhất thiết đại diện tốt cho phân bố văn
bản **đa dạng hơn nhiều** ở test-period (2021-2026, có cả tin brand-name như "Vinamilk lỗ 50%").
→ 64 chiều test-period sau PCA có thể là hình chiếu **kém khớp**, thêm nhiễu thay vì tín hiệu.
(So sánh: baseline 2026-07-07 dùng báo cáo phân tích — n_train PCA lớn hơn nhiều vì corpus lớn
hơn ~20 lần → basis ổn định hơn.)

### 4.3 6/30 mã KHÔNG có bất kỳ record nào (từ log extraction: "wrote 24/30 ticker caches")

Với 6 mã này, `mask` luôn = 0 tuyệt đối ở MỌI cửa sổ/ngày → `news_rep` cho các mã này = đúng
`no_news_token` đã qua LSTM 22 bước giống hệt nhau ở MỌI sample — tham số fusion cho các mã này
không có cách nào phân biệt được "mã X không có tin" khác gì "mã Y không có tin" (embedding đầu
vào giống hệt), lãng phí công suất nhưng không trực tiếp gây hại (không thêm nhiễu thời gian, chỉ
thêm 1 hằng số dùng chung).

### 4.4 So sánh có kiểm soát (không epoch-matched, nhưng cùng kiến trúc)

| Baseline | Nguồn tin | Coverage (ước tính) | Epoch | Test DirAcc |
|---|---|---|---|---|
| HAR-only | — (không có nhánh news) | — | 70 | **69.98%** |
| Embedding baseline | báo cáo phân tích CTCK | ~5.5% | 40 | 68.76% |
| Latent noise | báo cáo phân tích + noise train-only | ~5.5% | 10 | 69.33% |
| **Objective news** | sự kiện DN + tin brand-name | **0.21-0.28%** (đo trực tiếp) | 10 | **67.87%** |

Cả 3 biến thể có nhánh news đều dùng **kiến trúc giống hệt nhau** (`EmbeddingBaseline`, không
sửa) — biến số khác nhau DUY NHẤT là **input embedding** (nguồn dữ liệu + coverage). Objective
news có coverage thấp hơn ~20 lần so với 2 biến thể kia, và đạt kết quả **thấp hơn cả 2 biến thể
kia lẫn HAR-only** — nhất quán với giả thuyết ở mục 4.1: khi coverage quá thấp, nhánh news không
còn là "thêm tín hiệu yếu" mà trở thành "thêm tham số + nhiễu ròng" trong ngân sách 10 epoch.

### 4.5 Yếu tố PHỤ (không phải nguyên nhân chính, nhưng đáng ghi nhận)

- **Không epoch-matched** với HAR-only (10 vs 70) — nhưng latent-noise cũng chỉ 10 epoch và vẫn
  vượt HAR gần bằng, nên đây không phải lý do chính (nếu là lý do chính, latent-noise cũng phải
  thấp tương tự).
- **32 mã** thay vì đúng 30 VN30 (kế thừa từ pipeline chung, không phải lỗi riêng baseline này) —
  không ảnh hưởng đến so sánh vì tất cả biến thể đều dùng chung `data/processed` 32 mã.
- **Outlier removal riêng từng mã** (2096 điểm bị loại, ~2%) — chuẩn, giống các baseline khác.

---

## 5. Code walkthrough (file → vai trò → dòng chính)

| File | Vai trò | Điểm cần chú ý |
|---|---|---|
| `code/extract_objective_embeddings.py` | Trích embedding từ raw objective CSV | `build_records()` (dòng 114-227): match ticker + brand-alias, leakage guard, dedup, incremental skip. `main()` (230+): PhoBERT encode chỉ record MỚI, PCA persist/reuse, merge cache. |
| `2026-07-07_embedding_baseline/code/dataset_embedding.py` (đọc, không sửa) | Build sequence + gắn embedding theo ngày | `_create_sequences()` (94-179): vòng lặp cửa sổ trượt, đếm coverage, RAISE nếu 0% (an toàn — không silent fail). `create_embedding_dataloaders()` (208+): split raw trước, sinh HAR riêng từng split, fit normalizer chỉ trên train. |
| `2026-07-07_embedding_baseline/code/model_embedding.py` (đọc, không sửa) | Kiến trúc model | `ArticleSetAttentionPooling` (25-60): attention + `no_news_token` (điểm mấu chốt mục 4.1). `NewsTemporalEncoder` (63-80): LSTM 22 bước. `EmbeddingBaseline.forward` (116-129): concat 3 nhánh → fusion. |
| `2026-07-07_embedding_baseline/code/train_embedding_baseline.py` (đọc, không sửa) | Training loop | Chạy thẳng qua CLI `--emb_dir data/objective_embedding`, không có dòng code riêng cho baseline này (đúng nguyên tắc Anti-Abstraction Gate — không viết train script trùng lặp). |
| `src/lstm_gat_hybrid/dataset_with_graph_method.py` (đọc, không sửa) | Split + HAR + outlier | `_load_raw_stock_data` (551), `_split_raw_data_by_date` (624), `_generate_har_for_split` (684) — chuỗi 3 hàm này đảm bảo KHÔNG leak tương lai vào train. |
| `src/lstm_gat_hybrid/model_parallel.py` (đọc, không sửa) | HAR-LSTM + HAR-GAT | `get_embeddings()` (196-226): LSTM riêng từng mã, GAT riêng từng ngày rồi mean-pool 22 ngày. |

---

## 6. Kết luận

**Root cause chính:** coverage tin tức 0.21-0.28% (đo trực tiếp từ log, không phải ước lượng) —
thấp đến mức nhánh news suy biến gần thành hằng số (`no_news_token` qua LSTM lặp) ở >99.7%
trường hợp, khiến 384-chiều fusion input phải "học cách bỏ qua" một nhánh gần-vô-nghĩa trong ngân
sách chỉ 10 epoch, kéo kết quả xuống DƯỚI CẢ HAR-only (không có nhánh news) lẫn 2 biến thể news
khác có coverage cao hơn ~20 lần.

**Không phải do:** thuật toán ticker/brand-alias matching (đã review + fix ở code review trước),
không phải do kiến trúc (giống hệt 2 biến thể kia), không phải do temporal split/leakage (đã kiểm
tra kỹ, đúng chuẩn CLAUDE.md).

**Khuyến nghị:** không đầu tư thêm cho hướng "objective corporate-event data" trừ khi tăng được
volume crawl tin phổ thông (hiện 4/5 nguồn gần như 0 tin khớp mã) — đúng kết luận NO-GO đã ghi ở
báo cáo trước (`2026-07-15_0230_objective_news_baseline_report.md`).
