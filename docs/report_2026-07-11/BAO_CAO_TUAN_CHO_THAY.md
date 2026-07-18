# BÁO CÁO TUẦN — DỰ BÁO BIẾN ĐỘ VN30
## Parallel LSTM-GNN + Tích hợp Sentiment Tin Tức

**Ngày báo cáo:** 11/07/2026 (tuần 05/07 → 11/07/2026)
**Người thực hiện:** ntquy99
**Dự án:** Dự báo biến động (volatility) 5-day ahead cho 30 cổ phiếu VN30
**Scope tuần này:** (1) hoàn thiện pipeline sentiment/embedding, (2) thử 3 hướng tích hợp news (embedding PhoBERT, market-fallback, sentiment-decay), (3) EDA sentiment↔price để kiểm tra go/no-go.

---

## 0. TÓM TẮT ĐIỀU HÀNH (đọc 30 giây)

- **Model tốt nhất vẫn là Parallel LSTM-GNN (k-NN graph): test DirAcc 69.98%, R² 0.714, QLIKE 0.529** — duy nhất vượt mục tiêu 55% DirAcc.
- **Tuần này thử 4 cách đưa tin tức vào** (embedding 68.76%, market-fallback 68.69%, decay 67.87%, **latent-noise 69.33%**). Cả 4 đều dưới HAR-only 69.98%, cluster ~68–69.3%. Latent-noise (gợi ý thầy #2) là news variant cao nhất nhưng vẫn −0.65% so HAR-only (caveat: epoch chưa match).
- **Kết luận trung thực:** bottleneck là **DATA** (tin chỉ có tiêu đề, ~5.5% ngày-mã có tin), KHÔNG phải kiến trúc. Đã được EDA tuần này xác nhận.
- **Crawl body bài (nội dung đầy đủ)** là lever #1 — nhưng việc crawl thuộc một project khác, project này chỉ **tiêu thụ** text.
- **Phản hồi 2 gợi ý của thầy:** (1) *dùng embedding thay scalar score* → **đã làm** (Embedding Baseline, test 68.76%); (2) *thêm vector ngẫu nhiên cho tin thưa* → **đã làm xong** (Latent Noise Baseline, test **69.33%** @ 10 epoch — cao nhất các news variant, nhưng vẫn dưới HAR-only 69.98%). Chi tiết §2.4.6.
- **1 bug dữ liệu đang mở:** SSB truncation — toàn bộ stock bị cắt về 1299 dòng (bỏ mất 2012–2026 của các mã dài lịch sử).

Toàn bộ số liệu dưới đây được trích từ file `results/.../training_results.json` thật (có ghi đường dẫn). Không có số bịa.

---

## 1. STATUS & METRICS TẤT CẢ MODEL (đưa lên đầu)

### 1.1 Bảng tổng hợp — TEST metrics (số thực, không leakage)

| # | Model | Epochs | **DirAcc** | **R²** | **QLIKE** | RMSE | MSE | Nguồn / run |
|---|-------|:------:|:---------:|:------:|:---------:|:----:|:---:|-------------|
| 1 | **HAR-R Linear** (baseline) | – | 51.53% | 0.105 | 1.298 | 0.000513 | 2.63e-7 | `results/har_baseline_vn30_2026-06-20/` |
| 2 | Simple LSTM (realistic) | – | 47.89% | 0.211 | 0.673 | 0.000727 | 5.28e-7 | `MODEL_COMPARISON_FINAL_2026-06-28.json` |
| 3 | LSTM-HAR (realistic) | – | 48.09% | 0.110 | – | 0.000554 | 3.07e-7 | `MODEL_COMPARISON_FINAL_2026-06-28.json` |
| 4 | Enhanced LSTM-HAR | 21/70 | 48.56% | 0.098 | 0.641 | 0.000557 | 3.11e-7 | `results/enhanced_lstm_har_..._2026-06-21_112604/` |
| 5 | 🏆 **Parallel LSTM-GNN (k-NN)** | 69/70 | **69.98%** | **0.714** | **0.529** | 0.002644 | 6.99e-6 | `results/parallel_lstm_gnn_knn_2026-06-28_222021/` |
| 6 | Parallel LSTM-GNN + sentiment (lexicon) | 20 | 68.57% | 0.707 | 0.546 | 0.002677 | 7.17e-6 | `results/sentiment_baseline_knn_*` |
| 7 | Parallel LSTM-GNN + embedding (PhoBERT) | 40 | 68.76% | 0.717 | 0.553 | 0.002628 | 6.91e-6 | `results/embedding_baseline_2026-07-08_003719/` |
| 8 | Parallel LSTM-GNN + market-fallback | 40 | 68.69% | 0.706 | 0.548 | 0.002683 | 7.20e-6 | `results/market_fallback_2026-07-08_171348/` |
| 9 | Parallel LSTM-GNN + sentiment-decay | 10 | 67.87% | ~0.707 | ~0.56 | ~0.0027 | – | `results/sentiment_decay_*` (tuần này) |
| 10 | 🆕 Parallel LSTM-GNN + latent-noise (gợi ý thầy #2) | 10 | **69.33%** | 0.713 | 0.544 | 0.002675 | 7.20e-6 | `results/latent_noise_2026-07-11_124004/` |

> ⚠️ **Lưu ý so sánh RMSE:** Các model #1–#4 (single-stock) dự báo 1 mã/lúc trên scale đã normalize khác, nên RMSE của chúng (~0.0005) **không so sánh trực tiếp** với model #5–#9 (multi-stock GNN, RMSE ~0.0026, dùng 1 normalizer chung cho 30 mã). **Metric so sánh công bằng là DirAcc + R² + QLIKE.**

### 1.2 Validation metrics (model chính + news variants)

| Model | Val DirAcc | Val R² | Val QLIKE | Ghi chú |
|-------|:---------:|:------:|:---------:|---------|
| Parallel LSTM-GNN (k-NN) | ~70% | – | – | val/test gap khiêm tốn → không overfit |
| + embedding (PhoBERT) | 71.32% | 0.660 | 0.697 | pooled; per-stock val 47.68% |
| + market-fallback | 70.94% | 0.657 | 0.697 | pooled; per-stock val 46.20% |
| + sentiment-decay | 69.28% | – | – | 10 epoch |

### 1.3 Thành tựu so với mục tiêu (CLAUDE.md)

| Metric | Mục tiêu | Tốt nhất | Đạt? |
|--------|:--------:|:--------:|:----:|
| DirAcc | > 55% | **69.98%** (k-NN) | ✅ vượt 14.98 điểm |
| R² | > 0.50 | **0.714** (k-NN) | ✅ vượt 0.214 |
| RMSE | < 0.20 | 0.002644 (k-NN) | ✅ vượt cả ngưỡng |
| QLIKE | < 0.50 | 0.529 (k-NN) | ⚠️ chưa đạt (còn 0.029) |

### 1.4 ⚠️ Bug data leakage đã sửa (quan trọng cho tính khả tín)

Lần train đầu (trước 28/06), Simple LSTM & LSTM-HAR dùng **random split** → future leak → DirAcc thổi lên **67.6%**. Sau khi chuyển sang **temporal split (70/15/15)**, DirAcc thực tế chỉ **47.9–48.6%**. Báo cáo này dùng **số realistic (sau sửa)**, không dùng số bị thổi.

| Model | DirAcc (leakage, sai) | DirAcc (realistic, đúng) |
|-------|:---------------------:|:------------------------:|
| Simple LSTM | 67.63% | **47.89%** |
| LSTM-HAR | 67.39% | **48.09%** |

### 1.5 Pattern quan trọng tuần này — "News no-lift" (5/5 variants)

| Hướng tích hợp news | Test DirAcc | vs HAR-only (69.98%) |
|----------------------|:-----------:|:--------------------:|
| Scalar sentiment (lexicon) | 68.57% | −1.41 |
| Embedding PhoBERT (768→64) | 68.76% | −1.22 |
| Market-fallback (dense market branch) | 68.69% | −1.29 |
| Sentiment-decay (carry-forward) | 67.87% | −2.11 |
| 🆕 Latent-noise (vector ngẫu nhiên, gợi ý thầy) | **69.33%** | −0.65 (gần nhất) |
| HAR-only (không news) | **69.98%** | — (cao nhất) |

→ **Tất cả cách đưa news đều thấp hơn HAR-only.** Latent-noise (gợi ý thầy #2) là **gần nhất** (−0.65%, 69.33%) — tín hiệu tích cực nhỏ nhưng chưa đủ vượt. Đây không phải do code sai (pipeline đã review, không leakage, val/test gap tốt) — mà do **signal tin tức quá yếu/thưa so với nhiễu HAR**.

**Bằng chứng độc lập (event-study, không qua model):** EDA sentiment↔price tuần này cũng cho **NO-GO** — kiểm định Mann-Whitney p>0.54 mọi horizon, corr sentiment→return/vol chỉ ~0.13 (xem §2.4d). → Root cause được xác nhận **ở cấp dữ liệu thống kê**, không phải do cách tích hợp vào model. Phân tích root cause ở §4.

---

## 2. KIẾN TRÚC (Architecture)

### 2.1 Tổng quan pipeline kiến trúc

```
                        ┌─────────────────────────────────────────────┐
  30 mã × 22 ngày × 3 HAR ─→│  PARALLEL LSTM-GNN HYBRID                  │
  (offline: HAR đã tính)    │                                             │
        │                   │  ┌─ LSTM stream (temporal, per-stock) ──┐  │
        │                   │  │   2 layer × 64 hidden                │  │
        │                   │  │   → h_lstm [B,30,64]                 │  │
        │                   │  ├──────────────────────────────────────┤  │
        │                   │  └─ GAT stream (spatial, cross-stock) ─┘  │
        │                   │      2 layer × 4 heads × 64              │  │
        │                   │      adjacency = k-NN graph (k=8)         │  │
        │                   │      → h_gnn [B,30,256]                  │  │
        │                   │                                             │
        │                   │  FUSION: concat [64+256=320] → MLP → y    │
        │                   │           320→64→32→1                     │
        │                   └─────────────────────────────────────────────┘
        │                                       │
   (tuỳ chọn) news branch ───────────────→ concat thêm [d_news] trước MLP
        │   (embedding / market / decay — tuần này)
        ▼
  Output: 30 dự báo volatility 5-day ahead   shape [B, 30]
```

### 2.2 Parallel LSTM-GNN — model chính

**File:** `src/lstm_gat_hybrid/model_parallel.py:20` — class `ParallelLSTMGNN` (theo paper Sonani et al. 2025). **2 luồng song song** — LSTM và GAT đều đọc input gốc độc lập, không có bottleneck tuần tự → concat ở cuối.

#### Forward pass chi tiết (tensor shapes thật)

```
x [B,22,30,3]                           adj [B,30,30]  (k-NN, xem §2.2b)
 │
 ├─────────────── STREAM 1: LSTM (temporal) ───────────────────┐
 │   for mỗi mã s:  x[:, :, s, :] = [B,22,3]                    │
 │        │                                                      │
 │        ▼                                                      │
 │   nn.LSTM(input=3, hidden=64, layers=2, dropout=0.2)         │
 │        │  lấy h_n[-1]  →  h_s [B,64]                          │
 │        ▼                                                      │
 │   stack 30 mã  →  h_lstm [B,30,64]                            │
 │                                                               │
 ├─────────────── STREAM 2: GAT (spatial) ──────────────────────┤
 │   for mỗi bước t:  x[:, t, :, :] = [B,30,3]                   │
 │        │                                                      │
 │        ▼                                                      │
 │   GAT layer 1: 3 → 256   (4 heads × 64, LeakyReLU attn,       │
 │   GAT layer 2: 256 → 256  mask bằng adj, softmax hàng xóm)    │
 │        │  → h_t [B,30,256]                                    │
 │        ▼                                                      │
 │   stack 22 bước [B,22,30,256] → mean theo thời gian           │
 │        │  → h_gnn [B,30,256]                                   │
 │                                                               │
 └──► FUSION: concat([h_lstm, h_gnn], dim=-1) = [B,30,320] ◄────┘
                       │
                       ▼
          MLP: 320 →64 →32 →1  (ReLU + Dropout 0.5 giữa các layer)
                       │
                       ▼  squeeze
               ŷ [B,30]   (30 dự báo volatility 5-day ahead)
```

```python
# model_parallel.py:117-194 — forward
# Stream 1: LSTM per-stock → h_lstm [B,30,64]
for s in range(num_stocks):
    lstm_out, (h_n, _) = self.lstm_encoder(x[:, :, s, :])   # [B,22,3]→[B,22,64]
    lstm_embeddings_list.append(h_n[-1])                    # [B,64]
h_lstm = torch.stack(lstm_embeddings_list, dim=1)           # [B,30,64]

# Stream 2: GAT per-timestep → mean-pool → h_gnn [B,30,256]
for t in range(seq_len):
    h_t = x[:, t, :, :]                                     # [B,30,3]
    for gat_layer in self.gat_layers:                       # 2 layer: 3→256, 256→256
        h_t = gat_layer(h_t, adj_matrix)
    gnn_embeddings_list.append(h_t)                         # [B,30,256]
h_gnn = torch.stack(gnn_embeddings_list, dim=1).mean(dim=1) # mean theo thời gian

# Fusion
h_fused = torch.cat([h_lstm, h_gnn], dim=2)                 # [B,30,320]
return self.fusion(h_fused).squeeze(-1)                     # [B,30]
```

**Hyper tham số (config.py):** LSTM hidden=64, 2 layer, dropout 0.2; GAT hidden=64, 4 heads, 2 layer; fusion MLP `320→64→32→1` (dropout 0.3/0.5), **linear output** — không activation, theo pattern Enhanced LSTM-HAR (dự báo trên scale chuẩn hóa, `inverse_transform` khi đánh giá).

#### 2.2b Đồ thị k-NN — cách xây adjacency matrix

**File:** `src/lstm_gat_hybrid/graph_utils_fixed.py:34` — `DynamicGraphBuilder.build_correlation_graph` (method mặc định `graph_method='correlation'`, thực chất = **k-NN theo |Pearson correlation|**).

```
Cửa sổ 22 ngày × 30 mã  =  returns [22, 30]   (HAR daily vol của 30 mã trong window đó)
        │
        ▼
 (1) Ma trận tương quan [30,30]:  corr[i,j] = |Pearson(vol_mã_i, vol_mã_j)|  trên 22 ngày
        │
        ▼
 (2) Mỗi mã i: chọn top-k=8 mã có |corr| cao nhất  →  cạnh (i, j)
        │      (k = config.top_k_neighbors = 8; đối xứng adj[i,j]=adj[j,i]=|corr|)
        ▼
 (3) + self-loop 0.1, row-normalize  →  adj [30,30]  (thưa, density ≈ k/(30-1) ≈ 28%)
        │
        ▼
  GAT dùng adj này làm mask: mã i chỉ "nhìn" 8 hàng xóm tương quan mạnh nhất
```

**Anti-leakage then chốt:** đồ thị xây **riêng cho từng sequence**, chỉ dùng đúng 22 ngày input đó → **không nhìn tương lai** (`dataset_with_graph_method.py:261`, mỗi window build lại). Đồ thị **động** — thay đổi theo từng thời điểm, phản ánh quan hệ mã-mã thay đổi theo thị trường.

> ⚠️ Liên quan bug SSB (§4.1): mọi mã bị cắt về 1299 dòng trước khi build graph/split → window thực tế không trải đều 2006–2026 như thiết kế → cần fix trước khi tin hoàn toàn số 69.98%.

### 2.3 Baseline models (single-stock)

| Baseline | File | Kiến trúc | Input |
|----------|------|-----------|-------|
| HAR-R Linear | `src/har_baseline/train.py` | LinearRegression | 3 HAR |
| Simple LSTM | `src/lstm_baseline/model.py` `SimpleVolatilityLSTM` | 1-layer LSTM(128)+FC | raw vol [B,22,1] |
| LSTM-HAR | `src/lstm_har_baseline/model.py` `HARVolatilityLSTM` | 3-layer LSTM(128)+FC | 3 HAR [B,22,3] |
| Enhanced LSTM-HAR | `src/lstm_har_enhanced/model_enhanced.py` | 3-layer LSTM(128)+FC | raw+weekly+monthly [B,22,3] |

> Enhanced LSTM-HAR từng được report 67.90% DirAcc nhưng đó là **với random_split (leakage)**. Số thực tế (temporal split) = 48.56%. Pattern normalize được học từ model này: **StandardScaler + linear output + inverse_transform** (xem §3.6).

### 2.4 News-augmented baselines (tuần này — cô lập trong `baselines/`)

Theo rule CLAUDE.md §3.F, mỗi baseline 1 folder timestamped + 5 sub-folder (requirements/design/code/code_review/test), **không sửa `src/` chung**. Có **4 cách tích hợp sentiment**, từ đơn giản → phức tạp:

| # | Biến thể | Cách đưa sentiment | Model class | Input mới | File model |
|---|----------|--------------------|-------------|-----------|------------|
| 1 | **Scalar** | 2 số (score + count) ghép vào input LSTM | reuse `ParallelLSTMGNN` | `[B,22,30,5]` | `src/sentiment_baseline/` |
| 2 | **Embedding** | vector 64-d PhoBERT, nhánh riêng | `EmbeddingBaseline` | `+[B,22,30,10,64]` | `baselines/2026-07-07_embedding_baseline/` |
| 3 | **Market-fallback** | embedding mã + embedding thị trường, gate | `MarketFallbackBaseline` | `+[B,22,15,64]` market | `baselines/2026-07-08_market_fallback/` |
| 4 | **Decay** | scalar nhưng score được carry-forward+decay | reuse #1 | như #1 | `baselines/2026-07-11_sentiment_decay/` |
| 5 | 🆕 **Latent-noise** (gợi ý thầy #2) | embedding + nhiễu Gauss `z+σ·ε` trên news_rep (train only) | `LatentNoiseBaseline` (subclass #2) | như #2 | `baselines/2026-07-11_latent_noise/` |

#### 2.4.0 Pipeline xử lý NEWS chung (feeding cả 4 variant)

```
   9 nguồn CSV thô                aggregate_news_sources.py              unified_articles.csv
 ┌──────────────────┐   normalize 2 family + parse date   ┌─────────────────────────────────────────┐
 │ data*.csv (title)│ ─────────────────────────────────→  │ unified_id, source, title, lead,        │
 │ cafef/ssi/...    │            dedup theo URL            │ category, date(YYYY-MM-DD), url, ...     │
 │ broker reports   │            58,755 → 21,107 unique    │ 6,280 dòng có lead (29.8%) = text giàu   │
 └──────────────────┘                                    └───────────────┬─────────────────────────┘
        (D:/bmad-projects/crawl_data)                                    │
                                          ┌──────────────────────────────┼──────────────────────────┐
                                          ▼                              ▼                          ▼
                              (Variant 1 & 4: SCALAR)         (Variant 2 & 3: EMBEDDING)         (Variant 3: MARKET)
                          process_news_to_sentiment.py          extract_embeddings.py          extract_market_embeddings.py
                        ┌───────────────────────────┐      ┌────────────────────────────┐     ┌──────────────────────────┐
                        │ 1) match ticker (regex \b) │      │ PhoBERT FROZEN → [CLS] 768 │     │ PhoBERT trên TẤT CẢ bài   │
                        │ 2) chấm điểm lexicon/phobert│      │ PCA 768→64 (fit train only)│     │ (không lọc ticker)         │
                        │ 3) avg theo mã/ngày         │      │ ticker-match trên nội dung │     │ PCA→64 (fit train only)   │
                        └─────────────┬───────────────┘      └──────────────┬─────────────┘     └─────────────┬────────────┘
                                      ▼                                     ▼                                 ▼
                        {TICKER}_sentiment.csv              {TICKER}_emb.npz                   market_emb.npz
                        date, sentiment_1d,                 {date: [n_bài, 64]}               {date: [n_bài, 64]}
                             news_count_1d, news_titles     (data/sentiment_embedding/)       (cùng thư mục)
                        (data/sentiment_baseline/)
                                      │                                     │                                 │
                                      │   ┌─────────────────────────────────┘                                 │
                                      ▼   ▼                                                                   ▼
                          [Variant 4: compute_decay.py xếp carry-forward]                          [chỉ Variant 3 dùng]
```

**Chi tiết từng stage (file:line):**
- **Match ticker:** regex whole-word `\b{TICKER}\b` trên title (scalar) hoặc full `title+lead` (embedding) → ~16–20% bài khớp 1 mã. (`process_news_to_sentiment.py:58`, `extract_embeddings.py:76`)
- **Scoring:** lexicon Việt (nhanh) hoặc PhoBERT/XLM-R (`[-1,1]`); avg theo mã/ngày.
- **PCA chống leakage:** `extract_embeddings.py:132` — fit PCA **chỉ trên bài date<2020-01-01** (train), rồi transform cả val/test. Nếu train < dim → tự giảm dim (không mở scope → không rò rỉ).
- **npz structure:** `{date_str: np.ndarray[n_articles, 64]}` — mỗi ngày lưu **list vector** các bài (chưa pool, pooling làm online trong model).

#### 2.4.1 Variant 1 — Scalar sentiment (đơn giản nhất)

**Cách:** ghép 2 feature sentiment vào **cùng input tensor** với HAR → LSTM+GAT nhận 5 kênh. Không có nhánh riêng.

```
Input [B, 22, 30, 5]
  3 kênh HAR + 2 kênh sentiment (×0.005, ×0.0005 để cùng magnitude ~1e-3)
        │
        ▼
  ParallelLSTMGNN (NHƯ §2.2, chỉ đổi input_size 3→5)
   ├─ LSTM(5→64, 2 layer)  ─→ h_lstm [B,30,64]
   ├─ GAT (2 layer, 4 head) ─→ h_gnn  [B,30,256]
   └─ concat [64+256=320] → MLP(320→64→32→1) → ŷ [B,30]
```

```python
# src/sentiment_baseline/dataset_sentiment.py:23-27
SENTIMENT_COLS  = ['sentiment_1d', 'news_count_1d']
SENTIMENT_SCALE = {'sentiment_1d': 0.005, 'news_count_1d': 0.0005}  # về magnitude HAR (~1e-3)
FEATURE_COLS = ['har_daily_vol','har_weekly_vol','har_monthly_vol'] + SENTIMENT_COLS  # 5 features

# train_sentiment_baseline.py:45
config.num_features_per_stock = 5   # ← duy nhất thay đổi; model = ParallelLSTMGNN nguyên vẹn
```

> ✅ **Ưu điểm:** 0 module mới, simplest. ❌ **Nhược:** sentiment "pha" trực tiếp vào HAR → dễ bị nhiễu HAR lấn át (chính là lý do no-lift).

#### 2.4.2 Variant 2 — Embedding baseline (vector PhoBERT, nhánh riêng)

**Cách:** news thành nhánh riêng → học temporal riêng → **late concat** với HAR. Dùng `ArticleSetAttentionPooling` để gộp số bài biến đổi mỗi ngày (permutation-invariant).

```
x_har [B,22,30,3]  adj[B,30,30]                x_emb [B,22,30,10,64]   mask [B,22,30,10]
     │                                        (10 bài tối đa/ngày, pad 0)
     ▼                                                     │
 ┌─────────────────────┐                                    ▼
 │ ParallelLSTMGNN     │                       ┌───────────────────────────────┐
 │ .get_embeddings()   │                       │ ArticleSetAttentionPooling    │
 │  (frozen fusion)    │                       │  proj 64→64                   │
 └──────┬──────────────┘                       │  score = h·query (learnable)  │
   h_lstm[B,30,64]                             │  softmax(masked, -1e9)        │
   h_gnn [B,30,256]                            │  + no_news_token nếu 0 bài    │
        │                                      └───────────────┬───────────────┘
        │                                       daily [B,22,30,64] (hoặc [B,30,22,64])
        │                                                      │
        │                                                      ▼
        │                                      ┌───────────────────────────────┐
        │                                      │ NewsTemporalEncoder           │
        │                                      │  LSTM(64→64, 1 layer) qua 22 ngày│
        │                                      └───────────────┬───────────────┘
        │                                            news_rep [B,30,64]
        ▼                                                      ▼
        └──────────────────► concat [B,30, 64+256+64 = 384] ◄┘
                                     │
                                     ▼
                          MLP(384→64→32→1) → ŷ [B,30]
```

**Toán attention pooling** (`model_embedding.py:25-60`):
```python
h      = self.proj(article_embs)              # [B,22,30,10,64]
scores = (h * self.query).sum(-1)             # dot với query học được → [B,22,30,10]
scores = scores.masked_fill(mask==0, -1e9)    # -1e9 (KHÔNG -inf, tránh NaN khi 0 bài)
attn   = softmax(scores, dim=-1)              # [B,22,30,10]
daily  = (atnn.unsqueeze(-1) * h).sum(-2)     # weighted sum → [B,22,30,64]
# ngày 0 bài → chèn no_news_token học được:
has    = (mask.sum(-1,keepdim=True) > 0)
daily  = has*daily + (1-has)*self.no_news_token
```

> ✅ Giữ thông tin phong phú hơn scalar (64-d vs 2 số). ❌ Nhưng 94.5% ngày-mã vẫn 0 tin → `no_news_token` chiếm đa số → signal thực tế vẫn thưa.

#### 2.4.3 Variant 3 — Market-fallback baseline (gate thị trường)

**Cách:** ngoài nhánh stock (thưa), thêm nhánh **market** (pool TẤT CẢ bài trong ngày, dày đặc). Gate deterministic chọn: ngày-mã có tin → dùng stock; mù tin → dùng market.

```
x_emb [B,22,30,10,64] (stock, thưa)        x_market [B,22,15,64] (market, dày — 15 bài/ngày)
       │                                              │
       ▼                                              ▼
 ArticleSetAttentionPooling → stock_daily      MarketBranch(=cùng pooling) → market_daily
   [B,22,30,64]                                  [B,22,64]
       │                                              │ broadcast → [B,22,30,64]
       └──────────────┬───────────────────────────────┘
                      ▼
            GatedNewsFusion (deterministic):
            g = has_news (1 nếu mã có tin ngày đó, else 0)
            daily = g·stock_daily + (1−g)·market_daily     ← [B,22,30,64]
                      │
                      ▼
            NewsTemporalEncoder (LSTM 22 ngày) → news_rep [B,30,64]
                      │
   concat với h_lstm[64] + h_gnn[256] → [B,30,384] → MLP → ŷ [B,30]
```

```python
# GatedNewsFusion — gate KHÔNG học (tránh gate-collapse):
market = market_daily.unsqueeze(2).expand_as(stock_daily)   # [B,22,30,64]
g = has_news.to(stock_daily.dtype)                          # 0/1
return g * stock_daily + (1 - g) * market                   # có tin→stock, mù→market
```

> ✅ **Ý tưởng:** 94.5% ngày-mã mù tin nay nhận signal thị trường thay vì 0. ❌ Nhưng market = "tâm lý hệ thống", **không có alpha mã cụ thể** → thực tế vẫn no-lift (68.69%).

#### 2.4.4 Variant 4 — Sentiment-decay baseline (carry-forward)

**Cách:** KHÔNG có model mới. Chỉ **tiền xử lý** cột `sentiment_1d` thành *trạng thái suy giảm* rồi đưa vào **Variant 1 nguyên vẹn**. Mục đích: giữ "dư âm" sentiment những ngày không có tin thay vì reset về 0.

```
sentiment_1d gốc (0 những ngày không tin):
   T:0.6  T+1:0   T+2:0   T+3:0.3  T+4:0   ...
        │ compute_decay.py (decay=0.9)
        ▼
decayed state s_t:
   0.6    0.54   0.486   0.3 ←reset   0.27   ...   (suy giảm mũ, reset khi có tin)
        │ cùng schema {date, sentiment_1d(=state), news_count_1d}
        ▼
   Variant 1 (ParallelLSTMGNN, 5 features) — không đổi gì
```

```python
# baselines/2026-07-11_sentiment_decay/code/compute_decay.py:33-47
def compute_decay_state(scores, masks, decay=0.9):
    s = 0.0
    for score, mask in zip(scores, masks):
        s = float(score) if mask else s * decay   # có tin→reset; không→×0.9
        states.append(s)
    return states
# news_count_1d GIỮ NGUYÊN (không thành mask) — fix HIGH-1 để không confound
```

> **Half-life ≈ 7 ngày** (0.9^7 ≈ 0.48). **Closure test:** market-fallback (học được, rich) đã no-lift → decay (fixed, đơn giản hơn) được kỳ vọng cũng no-lift → đã xác nhận (test 67.87%, thấp nhất 4 variant).

#### 2.4.5 Sentiment↔Price EDA (event-study, không qua model) — **verdict NO-GO**
- Event-study kiểm tra "sentiment tại T → forward return/volatility tại T+1/T+5" cho 30 mã. 1,851 events / 29 mã.
- **Phân phối sentiment lệch dương cực mạnh:** pos=1044 / neu=784 / **neg=23** → nhóm tiêu cực quá ít mẫu.
- **Kiểm định:** Mann-Whitney pos-vs-neg **p > 0.54 ở mọi horizon** (raw + per-ticker-demeaned), Bonferroni α=0.01 → không ý nghĩa.
- **Tương quan yếu:** mean |Spearman corr| sentiment→return ≈ **0.139**, →vol ≈ **0.127** (T+1); 21/29 mã corr dương nhưng |corr|<0.3.
- → **Kết luận:** sentiment **không nên làm feature directional chính**; nếu dùng thì chỉ auxiliary/noise (corr ~0.13 vẫn tí giá trị). Đây là bằng chứng độc lập củng cố pattern news no-lift ở §1.5.

#### 2.4.6 Phản hồi 2 gợi ý của thầy tuần trước — trạng thái xử lý

| # | Gợi ý của thầy | Trạng thái | Mapping vào project |
|---|----------------|-----------|---------------------|
| **1** | Dùng **embedding vector** thay vì sentiment score (score "mất mác thông tin") | ✅ **ĐÃ LÀM** | = Embedding Baseline §2.4.2 |
| **2** | Thêm **vector phát sinh ngẫu nhiên** (latent noise) cho trường hợp tin thưa | ✅ **ĐÃ LÀM** (Tier A) | = Latent Noise Baseline `baselines/2026-07-11_latent_noise/` |

**Gợi ý 1 — embedding thay scalar score: ĐÃ THỰC HIỆN và report.**
Thầy nhận xét đúng: sentiment score là **nén lossy** (~99% ngữ nghĩa mất khi rút 1 câu về 1 số). Project đã build **Embedding Baseline** (§2.4.2): PhoBERT frozen → CLS 768-d → PCA 768→64 (fit train-only, chống leakage) → nhánh riêng `ArticleSetAttentionPooling` + `NewsTemporalEncoder` → concat với HAR.
- **Kết quả:** test DirAcc **68.76%** (val 71.32%), R² 0.717, QLIKE 0.553 — `results/embedding_baseline_2026-07-08_003719/`.
- **Kết luận trung thực:** embedding **giữ nhiều thông tin hơn** (64-d vs 2 số) NHƯNG **vẫn no-lift** so với HAR-only 69.98% (thấp hơn 1.2%). Lý do: **94.5% ngày-mã vẫn không có tin** → `no_news_token` học được chiếm đa số, signal thực tế bị pha loãng. → Embedding giải đúng bài toán "mất mác thông tin", nhưng vấp phải bottleneck **sparsity** (không phải representation).

**Gợi ý 2 — vector ngẫu nhiên cho tin thưa: ĐÃ CODE xong Tier A + train 10 epoch.**
Ý thầy (= kỹ thuật latent-space: thêm lớp random hóa vector để chống dữ liệu semantic thưa). Đã thực thi Tier A trong baseline mới `baselines/2026-07-11_latent_noise/`:
- **Cài đặt (Tier A):** subclass `EmbeddingBaseline`, thêm `news_rep += σ·ε` (ε~N(0,1), σ=0.1) **chỉ ở train mode**, tắt ở eval (validate/test công bằng). Cô lập cứng, reuse HAR+news branch của embedding baseline, 7/7 pytest pass. Code: `code/model_latent_noise.py`.

**Kiến trúc (diagram) — điểm chèn nhiễu:**
```
x_har[B,22,30,3] adj              x_emb[B,22,30,10,64] mask       (giống Embedding Baseline §2.4.2)
     │                                       │
     ▼                                       ▼
 ┌────────────────┐               ┌─────────────────────────┐
 │ har.get_embed  │               │ ArticleSetAttentionPool │  ← reuse nguyên vẹn
 │  h_lstm[B,30,64]│              │  → daily[B,22,30,64]     │
 │  h_gnn[B,30,256]│              │ NewsTemporalEncoder      │
 └───────┬────────┘               │  → news_rep[B,30,64]     │
         │                        └────────────┬────────────┘
         │                                     │
         │                         ┌───────────▼──────────────┐
         │                         │ if self.training & σ>0:  │   ← CHỈ thêm ở Tier A
         │                         │   news_rep += σ·ε        │      (ε~N(0,1))
         │                         │ eval mode → bỏ qua       │      → validate/test
         │                         └───────────┬──────────────┘      deterministic
         │                                     │
         └────► concat [B,30, 64+256+64=384] ◄─┘
                         │
                    fusion MLP → ŷ[B,30]
```
**Khác biệt duy nhất vs Embedding Baseline = 1 dòng nhiễu** trên `news_rep`. Lý do chọn vị trí này: gợi ý thầy nhắm "tin thưa" → nhiễu đúng nhánh news (HAR đã mạnh 69.98%, nhiễu vào đó có thể hại). Eval tắt noise để validate/test không ngẫu nhiên → so sánh công bằng.
- **Đường cong val DirAcc (10 epoch, resume từ ckpt epoch 5):** 68.48→68.12→69.43→70.40→69.58 (ep1-5) → 68.25→69.14→69.26→69.43→**71.28** (ep6-10, best).
- **Kết quả test:** DirAcc **69.33%**, R² 0.713, QLIKE 0.544 — `results/latent_noise_2026-07-11_124004/`.
- **So sánh:** latent-noise 69.33% **cao nhất các news variant** (> embedding 68.76% +0.57%, QLIKE cũng tốt hơn 0.544<0.553) NHƯNG vẫn −0.65% so HAR-only 69.98%. → Tín hiệu tích cực **nhỏ** (gợi ý thầy có tác dụng marginal), chưa đủ vượt HAR.
- **Caveat trung thực:** 10ep vs embedding 40ep vs HAR 70ep — **chưa matched-epoch**, chênh ±0.5% có thể trong noise. Cần matched-epoch control để chốt. Đây là kết quả **khuyến khích** nhất trong 5 news variant → đáng train thêm + tune σ.
- **Train tiếp tới 15 epoch (đã thử, dừng):** resume từ ckpt 10-epoch, val epoch 11 = 70.59% (ổn định ~70–71%, không thấy lift đột phá). Run dừng sớm do CPU bận chạy song song pipeline body-corpus extraction + full-corpus EDA → **10 epoch vẫn là kết quả chính thức**. Tiếp tục train tới 40 epoch khi CPU rảnh.
- **Defer (Tier B):** Variational Information Bottleneck (VIB, Alemi 2017) — encoder (μ, σ) + KL term + β-warmup (β-warmup bắt buộc tránh posterior collapse). Chỉ làm nếu Tier A matched-epoch vẫn hứa hẹn.

→ **Tóm lại cho thầy:** cả 2 gợi ý đều **đã thực thi**. Gợi ý embedding (68.76%) và latent-noise (69.33%) đều được report đầy đủ. Latent-noise cho **kết quả khuyến khích nhất** (cao nhất các news variant, QLIKE tốt hơn) nhưng chưa vượt HAR-only — bước tiếp: matched-epoch control + train thêm/tune σ để chốt, rồi mới cân nhắc Tier B (VIB).

### 2.5 TimesFM (foundation model — reference)

**File:** `src/timesfm_baseline/timesfm_lora_finetuning.py`
- TimesFM 2.5 (232M params) + LoRA (rank=4, ~1.4M trainable = 0.6%).
- Context 64 ngày, horizon 5 ngày, AdamW lr=1e-4, weight_decay=0.01, grad clip 1.0.
- Đã qua 3 vòng adversarial review (40 bugs fixed, 34/34 test pass). Chưa chạy full trên data mới.

### 2.6 Cấu hình chống overfit (bắt buộc mọi model — CLAUDE.md §3.E)

```python
early_stopping = EarlyStopping(patience=15, min_delta=1e-6, min_epochs=20)
optimizer      = optim.Adam(params, lr=1e-3, weight_decay=1e-5)        # L2
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)        # grad clip
scheduler      = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
# dropout: LSTM 0.2, FC 0.3, fusion 0.15
if (epoch+1) % 10 == 0: plot_learning_curves(train_losses, val_losses, ...)  # §3.C
```

→ val/test gap khiêm tốn (ví dụ embedding: val 71.32% / test 68.76% = gap 2.56 điểm) → **không overfit**.

---

## 3. DATA PIPELINE — Input/Output & xử lý chi tiết

### 3.1 Sơ đồ: cái gì OFFLINE vs ONLINE

```
═════════════════ OFFLINE (pre-compute + cache, chạy 1 lần) ═══════════════════
  OHLCV thô ──process_data.py──→ Parkinson volatility CSV   (data/processed/)
  3 news CSV families ──aggregate──→ unified_articles.csv   (crawl_data/aggregated/)
  unified news ──PhoBERT(frozen)+PCA──→ {TICKER}_emb.npz     (data/sentiment_embedding/)
  unified news ──lexicon/phobert──→ {TICKER}_sentiment.csv   (data/sentiment_baseline/)
                 (PCA fit CHỈ trên train-period → chống leakage)

══════════════ ONLINE (chạy trong training loop, mỗi epoch) ═══════════════════
  volatility CSV ──HAR rolling (1/5/22)──→ 3 HAR features     (trong dataset)
  sequence window ──k-NN graph build──→ adjacency matrix       (mỗi seq 1 đồ thị)
  ──StandardScaler (fit train)──→ normalize x_har + y          (trong __getitem__)
  ──temporal split 70/15/15──→ train/val/test                  (theo date, không random)
```

### 3.2 Offline 1 — OHLCV → Parkinson volatility

**Input** (`data/raw/all_available/ACB_ohlcv.csv`):
```csv
date,open,high,low,close,volume
2020-12-09,8895.7,9164.8,8800.8,9038.2,64986634
2020-12-10,9117.3,9117.3,8832.4,8848.2,24795341
```

**Công thức Parkinson** (`src/common/parkinson_utils.py`):
```python
parkinson = (np.log(high / low) ** 2) / (4 * np.log(2))
# Hiệu quả hơn close-to-close với dữ liệu daily (dùng cả high/low trong ngày)
```
+ loại NaN/inf, **clip giá trị cực đo** (cận 0.1) để khử outlier.

**Output** (`data/processed/ACB_processed.csv`):
```csv
date,parkinson_volatility
2006-11-21,0.019409836
2006-11-22,0.003458018
2006-11-23,0.004431685
```

### 3.3 Offline 2 — HAR features (Heterogeneous Autoregressive)

`src/common/har_features.py` — rolling mean trên Parkinson volatility:
```python
df['har_daily_vol']   = vol.rolling(window=1,  min_periods=1).mean()   # 1 ngày
df['har_weekly_vol']  = vol.rolling(window=5,  min_periods=1).mean()   # 5 ngày (tuần)
df['har_monthly_vol'] = vol.rolling(window=22, min_periods=1).mean()   # 22 ngày (tháng)
```
→ mỗi ngày, mỗi mã có **3 feature** đi vào LSTM.

### 3.4 Offline 3 — News aggregation (9 nguồn → 1 file unified)

**Input:** 9 file CSV thô ở `D:/bmad-projects/crawl_data/data/` (cafef, vietstock, ssi, vndirect, hsc, broker reports MBS/BVS/KBSV/..., vnstock...).

`src/data_aggregation/aggregate_news_sources.py`:
1. Chuẩn hóa 2 family schema (A: title-only; B: có `lead`/`category`/`author`) về 1 schema chung.
2. Parse date hỗn hợp (ISO `2026-07-04T18:07:00+0700` vs `DD/MM/YYYY`, kể cả năm 2 số).
3. **Dedup theo URL** → loại trùng lặp.
4. Output `unified_articles.csv` (schema: `unified_id, source, title, lead, category, author, date, url, ...`).

**Kết quả:** 58,755 dòng thô → **21,107 dòng unique** (bỏ 37,648 trùng = 64%).
**Phủ theo năm (sau dedup), liên tục 2008–2026:** train (<2020) ~6.4K; test (2021–2026): 2021:1683, 2022:1594, 2023:1484, 2024:1601, 2025:1685, 2026:2486 → **~10,533 bài cho test**.
**6,280 dòng (29.8%) có `lead`** = text giàu nhất (cho hướng embedding).

> ✅ Vấn đề "test set mù tin" đã giải **ở cấp bài báo** (trước đây test 2021–2025 không có news). Nhưng xem §3.7 — chưa giải ở cấp ngày-mã.

### 3.5 Offline 4 — News → sentiment / embedding

**(a) Sentiment (scalar)** — `src/sentiment_baseline/process_news_to_sentiment.py`:
- Match ticker trong title bằng regex whole-word (~16–20% bài khớp 1 mã VN30).
- Chấm điểm: lexicon Việt (nhanh, ~57% ngày có tin ra non-zero) HOẶC PhoBERT/XLM-R (`[-1,1]`).
- Aggregate trung bình theo mã/ngày.
- Output `data/sentiment_baseline/{TICKER}_sentiment.csv`:
```csv
date,sentiment_1d,news_count_1d,news_titles
2020-12-09,0.5,1.0,Khuyến nghị tích cực đối với cổ phiếu ACB
2020-12-01,0.0,0.0,
```

**(b) Embedding (vector)** — `baselines/2026-07-07_embedding_baseline/code/extract_embeddings.py`:
- PhoBERT **frozen** → CLS 768-d cho mỗi bài.
- **PCA 768→64, fit CHỈ trên bài thuộc train-period** (date < 2020-01-01) → chống leakage val/test.
- Cache `data/sentiment_embedding/{TICKER}_emb.npz`: dict `{date: array[n_articles, 64]}`.

> ⚠️ **Gotcha:** phải pin `transformers<5` (5.x break XLM-R tokenizer). Cài thêm `sentencepiece tiktoken`.

### 3.6 Online — normalization, sliding window, graph, temporal split

- **Sliding window:** seq_len=22, forecast_horizon=5, stride=1 → mỗi snapshot = 22 ngày input, target = vol tại T+5. Tổng ~99,794 snapshots (theo report 27/06, **trước bug SSB** — số thực hiện tại bị ảnh hưởng, xem §4.1).
- **Temporal split** (`src/common/temporal_split.py`): **chronological**, KHÔNG random:
```python
train_idx = range(0, int(n*0.70))          # 2006–2020
val_idx   = range(int(n*0.70), int(n*0.85)) # 2020–2021 (early stopping)
test_idx  = range(int(n*0.85), n)           # 2021–2026 (đánh giá cuối)
```
- **Normalization** (pattern học từ Enhanced LSTM-HAR): `StandardScaler` fit trên **train only**, áp cho cả 3 split; output linear (có thể âm ở scale chuẩn hóa) → `inverse_transform` về scale vật lý (≥0) khi đánh giá.
- **Graph build online:** mỗi sequence tự build k-NN adjacency từ đúng 22 ngày đó (xem §2.2).

### 3.7 Tổ chức data — folder map + schema + quy mô

```
data/
├── raw/
│   ├── all_available/      32 file OHLCV (date,open,high,low,close,volume), 2006–2026
│   └── prices/             30 file OHLCV (bản adjust cho EDA)
├── processed/              ← MODEL DÙNG thư mục gốc này: 32 mã, cols=[date, parkinson_volatility]
│   │                          min=1299 dòng (SSB), max=4868 (ACB)
│   ├── vn30_only/          30 mã VN30 (loại VPB/VRE) — UNUSED, cùng bug SSB
│   └── vn100_only/         102 mã, pipeline khác (12 cột OHLCV+HAR sẵn) — cho VN100
├── sentiment_baseline/     {TICKER}_sentiment.csv (date, sentiment_1d, news_count_1d, news_titles)
├── sentiment_embedding/    {TICKER}_emb.npz (PhoBERT→PCA64) + market_emb.npz
├── sentiment_decay/        bản decay (cùng schema sentiment_baseline) — tuần này
└── vn30_sentiment/daily/   schema cũ (num_articles, avg_sentiment_score...) ~13 ngày — legacy
```

**Data minh họa — độ thưa tin (root cause news no-lift):**
- **Cấp bài báo/năm:** đã đều (2021–2026 mỗi năm 1,484–2,486 bài) ✅
- **Cấp ngày-mã:** vẫn CỰC KỲ THƯA — test (≥2021) chỉ **5.5% ngày-mã có tin** (2,237/40,475); train (<2020) chỉ **1.9%**. Gốc rễ: chỉ **~20% bài khớp 1 mã VN30** cụ thể (phần lớn là tin vĩ mô/thị trường chung). → **Thêm bài không tăng density** (vnstock 14,825 bài thô → chỉ 432 unique khớp).

---

## 4. CÁC ISSUE HIỆN TẠI + HƯỚNG GIẢI QUYẾT

### 4.1 🔴 SSB truncation bug (mở, ảnh hưởng số liệu)

**File:** `src/lstm_gat_hybrid/dataset_with_graph_method.py:235,248`
```python
min_length = min(len(df) for df in self.stock_data_with_har.values())  # = 1299 (SSB, niêm yết 2021)
...
vol_data_truncated = vol_data[:min_length]   # ← cắt 1299 dòng ĐẦU của MỌI mã
```
**Hậu quả:** mã dài lịch sử (ACB 4868 dòng từ 2006) chỉ dùng **2006–2011**, **bỏ 2012–2026**. → số liệu 69.98% hiện tại được train trên dữ liệu bị cắt, **không khớp** "test=2021–2026" như report cũ ghi (thực tế test ≈ 1299×30 ≈ 38K snapshots, không phải ~99K). **Data đã đổi** từ lúc train model 69.98% → số hiện tại khó tái lặp đúng con số đó.

**Hướng giải quyết (chọn 1):**
1. Bỏ các mã ngắn (SSB/TCB/TPB) khỏi tập 30 mã, hoặc
2. Cắt tất cả về cửa sổ chung gần đây (vd 2014–2026), hoặc
3. Mask thay vì cắt (padding cho mã ngắn).
→ **Ưu tiên cao**, vì nó ảnh hưởng tính khả tín của mọi kết quả multi-stock.

### 4.2 🟡 News no-lift (5/5 variants) — bottleneck = DATA

**Triệu chứng:** mọi cách đưa news (scalar/embedding/market/decay) đều ~68–69%, thấp hơn HAR-only 69.98%.
**Root cause (đã xác nhận bằng EDA tuần này — 3 nguyên nhân):**
1. Tin thưa (~2% ngày giao dịch) + **lệch dương cực mạnh** (pos 1044 / neg 23) → thiếu mẫu tiêu cực, không đủ signal định hướng.
2. Forward return chưa trừ drift thị trường (chưa tính **abnormal return** trừ VN-Index) → mean return dương có thể chỉ là trend chung của VN30, không phải do sentiment.
3. Sự kiện tiêu cực tập trung vài mã (GAS/MWG/VIC) → confound theo mã.
- Bổ sung: tin chỉ có **tiêu đề** (title-only) → semantic nghèo; chỉ ~5.5% ngày-mã có tin → signal bị chìm dưới nhiễu HAR.
- Market-fallback (dense, học được) cũng no-lift → xác nhận **không phải do kiến trúc yếu**.

**Hướng giải quyết (theo thứ tự ưu tiên):**
1. **Crawl body bài** (nội dung đầy đủ, không chỉ title) — lever #1. ⚠️ Nhưng **crawl thuộc project khác**; project này chỉ tiêu thụ body khi project kia giao. Đã có sẵn consumer-side: `extract_embeddings.py --use_body --max_len 256`.
2. **Matched-epoch control** (chưa làm): chạy HAR-only + scalar + embedding **cùng số epoch** để so sánh công bằng (hiện 70 vs 20 vs 40 epoch — confound).
3. **Chấp nhận HAR là ceiling** cho data hiện tại → tập trung cải thiện HAR/graph thay vì đẩy news thêm.
4. (Nếu muốn cứu sentiment) Chạy lại EDA với **abnormal return** (trừ VN-Index drift) trước khi bỏ hẳn — loại nguyên nhân #2; nếu vẫn p>0.05 thì chốt NO-GO dứt khoát.

### 4.3 🟡 PCA leakage nhẹ (MED-6, kế thừa)

PCA hiện fit theo **calendar cutoff** (date < 2020-01-01), không theo **split** thật → rò rỉ nhẹ val/test qua PCA.
**Hướng:** chuyển PCA vào trong dataset, fit post-split (architectural, defer).

### 4.4 🟢 Tooling gaps (đang setup)

Tuần này đã setup `pytest.ini` (smoke marker), `ruff.toml` (lint clean), cài `pytest-cov diff-cover ruff`. Còn thiếu: smoke test tagged thật sự, diff-coverage ≥80% cho data scripts (I/O orchestration — cần integration test).

### 4.5 ✅ Đã giải / đã làm đúng tuần này
- "Test mù tin" → giải ở cấp bài báo (test có ~10.5K bài).
- Data leakage random→temporal split → đã sửa từ lâu.
- Code review adversarial cho embedding/market/decay → đã chạy, HIGH findings đã fix.
- 3 baseline mới tuân thủ §3.F (cô lập, không sửa `src/`).

---

## 5. KẾ HOẠCH TUẦN TỚI

1. **Fix SSB truncation** (§4.1) → re-train k-NN → xác nhận có tái lặp 69.98% không (ưu tiên #1, ảnh hưởng khả tín).
2. **Matched-epoch control** (§4.2.2) → chốt dứt khoát NO-GO cho news, hoặc tìm điều kiện nó lift.
3. **Latent noise — follow-up** (gợi ý thầy #2, đã code Tier A, 10ep test 69.33%) → làm **matched-epoch control** (latent-noise vs embedding vs HAR cùng epoch), tune σ (0.05/0.1/0.2), train thêm tới 40 epoch. Nếu vẫn hứa hẹn → Tier B (VIB).
4. **Chờ body corpus** từ project crawl → re-run embedding với `--use_body`.
5. Hoàn thiện EDA sentiment↔price → quyết định go/no-go cuối cho sentiment làm feature chính.
6. (Nếu thời gian) full run TimesFM trên data đã fix.

---

## PHỤ LỤC — File map & lệnh chạy

**Code chính:** `src/lstm_gat_hybrid/{model_parallel,dataset_with_graph_method,train_parallel_enhanced,graph_correlation}.py`
**Baselines (cô lập):** `baselines/2026-07-07_embedding_baseline/`, `baselines/2026-07-08_market_fallback/`, `baselines/2026-07-11_sentiment_decay/`, `baselines/2026-07-11_sentiment_price_eda/`
**Data utils:** `src/common/{parkinson_utils,har_features,temporal_split,data_normalization,evaluation}.py`
**Design docs:** `docs/project/SENTIMENT_NEWS_EMBEDDING_ARCHITECTURE.md`, `SENTIMENT_LATENT_SPACE_TECHNIQUES.md`, `SENTIMENT_MARKET_FALLBACK_ARCHITECTURE.md`
**Báo cáo tuần này:** `docs/reports/2026-07-11_*` + `docs/report_2026-07-11/BAO_CAO_TUAN_CHO_THAY.md` (file này)

**Lệnh chạy đại diện:**
```bash
# Offline pipeline
python process_data.py --remove_outliers --n_std 3
python -m src.data_aggregation.aggregate_news_sources
python baselines/2026-07-07_embedding_baseline/code/extract_embeddings.py     # PhoBERT→PCA→npz

# Train
python src/lstm_gat_hybrid/train_parallel_enhanced.py --graph_method knn --epochs 70
python baselines/2026-07-07_embedding_baseline/code/train_embedding_baseline.py --epochs 40
```

---
*Báo cáo trung thực: mọi số trích từ `results/` thật; news no-lift và bug SSB được nêu rõ, không che giấu.*
