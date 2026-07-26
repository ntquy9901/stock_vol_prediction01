# Design (Plan) — Dual-Group News Embedding Baseline

**Simplicity Gate:** feature scope = `mode="ewma"` (146 cols), không `mode="full"` (480 cols) — xem
requirements.md §3. **Anti-Abstraction Gate:** vendor thẳng code `data_eda` đã chạy được (PCA/EWMA
math đã proven), KHÔNG viết lại từ đầu — giảm rủi ro bug so với tự re-derive công thức.
**Vi phạm gate nào không:** không vi phạm — vendor code có sẵn = dùng thẳng lib/code sẵn có, đúng
tinh thần Anti-Abstraction (không tự "wrap" khi có code chạy được rồi).

## 1. Data flow (2 giai đoạn)

### Giai đoạn A — Copy + rebuild aggregation (data prep, chạy 1 lần, ONE-OFF script)

```
[COPY — không sửa data_eda]
C:\luanvan\data_eda\data\features\news_emb_articles_*.parquet   (~4.4GB, PhoBERT cache, KHÔNG rebuild)
     → copy nguyên → stock_vol_prediction01/data/external_news_embeddings/raw_cache/

C:\luanvan\data_eda\src\{features,modeling,data,nlp}\*.py  (chỉ phần cần cho aggregation)
     → copy + rút gọn (bỏ eda.common/pure_news/log_processing — không cần) →
       baselines/2026-07-25_.../code/vendor_data_eda/*.py

[REBUILD — chạy trong project này, đọc bản copy ở trên]
vendor_config.py:
  PROJECT_ROOT   = stock_vol_prediction01 root (Path(__file__).resolve().parents[N], KHÔNG hardcode)
  CRAWL_DATA_ROOT = PROJECT_ROOT.parent / "crawl_data" / "data"   (sibling CÓ SẴN, cùng dir mà
                     src/data_aggregation/aggregate_news_sources.py của project này đã dùng)
  FEATURES_DIR    = .../data/external_news_embeddings/raw_cache   (bản copy, KHÔNG phải data_eda)
  VN30_TICKERS    = import từ src.sentiment.data_collection.tickers (đã có sẵn trong project này)
  PRICE_DATA_DIR  = PROJECT_ROOT / "data" / "processed"   (đã có {TICKER}_processed.csv, cột `date`
                     dùng làm trading calendar — thay cho data_eda's OHLCV riêng)

build_dual_group_panel.py (script MỚI, gọi vendor_data_eda.dual_news_features.build_advanced_features):
     → discover_source_files() scan crawl_data (sibling, đọc-only)
     → _get_article_embeddings() match theo url với raw_cache đã copy → 0 cache-miss dự kiến
       (mọi url trong crawl_data hiện tại đã có trong cache tính đến 2026-07-24 21:49)
       => KHÔNG gọi PhoBERT (đúng ràng buộc "không rebuild embedding")
     → PCA (768→32, shared basis 2 group) + EWMA(30d) + topic flags
     → output: data/features/dual_group_news_panel.parquet  (ticker, date, 146 cols ADV_FEATURES_DUAL+EWMA)
```

### Giai đoạn B — Baseline training (giống pattern `2026-07-07_embedding_baseline`)

```
dual_group_news_panel.parquet (ticker, date, 146 cols)
                    │
     dataset_dual_news.py :: MultiStockDatasetWithDualNews (subclass MultiStockDatasetWithPreSplitData)
     __getitem__ trả (x_har[22,32,3], adj[32,32], x_news[22,32,146], y[32])
     — ĐƠN GIẢN HƠN bản gốc: x_news là 1 vector/ngày (đã aggregate), KHÔNG cần pad/mask
       article-set như bản PCA-64 gốc (đó là feature CHƯA aggregate, đây ĐÃ aggregate theo ngày)
                    │
     model_dual_news.py :: DualGroupNewsBaseline
     h_lstm, h_gnn = ParallelLSTMGNN.get_embeddings(x_har, adj)   # [B,32,64],[B,32,256] (read-only reuse)
     news_rep = NewsFeatureLSTM(x_news)                            # LSTM 1 lớp, 146→64, qua 22 ngày
     h = concat([h_lstm, h_gnn, news_rep])                         # [B,32,384]
     pred = MLP(h)                                                 # như bản gốc
                    │
     train_dual_news.py — train loop chuẩn (MSE loss, 6 metric bắt buộc, learning curve mỗi 5 epoch,
     early stopping patience=15, weight_decay=1e-5, dropout=0.2/0.5, grad clip=1.0 — theo CLAUDE.md §3.E)
```

## 2. File list

| File | Trách nhiệm |
|---|---|
| `code/vendor_data_eda/discover_news.py` | Rút gọn từ data_eda: `discover_source_files`, `load_source` (bỏ `log_processing`) |
| `code/vendor_data_eda/phase04_news_helpers.py` | Rút gọn: `TOPIC_CATEGORIES`, `effective_trading_date`, `_trading_calendar` (đọc `data/processed/`), `SOURCE_DAYFIRST`, `VN_TZ`, `MARKET_CLOSE_HOUR` |
| `code/vendor_data_eda/phobert_embeddings.py` | Copy nguyên `src/nlp/embeddings.py` (chỉ để import không lỗi; dự kiến KHÔNG được gọi vì cache đầy đủ) |
| `code/vendor_data_eda/news_embeddings.py` | Rút gọn từ `src/features/news_embeddings.py` (bỏ `run()`/`log_processing`) |
| `code/vendor_data_eda/dual_news_features.py` | Rút gọn từ `src/modeling/features.py` (bỏ `PURE_NEWS_FEATURES` re-export, bỏ `run()`/eda.common) |
| `code/vendor_config.py` | Path config trỏ về project này (xem Giai đoạn A) |
| `code/build_dual_group_panel.py` | Script chạy 1 lần: gọi `build_advanced_features(mode="ewma")` → `data/features/dual_group_news_panel.parquet` |
| `code/dataset_dual_news.py` | Subclass dataset, đọc panel parquet thay vì `.npz` |
| `code/model_dual_news.py` | `DualGroupNewsBaseline` — HAR branch reuse + `NewsFeatureLSTM` mới |
| `code/train_dual_news.py` | Train loop (10 epoch thử nghiệm) |
| `test/test_build_panel_smoke.py` | Chạy `build_dual_group_panel.py` trên 1 lát cắt nhỏ raw_cache thật → assert output shape/coverage hợp lý (real-data-sample smoke, theo CLAUDE.md Testing rules) |
| `test/test_dataset_smoke.py` | Dataset trả đúng shape (x_har, adj, x_news, y) |
| `test/test_model_smoke.py` | Forward + backward không NaN, đúng shape output `[B, 32]` |

## 3. Isolation

- KHÔNG sửa file nào trong `C:\luanvan\data_eda` (chỉ đọc để copy).
- KHÔNG sửa `src/` chung hay baseline khác trong `stock_vol_prediction01` — chỉ import read-only
  (`ParallelLSTMGNN`, `MultiStockDatasetWithPreSplitData`, `VolatilityNormalizer`, `VN30_TICKERS`).
- Data copy nằm ở `data/external_news_embeddings/raw_cache/` (mới, không đụng data hiện có).
- Output: `results/dual_group_news_<timestamp>/`, `models/dual_group_news_<timestamp>/`.

## 4. Risk / lưu ý

- **Kích thước copy ~4.4GB** — đã kiểm tra disk free (212GB) đủ chỗ. 3 file `_root` (thanhnien/
  tuoitre/vietnamplus) chiếm phần lớn dung lượng (full site crawl, không chỉ VN30-relevant).
- **0 cache-miss kỳ vọng** — nếu `build_dual_group_panel.py` log > 0 cache-miss (nghĩa là gọi
  PhoBERT thật), dừng lại và báo user (vi phạm ràng buộc "không rebuild embedding"; có thể do
  crawl_data có bài mới hơn lần data_eda cache cuối).
- **Coverage risk** — dual-group cần match ticker theo regex trên title+lead; nếu coverage (số
  ngày có tin) thấp hơn hẳn baseline PCA-64 gốc, nghi ngờ lỗi trong bước vendor (vd sai
  `TICKER_PATTERN`, sai `effective_trading_date`) — kiểm tra trước khi train.
