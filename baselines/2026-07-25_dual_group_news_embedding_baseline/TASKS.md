# Tasks — Dual-Group News Embedding Baseline

**Epic:** Đưa pipeline dual-group PhoBERT+EWMA embedding (từ `data_eda`) vào `stock_vol_prediction01`
thành 1 baseline mới, so sánh với `EmbeddingBaseline` gốc (68.76%@40ep).

---

## Story 1 — Copy dữ liệu + code cần thiết từ `data_eda` (không sửa gì ở đó)

| # | Task | Verify |
|---|---|---|
| 1.1 | Copy `data_eda/data/features/news_emb_articles_*.parquet` (+ 2 file `news_emb_group_*.parquet`) → `data/external_news_embeddings/raw_cache/` | `ls` số file khớp nguồn, tổng dung lượng ~4.4GB, `data_eda` không bị sửa (`git status` phía đó nếu có, hoặc so mtime trước/sau) |
| 1.2 | Copy + rút gọn `discover_news.py`, `news_embeddings.py`, `modeling/features.py`, phần cần của `phase04_news_eda.py`, `nlp/embeddings.py` → `code/vendor_data_eda/` | Mỗi file có docstring nói rõ "vendored from data_eda YYYY-MM-DD, trimmed: <lý do>"; import graph tự đóng (không còn ref tới `src.eda.common`/`pure_news`/`log_processing`) |
| 1.3 | Viết `code/vendor_config.py` (PROJECT_ROOT/CRAWL_DATA_ROOT/FEATURES_DIR/PRICE_DATA_DIR/VN30_TICKERS trỏ về project này) | Import thử `python -c "from vendor_config import *; print(CRAWL_DATA_ROOT.exists())"` → True |

## Story 2 — Rebuild aggregation (PCA/EWMA), KHÔNG rebuild PhoBERT

| # | Task | Verify |
|---|---|---|
| 2.1 | Viết `code/build_dual_group_panel.py` gọi `build_advanced_features(mode="ewma")` | Chạy xong không exception |
| 2.2 | Xác nhận 0 cache-miss (không gọi PhoBERT thật) | Log script in ra "0 new rows encoded" hoặc tương đương; nếu >0 → dừng, báo cáo |
| 2.3 | Output `data/features/dual_group_news_panel.parquet` (ticker, date, 146 cols) | Shape đúng (n_ticker × n_trading_day, 146+2 cols); coverage (% ngày có tin, không NaN toàn bộ) so sánh không thấp bất thường so với baseline PCA-64 gốc |
| 2.4 | Test smoke thật: `test/test_build_panel_smoke.py` chạy trên 1 lát cắt nhỏ raw_cache (vài ticker/vài trăm dòng) | `pytest test/test_build_panel_smoke.py -v` pass |

## Story 3 — Dataset + Model (tái dùng pattern `2026-07-07_embedding_baseline`)

| # | Task | Verify |
|---|---|---|
| 3.1 | `code/dataset_dual_news.py` — subclass đọc panel parquet, trả (x_har, adj, x_news, y) | Unit test shape đúng cho 1 batch |
| 3.2 | `code/model_dual_news.py` — `DualGroupNewsBaseline` (HAR reuse + `NewsFeatureLSTM` mới) | Forward+backward không NaN, output shape `[B, 32]` |
| 3.3 | `code/train_dual_news.py` — train loop 10 epoch, in đủ 6 metric console+JSON mỗi 5 epoch + learning curve | Chạy xong 10 epoch không crash |

## Story 4 — Test, Review, Report

| # | Task | Verify |
|---|---|---|
| 4.1 | Unit test cho dataset/model (`test/test_dataset_smoke.py`, `test/test_model_smoke.py`) | `pytest test/ -v` toàn bộ pass |
| 4.2 | Chạy `/code-review` (adversarial, 3-layer) trên toàn bộ code mới | Fix hết finding HIGH/MEDIUM; lưu kết quả `code_review/code_review_2026-07-25.md` |
| 4.3 | Train thật 10 epoch, so sánh DirAcc với baseline gốc | Console + JSON kết quả val/test 6 metric; so sánh bảng với 68.76%/68.44%/70.29% |
| 4.4 | Summary report | `docs/reports/2026-07-25_HHMM_summaryOfUpdate_report.md` theo CLAUDE.md template |

---

**Go/no-go giữa Story 2 và Story 3:** nếu panel fresh coverage bất thường thấp (nghi bug), DỪNG
và debug trước khi sang Story 3 (không train trên feature lỗi).

**Go/no-go sau Story 4.3:** nếu kết quả xấu hơn hẳn baseline gốc + không có dấu hiệu cải thiện,
báo cáo trung thực trong summary report — không cần "thắng" để coi baseline "done" (đây là thí
nghiệm so sánh, không phải production gate — xem requirements.md §5).
