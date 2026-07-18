# Design (Plan) — Objective News Baseline

**Baseline:** `2026-07-15_objective_news_baseline` · Pha Plan theo SDD (CLAUDE.md §1.5).

## 1. Quyết định design

**Reuse tối đa, viết mới tối thiểu.** Model + dataset + training loop của
`2026-07-07_embedding_baseline` đã nhận `--emb_dir` làm tham số (`dataset_embedding.py:208`,
`train_embedding_baseline.py:109`) — nghĩa là **không cần sửa/viết lại model hay train script**.
Chỉ cần 1 script trích xuất mới ghi ra đúng format cache `.npz` mà `MultiStockDatasetWithEmbedding`
đọc được, trỏ vào 1 thư mục output mới.

- **Simplicity Gate:** pass — không thêm project/abstraction mới, chỉ 1 script trích xuất.
- **Anti-Abstraction Gate:** pass — dùng thẳng `train_embedding_baseline.py` có sẵn qua CLI arg
  `--emb_dir`, không tự wrap thêm lớp train mới.

## 2. Data flow

```
crawl_data/data/objective/{vietstock_records,vsdc_records}.csv   (company_code có sẵn)
crawl_data/data/objective/news_unenriched_{5 nguồn}.csv          (company_code rỗng)
                    │
     extract_objective_embeddings.py (code MỚI, baseline này)
                    │
   1. Load 7 file (skip objective_v2026-07-* theo requirements.md §2)
   2. company_code có sẵn → filter VN30 trực tiếp
      company_code rỗng   → ticker-regex trên title+raw_text (strip HTML trước)
   3. Dedup theo document_id/checksum (đã có sẵn từ nguồn, ít khả năng trùng)
   4. Text = title (+ raw_text nếu khác title, tránh nhân đôi cho vietstock/vsdc)
   5. Date = publish_time → date (YYYY-MM-DD); drop 5 dòng publish_time > crawl_time (leakage)
   6. PhoBERT encode (vinai/phobert-base, max_len=64) — CÙNG recipe với extract_embeddings.py
   7. PCA 768->64, fit TRAIN-only (date < 2020-01-01) — cùng TRAIN_CUTOFF
                    │
       data/objective_embedding/{TICKER}_emb.npz   (dir MỚI, không đụng data/sentiment_embedding)
                    │
   train_embedding_baseline.py --emb_dir data/objective_embedding --epochs 10   (KHÔNG sửa, chạy thẳng)
                    │
              results/objective_news_<ts>/
```

## 3. File list

| File | Mục đích |
|------|----------|
| `code/extract_objective_embeddings.py` | Đọc 7 CSV objective, ticker-match, PhoBERT encode, PCA, ghi `.npz` |
| `test/test_extract_objective_embeddings.py` | pytest: ticker-match (company_code sẵn vs regex), HTML strip, date-leakage drop, PCA train-only fit, dedup |

Không viết train script mới — chạy thẳng
`python baselines/2026-07-07_embedding_baseline/code/train_embedding_baseline.py --emb_dir data/objective_embedding ...`
(sibling script không bị sửa — chỉ gọi qua CLI, đúng tinh thần "import read-only" của §3.F.3).

## 3b. Incremental extraction (bổ sung 2026-07-15, theo yêu cầu user)

User crawl raw data mỗi ngày — extraction KHÔNG được re-encode toàn bộ corpus mỗi lần chạy.

- **Manifest** `data/objective_embedding/_manifest.json` = `{"document_ids": [...]}` — mọi
  `document_id` (fallback `checksum`) đã encode. Lần chạy sau, `build_records(tickers,
  skip_ids=processed_ids)` bỏ qua các dòng đã có trong manifest → chỉ encode dòng MỚI.
- **PCA persisted** `data/objective_embedding/_pca.pkl` (pickle, fit 1 lần) — lần chạy sau
  áp `pca.transform()` lên embedding MỚI, **KHÔNG refit** (refit trên dữ liệu mới, vốn luôn
  là test-period vì train_cutoff=2020-01-01 đã lùi xa, sẽ leak). `--refit_pca` là escape
  hatch thủ công, chỉ dùng nếu có data pre-2020 mới xuất hiện.
- **Merge, không ghi đè:** cache `.npz` cũ được load lại, record mới được nối thêm theo
  ngày (cùng ngày → concat mảng), ngày mới → thêm key mới, rồi ghi lại.
- **Dòng không có `document_id`/`checksum`:** không track được, bị quét lại mỗi lần chạy
  nhưng vô hại (bị lọc lại bởi cùng filter, không có state để cache).
- **Bootstrap:** lần đầu bật incremental, đã xóa `data/objective_embedding/` cũ (tạo trước
  khi có manifest) để tránh double-count khi merge.

## 4. Isolation

- Đọc: `D:/bmad-projects/crawl_data/data/objective/*.csv` (read-only, theo Data source rules).
- Ghi: `data/objective_embedding/` (mới), `results/objective_news_<ts>/`, `models/objective_news_<ts>/`
  (theo §3.D — không tạo trong folder baseline).
- Không sửa `src/`, không sửa `baselines/2026-07-07_embedding_baseline/` hay baseline khác.

## 5. Hyperparameters

Khớp `extract_embeddings.py` để so sánh công bằng: `model=vinai/phobert-base`, `dim=64` (PCA),
`train_cutoff=2020-01-01`, `max_len=64`, `batch_size=32`.
Training: `epochs=10` (đầu tiên, theo Training policy — KHÔNG phải 20), `lr=5e-3`,
`weight_decay=1e-5`, `graph_method=knn`, `dropout=0.5` — khớp embedding baseline / latent noise để
so sánh công bằng.
