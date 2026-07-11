# Requirements — Embedding Baseline

**Baseline:** `2026-07-07_embedding_baseline`
**Ngày bắt đầu:** 07/07/2026
**Tham chiếu kiến trúc:** `docs/project/SENTIMENT_NEWS_EMBEDDING_ARCHITECTURE.md`
**Rule tuân thủ:** `CLAUDE.md` mục 3.F (Baseline Implementation Structure)

---

## 1. Mục tiêu

Kiểm tra giả thuyết: **dùng news embedding vector 768-d (PhoBERT frozen) phong phú hơn sentiment score scalar** → cải thiện val DirAcc cho Parallel LSTM-GNN trên VN30 volatility (5-day ahead).

Giải quyết bottleneck từ sparsity analysis (06/07): chỉ ~20% bài match mã VN30 → mỗi bài match phải mang NHIỀU thông tin nhất có thể. Embedding giữ ngữ nghĩa gốc; score nén mất ~99%.

## 2. Input / Output

**Input:**
- `D:/bmad-projects/crawl_data/aggregated/unified_articles.csv` — 21,107 bài unique (có `title`, `lead`, `date`, `source`).
- `data/processed/{TICKER}_processed.csv` — HAR features + lịch giao dịch (READ ONLY).
- `data/processed/processing_summary.csv` — danh sách 30 mã VN30 (READ ONLY).

**Output (cô lập, không đụng folder cũ):**
- `data/sentiment_embedding/{TICKER}_emb.npz` — cache embedding offline (date → [num_articles, d_proj]).
- `results/embedding_baseline_<timestamp>/` — metrics JSON + console (6 metric bắt buộc).
- `models/embedding_baseline_<timestamp>/` — checkpoint.

## 3. Success Criteria (verifiable)

| # | Tiêu chí | Verify |
|---|----------|--------|
| 1 | Pipeline chạy end-to-end (extract → dataset → train) không lỗi shape | smoke test pass |
| 2 | `ArticleSetAttentionPooling` permutation-invariant + handle 0-news | unit test pass |
| 3 | Val metric không хуже HAR-only baseline (> ~70% DirAcc, không regression) | val DirAcc ≥ 68% @ matched epochs |
| 4 | **Go/no-go cốt lõi**: val DirAcc **embedding > scalar sentiment** tại cùng epoch (10/15/20) | so với `src/sentiment_baseline/` chạy cùng setup |

**Success criteria #4 là quyết định:** nếu embedding KHÔNG hơn scalar sentiment trên val → tín hiệu gốc yếu, DỪNG, không đầu tư Bước 3 (cross-attention MSGCA) hay unfreeze PhoBERT.

## 4. Scope (MVP — simplicity first)

**Làm:**
- Offline PhoBERT frozen → embedding 768-d → PCA 768→64 (default) → cache `.npz`.
- `ArticleSetAttentionPooling` per (stock, day) + learned "no-news" token.
- News temporal encoder (1-L LSTM over 22 ngày).
- Late-fusion **concat** [HAR-LSTM 64 + HAR-GAT 256 + news 64] → MLP.
- Reuse `ParallelLSTMGNN.get_embeddings` cho HAR branch (read-only import).

**KHÔNG làm (defer):**
- Cross-attention / MSGCA fusion (Bước 3).
- Unfreeze PhoBERT / Siamese head (Bước 4).
- VAE/latent layer trên nhánh news (separate baseline nếu cần).

## 5. Isolation (hard requirement)

- KHÔNG sửa: `src/`, `data/processed/`, `src/sentiment_baseline/`, hay baseline khác.
- Import read-only từ `src/lstm_gat_hybrid/` (model, dataset helpers) thì ĐƯỢC.
- Mọi output ra `data/sentiment_embedding/`, `results/`, `models/` (không ghi ngược vào data nguồn).

## 6. Known Risks (trung thực)

| Risk | Giảm nhẹ |
|---|---|
| Data hiệu dụng cực nhỏ (~953 train stock-days có tin) | PCA offline (0 trainable param projection) + dim 64 + dropout 0.5 |
| PhoBERT env: `transformers<5` bắt buộc (gotcha từ memory) | doc trong `extract_embeddings.py`; smoke test dùng dummy embedding không cần PhoBERT |
| Variable articles/day (0..15) | `MAX_ARTICLES=5` cap + mask; truncation caveat documented |
| Title-only cho 70% bài (không lead) | embedding từ title vẫn richer score; note ceiling |
