# Báo cáo — Embedding Baseline: kết quả & so sánh các model

**Ngày:** 08/07/2026
**Baseline:** `baselines/2026-07-07_embedding_baseline/`
**Run tham chiếu:** `results/embedding_baseline_2026-07-08_003719/` (40 epoch, real PhoBERT embeddings)
**Tham chiếu kiến trúc:** `docs/project/SENTIMENT_NEWS_EMBEDDING_ARCHITECTURE.md`

---

## TL;DR (trung thực)

**Embedding baseline KHÔNG cho ra tín hiệu rõ ràng hơn HAR-only.** Ở test set:

| | Embedding (40ep) | Parallel LSTM-GNN HAR-only (70ep) |
|---|---|---|
| DirAcc | 68.76% | **69.98%** ← vẫn cao nhất |
| R² | 0.7174 | 0.714 |
| QLIKE | 0.5534 | **0.529** ← vẫn tốt nhất |

→ **Go/no-go: KHÔNG conclusive, nghiêng NO-GO.** Embedding không vượt HAR-only ở DirAcc/QLIKE test (thực tế thấp hơn 1.2% DirAcc). R² gần như identical. **Kết quả này ĐÚNG với dự đoán** từ sparsity analysis (test chỉ 5.5% stock-day có tin, match-rate ~20%) — đã cảnh báo trong design doc.

⚠️ **Caveat quan trọng:** so sánh bị confound bởi **số epoch khác nhau** (embedding 40 vs HAR-only 70 vs scalar 20). Chưa thể kết luận dứt khoát. Cần matched-epoch control.

---

## 1. Setup

- **Kiến trúc:** HAR branch (LSTM+GAT, reuse `ParallelLSTMGNN.get_embeddings`) + News branch (offline PhoBERT frozen → PCA 768→64 → `ArticleSetAttentionPooling` → 1-L temporal LSTM) → concat → MLP.
- **Embedding thật:** 3,442 ticker-matched articles → PCA fit trên 625 train articles (explained var 82.9%, no leakage) → 30 caches `data/sentiment_embedding/`.
- **Hyperparams:** Adam lr=5e-3, weight_decay=1e-5, dropout=0.5, patience=15, d_news=64, MAX_ARTICLES=10. CPU-only.
- **Anti-overfit đầy đủ (§3.E):** weight_decay, dropout, early stopping, gradient clipping, frozen PhoBERT, frozen dead fusion MLP.

## 2. Kết quả Embedding Baseline (6 metrics, §3.B)

| Metric | Validation | Test | Diff |
|---|---|---|---|
| MSE | 6.01e-06 | 6.91e-06 | +8.95e-07 |
| RMSE | 0.002452 | 0.002628 | +0.000176 |
| MAE | 0.000729 | 0.000724 | −5.0e-06 |
| R² | 0.6597 | **0.7174** | +0.058 |
| QLIKE | 0.6968 | 0.5534 | −0.143 |
| **DirAcc (pooled)** | **71.32%** | **68.76%** | −2.55% |
| DirAcc (per-stock) | 47.68% | 48.03% | +0.35% |

**Đánh giá overfit (§3.C):** val/test gap khiêm tốn (DirAcc −2.55%, R² test còn cao hơn val) → **không overfit nặng**. Learning curve (epoch 40) cho thấy train/val loss hội tụ, không phân kỳ. Model **khỏe**, chỉ không tốt hơn.

## 3. So sánh toàn bộ model

| Model | Epochs | Test DirAcc | R² | RMSE | QLIKE | Ghi chú |
|---|---|---|---|---|---|---|
| HAR-R Linear | — | 51.53% | 0.105 | 0.000513 | 1.298 | regime cũ¹ |
| Simple LSTM | — | 67.55% | 0.155 | 0.000582 | 0.578 | regime cũ¹ |
| LSTM-HAR | — | 67.76% | 0.159 | 0.000563 | 0.880 | regime cũ¹ |
| Enhanced LSTM-HAR | — | 67.90% | 0.136 | 0.000603 | 0.587 | regime cũ¹ |
| Parallel LSTM-GNN (HAR-only) | 70 | **69.98%** | 0.714 | 0.002644 | **0.529** | base này mở rộng |
| Scalar sentiment (lexicon) | ~20 | 67.96% | 0.713 | 0.002600 | 0.576 | title-only score |
| **Embedding baseline (MỚI)** | 40 | 68.76% | **0.717** | 0.002628 | 0.553 | PhoBERT embedding |

¹ *Regime cũ (06-19): single-stock, R² thấp (0.10-0.16) — **không so sánh trực tiếp** với nhóm multi-stock (R² ~0.71). Để ở đây cho đủ lịch sử.*

**Nhóm so sánh hợp lệ (cùng regime R²~0.71):** Parallel LSTM-GNN (HAR) / scalar sentiment / embedding.

## 4. Phân tích trung thực — vì sao không có lift

1. **Sparsity gốc (đã dự đoán):** chỉ **5.5% stock-day** ở test có tin ticker-specific (sparsity report 06-07). Match-rate ~20% bài → 94.5% stock-day nhánh news = `no_news_token`. Embedding giàu hơn score NHƯNG không tạo tín hiệu mới ở phần lớn mẫu.
2. **Title-only:** 70% bài không có `lead` → embedding từ title ngắn VN, nhiễu.
3. **Confounding epoch:** embedding 40ep vs HAR-only 70ep — HAR-only được train lâu hơn. Có thể embedding chưa hội tụ hoàn toàn, HOẶC HAR-only được lợi thêm.
4. **ArticleSetAttentionPooling no-op ở ngày 1 bài (MEDIUM-7):** phần lớn stock-day có 0-1 bài → attention query không học được nhiều.

→ **Kết luận:** pipeline embedding **chạy đúng, không leakage, model khỏe**, nhưng **tín hiệu thật của news bị chìm** dưới sparsity. Đây là kết quả trung thực, không phải bug.

## 5. Go/no-go decision

**❌ NO-GO (tạm thời) cho Bước 3 (cross-attention MSGCA) + Bước 4 (unfreeze PhoBERT).** Lý do:
- Embedding chưa vượt HAR-only → không chứng minh được "embedding > scalar" (requirements §3 criterion #4).
- Đầu tư thêm ~200 dòng cross-attention khi base signal yếu là vi phạm luật Simplicity.

**Trước khi chốt NO-GO cuối cùng, cần loại bỏ confounding:**
- [ ] Chạy HAR-only + embedding + scalar sentiment tại **cùng epoch** (vd 40) → so sánh công bằng.
- [ ] Nếu embedding vẫn ≤ HAR-only tại matched epoch → NO-GO chốt, sentiment/embedding (với data title-only hiện tại) không đủ tín hiệu.

## 6. Khuyến nghị tiếp theo (theo priority)

| # | Hành động | Lý do |
|---|---|---|
| 1 | **Matched-epoch control** (HAR-only + scalar + embedding @ 40ep) | loại confound trước khi kết luận |
| 2 | Nếu vẫn no-lift → **crawl body bài** (không chỉ title) | title ngắn = ceiling thấp (Mục 10 design) |
| 3 | Thử **market-level sentiment fallback** (sparsity doc Solution 3) | tận dụng ~80% bài vĩ mô đang bị bỏ |
| 4 | (defer) Bước 3 MSGCA cross-attention | chỉ khi 1-3 cho tín hiệu |
| 5 | (tùy chọn) **statistical test** (skill `statistical-analysis`) ý nghĩa khác biệt DirAcc | xác nhận lift/không-lift không phải noise |

## 7. Files & provenance

- Code: `baselines/2026-07-07_embedding_baseline/code/` (review + 10 fixes: `code_review/code_review_2026-07-07.md`)
- Results: `results/embedding_baseline_2026-07-08_003719/` (`results.json` + 8 learning-curve PNG epoch 5→40)
- Embedding cache: `data/sentiment_embedding/{TICKER}_emb.npz` (30 tickers, dim=64)
- Sparsity analysis: `crawl_data/aggregated/sparsity_report.txt`
- Baseline metrics: `results/all_metrics_comparison_2026-06-19_073515.json`, `results/sentiment_baseline_knn_*/`

---

**Phiên bản:** 1.0 — 08/07/2026 · **Verdict:** NO-GO tạm thời (chờ matched-epoch control) · **Báo cáo trung thực:** pipeline đúng, model khỏe, không có lift do sparsity gốc.
