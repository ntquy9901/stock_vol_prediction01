# Requirements — Market Fallback Baseline

**Baseline:** `2026-07-08_market_fallback`
**Ngày bắt đầu:** 08/07/2026
**Tham chiếu:** `docs/project/SENTIMENT_MARKET_FALLBACK_ARCHITECTURE.md` (design đầy đủ), `EMBEDDING_BASELINE_REPORT_2026-07-08.md` (khuyến nghị #3)
**Rule:** CLAUDE.md §3.F (5 sub-folder), §3.E (overfit), §3.C (learning curves), pytest mandatory

---

## 1. Mục tiêu

Giải quyết no-lift của embedding baseline (test DirAcc 68.76% < HAR-only 69.98%): thêm **nhánh market news** (dense, ~100% ngày có) bên cạnh nhánh per-stock (sparse, 5.5%), qua **gate** fallback khi stock mù tin. Tận dụng ~80% bài vĩ mô đang bị bỏ.

## 2. Input / Output

**Input:**
- `D:/bmad-projects/crawl_data/aggregated/unified_articles.csv` (21,107 bài)
- `data/processed/{TICKER}_processed.csv` + `processing_summary.csv` (READ ONLY)
- PhoBERT frozen (env: `transformers<5`, đã có 4.57.6)

**Output (cô lập):**
- `data/sentiment_embedding/market_emb.npz` — cache market embedding {date: [n_articles, dim]}
- `results/market_fallback_<timestamp>/` — metrics + learning curves
- `models/market_fallback_<timestamp>/` — checkpoint

## 3. Success Criteria (verifiable)

| # | Tiêu chí | Verify |
|---|----------|--------|
| 1 | Pipeline end-to-end (extract market → dataset → train) không lỗi | smoke pass |
| 2 | Market cache coverage > 90% ngày giao dịch | count dates in market_emb.npz |
| 3 | Gate đúng: has_news=1 → stock, has_news=0 → market | unit test |
| 4 | MarketBranch permutation-invariant + 0-news handle | unit test |
| 5 | **Go/no-go cốt lõi:** val DirAcc **market-fallback > embedding-baseline** (68.76% test / 71.32% val) tại matched epoch | so results.json |

**Success #5 là quyết định:** nếu market-fallback KHÔNG vượt embedding baseline → market signal cũng không đủ → NO-GO chốt, ceiling do data.

## 4. Scope (MVP — simplicity first)

**Làm:**
- `extract_market_embeddings.py`: PhoBERT frozen, ALL articles (no ticker filter) → PCA 768→64 → `market_emb.npz`.
- `MarketBranch` (reuse `ArticleSetAttentionPooling`).
- `GatedNewsFusion` deterministic MVP: `g = has_news` → `daily = g·stock + (1−g)·market`.
- Tích hợp vào `MarketFallbackBaseline` (HAR reuse + per-stock news + market + gate + temporal + concat).

**KHÔNG làm (defer):**
- Learned soft gate (chỉ nếu MVP có tín hiệu).
- Sector-level fallback (cần phân loại ngành — chưa có).
- Cross-attention / unfreeze PhoBERT.

## 5. Isolation (hard)

- KHÔNG sửa `src/`, `baselines/2026-07-07_embedding_baseline/`, hay baseline khác.
- Import read-only từ `src/lstm_gat_hybrid/` thì ĐƯỢC.
- Output ra `data/sentiment_embedding/`, `results/`, `models/`.

## 6. Known risks

| Risk | Giảm nhẹ |
|---|---|
| Market cache nặng (~17K bài encode) | reuse PhoBERT pipeline; cache 1 lần offline |
| Gate collapse (luôn dùng market) | deterministic gate (không learned) + check var(news_rep) |
| Market = systematic mood, không stock-alpha | kỳ vọng lift NHỎ (honest); không overclaim |
| Ceiling do title-only data | nếu no-lift → crawl body (next step) |
