# Design — Market Fallback Baseline

**Baseline:** `2026-07-08_market_fallback`
**Tham chiếu đầy đủ:** `docs/project/SENTIMENT_MARKET_FALLBACK_ARCHITECTURE.md` (pseudocode + diagram + go/no-go)
**Rule:** CLAUDE.md §3.F

---

## Quyết định design (tóm tắt)

| Quyết định | Chọn | Lý do |
|---|---|---|
| Market encoder | PhoBERT frozen, offline | reuse pipeline embedding baseline |
| Market PCA | fit trên train-period market articles (date<2020) | no leakage, độc lập với per-stock PCA |
| Market branch | reuse `ArticleSetAttentionPooling` | đã test permutation + 0-news |
| Gate | **deterministic**: `g = has_news` (binary) | MVP, 0 tham số, không risk học sai |
| MAX_M (articles/day market) | 15 | percentile 99 (ngày vĩ mô đông tin) |
| HAR branch | reuse `ParallelLSTMGNN.get_embeddings` | proven (69.98% DirAcc) |
| Fusion | late concat [64+256+d_news] → MLP | same as embedding baseline |

## Data flow

```
[OFFLINE]
unified_articles.csv → (ALL articles, no ticker filter) → PhoBERT → 768 → PCA → 64
  → data/sentiment_embedding/market_emb.npz  {date: [n_articles, 64]}

[ONLINE — __getitem__ returns 7-tuple]
(x_har[seq,stocks,3], adj, x_emb[seq,stocks,MAX,64], mask,
 x_market[seq,MAX_M,64], market_mask[seq,MAX_M], y[stocks])

[forward]
h_lstm, h_gnn = ParallelLSTMGNN.get_embeddings(x_har, adj)
stock_daily = ArticleSetAttentionPooling(x_emb, mask)        # [B,seq,stocks,d]
market_daily = MarketBranch(x_market, market_mask)           # [B,seq,d]
has_news = (mask.sum(-1,keepdim=True) > 0).float()           # [B,seq,stocks,1]
daily = has_news * stock_daily + (1-has_news) * market_daily.unsqueeze(2)  # gated
news_rep = NewsTemporalEncoder(daily)                        # [B,stocks,d]
pred = MLP(concat[h_lstm, h_gnn, news_rep])                  # [B,stocks]
```

## File list (code/)

| File | Trách nhiệm | Delta vs embedding baseline |
|---|---|---|
| `extract_market_embeddings.py` | PhoBERT ALL articles → PCA → market_emb.npz | no ticker filter, group by date |
| `dataset_embedding.py` | 7-tuple, load market cache | + market loading + x_market build |
| `model_embedding.py` | MarketFallbackBaseline | + MarketBranch + GatedNewsFusion |
| `train_market_fallback.py` | training loop | 7-tuple unpacking, new forward |

## Critical correctness (per design doc §4)

1. PCA fit train-period market articles ONLY (date < 2020), no widen-to-all (HIGH-1 lesson).
2. Date keys normalized YYYY-MM-DD cả 2 bên (HIGH-2 lesson).
3. Gate deterministic MVP — `g = has_news` (binary), không learned.
4. Market broadcast đúng shape: `market_daily.unsqueeze(2).expand_as(stock_daily)`.
5. Finite guard (MEDIUM-6 lesson): skip non-finite articles in `_pad_articles`.
6. weight_decay=1e-5 (§3.E), pytest mandatory, learning curves (§3.C).

## Param budget (trainable, nhánh market mới)

- MarketBranch: 0 (reuse ArticleSetAttentionPooling → query+no_news_token ~128 params)
- GatedNewsFusion deterministic: 0 (binary gate)
- → nhánh market thêm ~128 trainable params. Negligible.

## Go/no-go (per requirements §3)

val DirAcc market-fallback > embedding-baseline (71.32% val / 68.76% test) @ matched epoch. Nếu không vượt → NO-GO chốt.
