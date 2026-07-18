# Design (Plan) — Pure Market-Vector Baseline

## 1. Quyết định design

**Reuse tối đa.** HAR branch: reuse `ParallelLSTMGNN.get_embeddings` (giống mọi baseline khác).
News pooling: reuse `ArticleSetAttentionPooling` + `NewsTemporalEncoder` từ
`2026-07-07_embedding_baseline/code/model_embedding.py` (read-only import) — các module này vốn
dimension-agnostic ở trục "stocks" (chỉ Linear + softmax + weighted-sum trên trục cuối), nên
dùng lại được cho market (stocks-dim=1) mà không cần viết pooling mới.

- **Simplicity Gate:** pass — 0 file extraction mới (tái dùng `market_emb.npz` có sẵn), chỉ 1
  dataset mới (đơn giản hơn `dataset_embedding.py` vì bỏ hẳn nhánh per-stock news) + 1 model nhỏ.
- **Anti-Abstraction Gate:** pass — dùng thẳng `ArticleSetAttentionPooling`/`NewsTemporalEncoder`
  có sẵn qua adapter unsqueeze/squeeze 1 dòng, không viết pooling logic trùng lặp.

## 2. Data flow

```
data/sentiment_embedding/market_emb.npz   (CÓ SẴN, {date: [n_articles, 64]}, KHÔNG lọc ticker)
                    │
     PureMarketDataset (code MỚI, subclass MultiStockDatasetWithPreSplitData)
     - mỗi cửa sổ 22 ngày: x_har[22,32,3] (như cũ, per-stock)
     - x_market[22, MAX_M=15, 64]  ← load market_emb theo NGÀY (không theo mã!)
     - market_mask[22, MAX_M]
     - y[32]  (như cũ)
                    │
     PureMarketBaseline (model MỚI)
     h_lstm[B,32,64], h_gnn[B,32,256] = HAR.get_embeddings(x_har, adj)   (như cũ)
     x_market5d = x_market.unsqueeze(2)          # [B,22,1,15,64]  (stocks-dim=1)
     market_mask4d = market_mask.unsqueeze(2)    # [B,22,1,15]
     daily = ArticleSetAttentionPooling(x_market5d, market_mask4d)   # [B,22,1,64]
     market_rep = NewsTemporalEncoder(daily).squeeze(1)              # [B,64]  (1 vector/sample)
     market_bc = market_rep.unsqueeze(1).expand(-1, 32, -1)          # [B,32,64] (BROADCAST, không gate)
     h = concat([h_lstm, h_gnn, market_bc], dim=-1)   # [B,32,384]
     fusion MLP (384->64->32->1, giống các baseline khác)            # [B,32]
                    │
              results/pure_market_<ts>/
```

## 3. File list

| File | Mục đích |
|---|---|
| `code/dataset_pure_market.py` | `PureMarketDataset` — subclass, load market cache theo ngày (không per-stock) |
| `code/model_pure_market.py` | `PureMarketBaseline` — HAR + market-broadcast (không gate, không per-stock branch) |
| `code/train_pure_market.py` | Training loop (thin — copy cấu trúc train loop từ sibling, KHÔNG sửa sibling) |
| `test/test_pure_market.py` | pytest: dataset shape, market load-by-date, broadcast identical across stocks, no-market-day fallback |

## 4. Isolation

- Đọc: `data/sentiment_embedding/market_emb.npz` (có sẵn, read-only).
- Không sửa `src/`, không sửa `2026-07-07_embedding_baseline/` hay `2026-07-08_market_fallback/`.
- Output: `results/pure_market_<ts>/`, `models/pure_market_<ts>/` (§3.D).

## 5. Hyperparameters

Khớp các baseline khác: `d_news=64`, `dropout=0.5`, `graph_method=knn`, `lr=5e-3`,
`weight_decay=1e-5`, `epochs=10` (thí nghiệm ban đầu), `MAX_MARKET=15` (khớp market_fallback,
percentile 99 số bài/ngày).
