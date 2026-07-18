# Design (Plan) — REST-TS Baseline

## Data flow

```
data/sentiment_embedding/{TICKER}_emb.npz  (CÓ SẴN, tái dùng nguyên baseline 2026-07-07)
                    │
     create_embedding_dataloaders (sibling, KHÔNG sửa, chỉ import read-only)
                    │
   x_har[B,22,32,3] adj[B,32,32]  x_emb[B,22,32,10,64] mask[B,22,32,10]  y[B,32]
                    │
     RestTsBaseline (model MỚI)
     h_lstm,h_gnn = HAR.get_embeddings(x_har,adj)     # [B,32,64],[B,32,256]
     har_pred = har_head(concat[h_lstm,h_gnn])         # [B,32]  — ĐỘC LẬP, không thấy news
     news_rep = news_temporal(news_pool(x_emb,mask))   # [B,32,64] (reuse pooling có sẵn)
     news_pred = news_head(news_rep)                   # [B,32]  — chỉ predict RESIDUAL
     combined = har_pred + news_pred                   # dự báo cuối
                    │
     train_resttext.py (loop MỚI, khác sibling vì loss 2 phần)
     loss_har  = MSE(har_pred, y)
     residual_target = (y - har_pred).detach()          # [KHÓA GRADIENT] ép news học residual thật
     loss_news = MSE(news_pred, residual_target)
     loss = loss_har + loss_news
     eval metrics đo trên `combined` (denormalized)
```

## File list

| File | Mục đích |
|---|---|
| `code/model_resttext.py` | `RestTsBaseline` — 2 đầu độc lập (har_head, news_head) |
| `code/train_resttext.py` | Train loop custom (loss 2 phần, residual detach) |
| `test/test_smoke.py` | forward shape, backward (cả 2 head), residual-target không có gradient chảy ngược qua har_pred |

## Isolation

Đọc `data/sentiment_embedding/` (có sẵn). Không sửa `src/`, không sửa baseline `2026-07-07`.
Output `results/resttext_<ts>/`, `models/resttext_<ts>/`.
