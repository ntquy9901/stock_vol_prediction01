# Design (Plan) — Gated Cross-Attention Baseline

## Data flow

```
data/sentiment_embedding/{TICKER}_emb.npz  (CÓ SẴN, tái dùng)
                    │
     create_embedding_dataloaders (sibling, KHÔNG sửa)
                    │
   x_har,adj,x_emb,mask,y
                    │
     GatedCrossAttnBaseline (model MỚI)
     h_lstm,h_gnn = HAR.get_embeddings(x_har,adj)
     har_embed = concat[h_lstm,h_gnn]                    # [B,S,320]  (query)
     news_rep  = news_temporal(news_pool(x_emb,mask))     # [B,S,64]   (key/value)
     attended, _ = MultiheadAttention(query=proj_q(har_embed), key=proj_kv(news_rep),
                                       value=proj_kv(news_rep))     # [B,S,64]
     gate = sigmoid(gate_mlp(concat[har_embed, attended]))          # [B,S,1] — HỌC ĐƯỢC
     fused = concat[har_embed, gate * attended]                     # [B,S,384]
     pred = fusion_mlp(fused)
                    │
     train_gated_crossattn.py (loop chuẩn, 1 loss MSE — không cần loss phụ)
```

## File list

| File | Mục đích |
|---|---|
| `code/model_gated_crossattn.py` | `GatedCrossAttnBaseline` — cross-attention + gate học được |
| `code/train_gated_crossattn.py` | Train loop (chuẩn, giống EmbeddingBaseline gốc, chỉ khác model) |
| `test/test_smoke.py` | forward shape, gate trong [0,1], gate→0 khi news toàn mask (an toàn) |

## Isolation

Đọc `data/sentiment_embedding/` (có sẵn). Không sửa `src/`, không sửa baseline `2026-07-07`.
Output `results/gated_crossattn_<ts>/`, `models/gated_crossattn_<ts>/`.

## Hyperparameter

`num_heads=4`, `d_attn=64` (khớp d_news).
