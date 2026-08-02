# Design (Plan) — Alignment-Loss Baseline

## Data flow

```
data/sentiment_embedding/{TICKER}_emb.npz  (CÓ SẴN, tái dùng)
                    │
     create_embedding_dataloaders (sibling, KHÔNG sửa)
                    │
   x_har,adj,x_emb,mask,y  (giống EmbeddingBaseline gốc)
                    │
     AlignmentLossBaseline (model MỚI, kiến trúc dự báo GIỐNG EmbeddingBaseline)
     h_lstm,h_gnn = HAR.get_embeddings(x_har,adj)
     news_rep = news_temporal(news_pool(x_emb,mask))
     pred = fusion(concat[h_lstm,h_gnn,news_rep])        # giống hệt EmbeddingBaseline
     # + 2 nhánh projection PHỤ (chỉ dùng lúc train, không ảnh hưởng pred)
     proj_har  = align_har(concat[h_lstm,h_gnn])          # [B,S,32]
     proj_news = align_news(news_rep)                     # [B,S,32]
                    │
     train_alignment.py (loop MỚI)
     loss_pred  = MSE(pred, y)
     align_loss = 1 - cosine_similarity(proj_har, proj_news, dim=-1).mean()
     loss = loss_pred + 0.1 * align_loss
```

## File list

| File | Mục đích |
|---|---|
| `code/model_alignment.py` | `AlignmentLossBaseline` — fusion giống gốc + 2 projection head phụ |
| `code/train_alignment.py` | Train loop custom (loss_pred + lambda*align_loss) |
| `test/test_smoke.py` | forward shape, align_loss trong [0,2], gradient chảy vào cả 2 projection head |

## Isolation

Đọc `data/sentiment_embedding/` (có sẵn). Không sửa `src/`, không sửa baseline `2026-07-07`.
Output `results/alignment_<ts>/`, `models/alignment_<ts>/`.

## Hyperparameter

`d_align=32`, `lambda_align=0.1` (mặc định, không tune — thí nghiệm đầu tiên).
