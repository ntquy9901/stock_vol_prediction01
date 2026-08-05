# Design — Latent Noise Injection Baseline

**Baseline:** `2026-07-11_latent_noise` · **Ngày:** 11/07/2026

## 1. Quyết định design

**Reuse bằng subclass, không duplicate.** `LatentNoiseBaseline` kế thừa `EmbeddingBaseline`
(đã có HAR branch + news branch + fusion). Override `forward` để chèn 1 dòng noise. Lý do:
- Embedding Baseline đã review + test + chạy thực (68.76%). Duplicate ~130 dòng = rủi ro bug mới.
- Subclass = cô lập cứng (không sửa embedding baseline), chỉ thêm 1 hook.

**Inject vào `news_rep` (sau NewsTemporalEncoder, trước concat).** Lý do:
- Gợi ý thầy = "trường hợp tin thưa" → nhánh news là đích. HAR branch đã mạnh (69.98%), noise vào đó
  có thể hại hơn lợi.
- `news_rep [B,30,64]` là representation cuối của news trước fusion → noise đây = "làm mờ" signal news,
  ép model dùng nó như gợi ý thay vì phụ thuộc cứng.

**Noise Gauss cộng, σ cố định, eval tắt.** Tier A đơn giản nhất: `z + σ·ε`, không reparameterization
(không học σ). Đó là Tier B (VIB). Eval tắt để validate/test deterministic + công bằng.

## 2. Data flow

```
x_har[B,22,30,3] adj[B,30,30]  x_emb[B,22,30,10,64] mask[B,22,30,10]
        │                              │
   har.get_embeddings            news_pool → news_temporal
   h_lstm[B,30,64] h_gnn[B,30,256]   news_rep[B,30,64]
        │                              │
        │                   ┌──────────┴──────────┐
        │                   │ if training & σ>0:  │
        │                   │   news_rep += σ·ε   │   ← Tier A
        │                   └──────────┬──────────┘
        └──► concat [B,30,64+256+64=384] ◄──┘
                          │
                     fusion MLP → ŷ[B,30]
```

## 3. File list

| File | Mục đích |
|------|----------|
| `code/model_latent_noise.py` | `LatentNoiseBaseline(EmbeddingBaseline)` + `build_default_model` |
| `code/train_latent_noise.py` | Train 5 epoch, reuse `create_embedding_dataloaders` + train/validate pattern, `--noise_std` |
| `test/test_latent_noise.py` | pytest: shape, noise-OFF-in-eval (determinism), noise-ON-in-train |

## 4. Isolation

- Không sửa `src/`, không sửa `baselines/2026-07-07_embedding_baseline/`. Import read-only:
  add `baselines/2026-07-07_embedding_baseline/code` vào `sys.path` rồi `from model_embedding import EmbeddingBaseline`.
- Output: `results/latent_noise_<ts>/`, `models/latent_noise_<ts>/` (theo §3.D, không tạo trong folder baseline).

## 5. Hyper tham số (mặc định)

- `noise_std=0.1`, `epochs=5`, `batch_size=32`, `lr=5e-3`, `weight_decay=1e-5` (§3.E),
  `d_news=64`, `dropout=0.5`, `graph_method=knn` — khớp embedding baseline để so sánh công bằng.
