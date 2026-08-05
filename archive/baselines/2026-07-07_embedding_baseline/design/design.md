# Design — Embedding Baseline

**Baseline:** `2026-07-07_embedding_baseline`
**Tham chiếu:** `docs/project/SENTIMENT_NEWS_EMBEDDIMENT_ARCHITECTURE.md` (= `SENTIMENT_NEWS_EMBEDDING_ARCHITECTURE.md`), `docs/project/SENTIMENT_LATENT_SPACE_TECHNIQUES.md`
**Rule:** `CLAUDE.md` mục 3.F

---

## 1. Quyết định design (sau check ChatGPT + phân tích param)

| Quyết định | Chọn | Lý do |
|---|---|---|
| Encoder PhoBERT | **frozen, offline** | ~12-21K title VN → overfit nếu unfreeze (FNSPID chỉ unfreeze với 15M) |
| Projection 768→d | **d_news=64** (không 128) | Ngang HAR branch (64); params giảm 2× vs 128 |
| Projection type | **PCA offline (default)**, Linear (option) | PCA = 0 trainable param → triệt tiêu overfit-risk của projection với data nhỏ |
| Vị trí project | **trước attention pooling** | Pool trên 64-d rẻ hơn 768-d |
| Article aggregation | **ArticleSetAttentionPooling** | Permutation-invariant, handle 0-news (learned token) |
| Temporal | 1-L LSTM over 22 ngày (vectorized, không per-stock loop) | Match HAR temporal approach |
| HAR branch | **reuse `ParallelLSTMGNN.get_embeddings`** (read-only) | Không duplicate ~60 dòng; proven 69.98% DirAcc |
| Fusion | **late concat** [64+256+64] → MLP 320→...→1 | Simplicity; defer MSGCA cross-attn (Bước 3) |

**Param budget (trainable, nhánh news):**
- PCA projection: **0** (offline)
- ArticleSetAttentionPooling: query(64) + no_news_token(64) = **128**
- News temporal LSTM(64→64): ~33K (4×(64×64+64×64+biases))
- Fusion MLP tăng nhẹ do +64 input
- → nhánh news ~33K trainable (vs Linear-projection variant ~82K). Phù hợp ~953 train examples.

## 2. Data flow

```
[OFFLINE — 1 lần]
unified_articles.csv
   → ticker extraction (\b TICKER \b trên title+lead)
   → PhoBERT frozen → CLS [768] per article
   → PCA fit trên TRAIN articles → 768→64
   → cache: data/sentiment_embedding/{TICKER}_emb.npz  (date → [n_articles, 64])

[ONLINE — training]
__getitem__:
  x_har  [22, 30, 3]      ← from {TICKER}_processed.csv (HAR)
  x_emb  [22, 30, MAX_ARTICLES=5, 64]   ← from {TICKER}_emb.npz (pad+mask)
  mask   [22, 30, 5]
  adj    [30, 30]          ← k-NN graph
  y      [30]              ← parkinson T+5

forward:
  h_lstm, h_gnn = ParallelLSTMGNN.get_embeddings(x_har, adj)   # [B,30,64], [B,30,256]
  daily = ArticleSetAttentionPooling(x_emb, mask)                # [B,30,22,64]
  news_rep = NewsTemporalLSTM(daily)                             # [B,30,64]
  h = concat([h_lstm, h_gnn, news_rep])                          # [B,30,384]
  pred = MLP(h)                                                  # [B,30]
```

## 3. File list (code/)

| File | Trách nhiệm |
|---|---|
| `extract_embeddings.py` | Offline: PhoBERT frozen → CLS 768 → PCA → cache `.npz`. CLI: `--scorer_model`, `--dim 64`, `--use_pca` |
| `dataset_embedding.py` | Subclass; `__getitem__` trả (x_har, adj, x_emb, mask, y). Pad article → MAX_ARTICLES |
| `model_embedding.py` | `ArticleSetAttentionPooling` + `EmbeddingBaseline` (HAR reuse + news branch + concat MLP) |
| `train_embedding_baseline.py` | Reuse `train_epoch`/`validate`/`EarlyStopping` pattern; handle 5-tuple batch; save results |

## 4. Critical correctness points

1. **PCA fit trên TRAIN articles ONLY** (không leakage). Transform val/test với PCA đã fit.
2. **MAX_ARTICLES=5 cap**: ngày >5 bài → giữ 5 đầu (caveat: mất bài; chấp nhận vì đa số 0-2 bài).
3. **Mask đúng**: mask=0 → attention score = -1e9 → không đóng góp. 0 bài thật → output = `no_news_token`.
4. **Windowing chống leakage**: trong snapshot kết thúc ngày T, chỉ dùng tin date ≤ T.
5. **Target normalization**: theo LSTM-HAR Enhanced (StandardScaler + linear output + inverse_transform).
6. **Inference dùng μ/news deterministic** (không sample — không có VAE ở MVP này).

## 5. Test plan (test/)

| Test | Check |
|---|---|
| `test_attention_pooling.py` | (a) permutation invariance; (b) 0-news → no_news_token; (c) output shape; (d) mask correctness |
| `test_smoke.py` | Full model forward với dummy x_emb/mask → đúng shape [B,30]; chạy được backward |
