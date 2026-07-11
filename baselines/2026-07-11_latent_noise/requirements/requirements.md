# Requirements — Latent Noise Injection Baseline

**Baseline:** `2026-07-11_latent_noise`
**Ngày:** 11/07/2026
**Tham chiếu:** `docs/project/SENTIMENT_LATENT_SPACE_TECHNIQUES.md` Tier A (Latent Noise Injection); gợi ý thầy "thêm vector phát sinh ngẫu nhiên cho trường hợp tin thưa".
**Rule:** CLAUDE.md §3.F (cô lập, 5 sub-folder), §3.E (chống overfit), Training Policy (experiment ≤10 epoch).

## 1. Mục tiêu

Test Tier A — **Latent Noise Injection** trên nhánh news của Embedding Baseline: thêm nhiễu Gauss
`z' = z + σ·ε` (ε~N(0,1)) vào news representation trong lúc train, tắt ở eval. Mục đích: ép model
không quá phụ thuộc vài ngày có tin (anti-overfit trên dữ liệu semantic thưa — 94.5% ngày-mã mù tin).

Không đổi loss (MSE), không thêm KL term (đó là Tier B / VIB — defer).

## 2. Input / Output

**Input (read-only, reuse):**
- `data/processed/` — Parkinson volatility 30+ mã (HAR 3 features).
- `data/sentiment_embedding/{TICKER}_emb.npz` — PhoBERT→PCA64 cache (đã có từ embedding baseline).
- Reuse class `EmbeddingBaseline` + `create_embedding_dataloaders` từ `baselines/2026-07-07_embedding_baseline/code/`.

**Output (cô lập):**
- `results/latent_noise_<ts>/results.json` — val/test metrics (6 mandatory).
- `models/latent_noise_<ts>/best.pt` — checkpoint.

## 3. Kiến trúc (Tier A)

`LatentNoiseBaseline(EmbeddingBaseline)` — subclass, override `forward`:
```
h_lstm, h_gnn = har.get_embeddings(x_har, adj)     # HAR branch, giữ nguyên
daily    = news_pool(x_emb, mask)                   # news branch, giữ nguyên
news_rep = news_temporal(daily)                     # [B,30,64]
if self.training and noise_std > 0:
    news_rep = news_rep + noise_std * randn_like(news_rep)   # ← TIÊU ĐIỂM
h = concat([h_lstm, h_gnn, news_rep])               # [B,30,384]
return fusion(h)                                     # [B,30]
```
- `noise_std` default 0.1 (arg `--noise_std`).
- Eval mode (`model.eval()`) → không noise → deterministic → validate/test công bằng.

## 4. Success Criteria / Go-No-Go

| # | Tiêu chí | Verify |
|---|----------|--------|
| 1 | Noise chỉ bật train mode, tắt eval | unit test (2 forward eval = nhau) |
| 2 | Shape output `[B, num_stocks]` | unit test |
| 3 | Pipeline chạy end-to-end 5 epoch | train không lỗi, in metrics |
| 4 | **Go/no-go:** test DirAcc **latent_noise > embedding baseline 68.76%** @ matched epoch | so results.json |

**Kỳ vọng trung thực (LOW lift):** EDA (§2.4.5) đã chứng minh signal sentiment yếu (corr ~0.13).
Latent noise là regularizer — chỉ giúp nếu signal đã có. Có khả năng no-lift hoặc hơi tệ (noise làm
khó học signal yếu). Đây là **closure test** giống decay (§2.4.4).

## 5. Giới hạn đã biết
- 5 epoch (Training Policy — experiment). Không hội tụ hoàn toàn → so sánh thô, không kết luận dứt khoát.
- σ=0.1 cố định, chưa tune.
- Chỉ noise trên news_rep (không noise HAR latent) — theo gợi ý "tin thưa".
