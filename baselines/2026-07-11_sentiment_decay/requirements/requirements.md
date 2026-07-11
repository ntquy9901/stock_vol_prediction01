# Requirements — Sentiment Decay Baseline

**Baseline:** `2026-07-11_sentiment_decay`
**Ngày:** 11/07/2026
**Tham chiếu:** `docs/suggestion/XU_LY_NEWS_THUA.md` Method 1 (Sentiment Decay State); `sentiment-sparsity-solution-2026-06-29.md` Solution 2
**Rule:** CLAUDE.md §3.F

## 1. Mục tiêu

Test Method 1 (Sentiment Decay State): sentiment **carry-forward + decay** khi không có tin, thay vì neutral-fill (0). Xem có vượt sentiment baseline hiện tại + HAR không.

`s_t = mask_t · score_t + (1 - mask_t) · s_{t-1} · decay`  (decay=0.9)

## 2. Input / Output

**Input (read-only):** `data/sentiment_baseline/{TICKER}_sentiment.csv` (lexicon sentiment_1d, news_count_1d).
**Output (cô lập):**
- `data/sentiment_decay/{TICKER}_sentiment.csv` — cùng schema (sentiment_1d = decayed state, news_count_1d = mask) → reuse pipeline `src/sentiment_baseline/`
- `results/sentiment_decay_*` (metrics), `models/sentiment_decay_*` (ckpt)

## 3. Success Criteria

| # | Tiêu chí | Verify |
|---|---|---|
| 1 | Decay computation đúng (reset/decay) | unit test |
| 2 | Pipeline chạy end-to-end | train không lỗi |
| 3 | **Go/no-go:** val DirAcc **decay > sentiment_baseline** (68.57% lexicon) @ matched epoch | so results |

**Kỳ vọng trung thực (LOW):** market_fallback (phiên bản rich hơn, học được) đã no-lift → decay (fixed, đơn giản hơn) **rất khó lift**. Đây là **closure test** — xác nhận carry-forward signal yếu không giúp.

## 4. Scope

**Làm:** compute_decay.py (preprocess) + reuse `src/sentiment_baseline/` pipeline (dataset + train) read-only.
**KHÔNG làm:** learnable decay, sector propagation (đã = market_fallback), new model architecture.

## 5. Isolation

- KHÔNG sửa `src/sentiment_baseline/`, baseline cũ, hay data nguồn.
- Decay files ra `data/sentiment_decay/` (riêng). Train qua `--sentiment_dir data/sentiment_decay`.
