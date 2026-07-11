# Design — Sentiment Decay Baseline

**Baseline:** `2026-07-11_sentiment_decay`
**Tham chiếu:** `docs/suggestion/XU_LY_NEWS_THUA.md` Method 1

## Quyết định design

| Quyết định | Chọn | Lý do |
|---|---|---|
| Decay formula | `s_t = mask·score + (1-mask)·s_{t-1}·decay` | đúng Method 1 |
| decay_rate | 0.9 (fixed) | MVP; learnable chỉ nếu fixed có hint |
| Sentiment source | lexicon (`data/sentiment_baseline/`) | consistent vs sentiment baseline để so công bằng |
| Output schema | **cùng sentiment_baseline** (date, sentiment_1d, news_count_1d) | reuse pipeline `src/sentiment_baseline/` read-only, 0 duplication |
| Model | reuse `ParallelLSTMGNN` (5 feat: HAR + sentiment_decay + mask) | giống sentiment baseline |
| Train | reuse `src/sentiment_baseline/train_sentiment_baseline.py --sentiment_dir` | 0 code train mới |

## Data flow

```
data/sentiment_baseline/{TICKER}_sentiment.csv  (lexicon sentiment_1d, news_count_1d)
   ↓ compute_decay.py (s_t = mask·score + (1-mask)·s_{t-1}·0.9, per stock chronological)
data/sentiment_decay/{TICKER}_sentiment.csv  (sentiment_1d=decayed, news_count_1d=mask)
   ↓ src/sentiment_baseline/ pipeline (--sentiment_dir data/sentiment_decay)
ParallelLSTMGNN (5 feat) → train @ 40ep → results/sentiment_decay_*
```

## Files

| File | Trách nhiệm |
|---|---|
| `code/compute_decay.py` | decay state per stock → `data/sentiment_decay/` (same schema) |
| `test/test_decay.py` | decay correctness (reset/decay/reset) |
| (reuse) `src/sentiment_baseline/dataset_sentiment.py` | 5-feat dataset (reads decay dir) |
| (reuse) `src/sentiment_baseline/train_sentiment_baseline.py` | training loop |

## Critical correctness

1. Decay tính PER STOCK, chronological (s_t phụ thuộc s_{t-1} toàn timeline, không per-window).
2. mask = news_count_1d > 0. Score = sentiment_1d (lexicon, đã có).
3. Decay files cùng schema → dataset không cần đổi.
4. Decay only carries signal tới ~10 ngày (0.9^10 ≈ 0.35) — phù hợp window 22 ngày.

## Honest pre-assessment

- market_fallback (propagation học được, dense) → no-lift (68.69%).
- Decay (carry-forward fixed, scalar) = phiên bản ĐƠN GIẢN HƠN → **rất khó làm tốt hơn market_fallback**.
- Đây là closure test cho họ "carry-forward / propagate signal yếu". Nếu no-lift → chốt: news-handling không cứu được no-signal.
