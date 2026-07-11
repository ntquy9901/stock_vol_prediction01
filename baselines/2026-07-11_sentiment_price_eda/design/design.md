# Design — Sentiment ↔ Price EDA (2026-07-11)

## Kiến trúc
Script đơn `code/sentiment_price_eda.py`, tự bootstrap `sys.path` (folder có dấu `-` không import bằng `python -m`). Chạy: `python baselines/2026-07-11_sentiment_price_eda/code/sentiment_price_eda.py`.

## Data flow
1. **Load** 30 mã OHLCV + sentiment_baseline, merge trên `date` (inner join, cùng trading-day index).
2. **Forward returns**: `ret_kd[T] = close[T+k] / close[T] - 1` cho k ∈ {1,2,3,5,10}. Winsorize 1/99%.
3. **Event = ngày có `news_count_1d > 0`**. Nhóm theo `sentiment_1d`: `pos` (>+0.2), `neg` (<−0.2), `neu` (giữa). Threshold là tham số, report sensitivity.
4. **Event study (pooled)**: mean/median forward return theo nhóm × horizon.
5. **Statistical tests**:
   - Mann-Whitney U (rank-based, robust outlier): `pos` vs `neg` mỗi horizon.
   - Wilcoxon one-sample: mean return nhóm `pos` ≠ 0, `neg` ≠ 0.
6. **Per-ticker**: Spearman corr(sentiment_1d, ret_kd) trên event days; đếm mã có signal cùng chiều (pos→dương, neg→âm).
7. **H5**: lặp lại với parkinson_volatility forward-change, so corr.

## Quyết định design
- **Nguồn sentiment**: `sentiment_baseline` (raw + titles, dễ spot-check). Decay variants chỉ khác ở ngày không-tin → không ảnh hưởng event study.
- **Robust statistics**: rank-based + winsorize vì giá có thể chưa adjust split.
- **Pooled + per-ticker**: pooled cho power, per-ticker cho heterogeneity (H4).
- **Temporal**: đây là EDA thuần (không train), không vi phạm temporal split. Nếu sau build model vẫn giữ rule §3.A.

## Files
- `code/sentiment_price_eda.py` — toàn bộ logic.
- `code/__init__.py`, `test/__init__.py` — package markers.
- `test/test_smoke.py` — shape correctness + property test (dummy data).
