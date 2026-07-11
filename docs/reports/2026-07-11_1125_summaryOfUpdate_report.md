# Summary Report — Sentiment ↔ Price Relationship EDA

**Date:** 2026-07-11
**Task:** Phân tích mối quan hệ giữa dữ liệu sentiment và giá (volatility) 30 cổ phiếu VN30.
**Verdict:** **NO-GO** — không có bằng chứng thống kê vững chắc rằng sentiment tin tức dự báo được hướng giá.

---

## 1. Câu hỏi gốc

> *Ngày T có sentiment tốt → T+1/T+5 giá lên? Sentiment xấu → giá xuống?*

Đã formal hóa thành H1–H5 và kiểm định bằng event study + lead-lag correlation.

## 2. Dữ liệu thực tế dùng

| Nguồn | Vai trò | Ghi chú |
|-------|---------|---------|
| `data/raw/prices/<TICKER>_ohlcv.csv` | Giá OHLCV (raw price) | 30 mã, 2006-11-21 → 2026-06-09. Dùng để tính forward return. |
| `data/sentiment_baseline/<TICKER>_sentiment.csv` | Sentiment | `sentiment_1d, news_count_1d`. **Rất thưa: ~63 event/mã (~2%)**. |
| `data/processed/vn30_only/<TICKER>_processed.csv` | Parkinson volatility | Dùng cho so sánh H5. |

**Lưu ý quan trọng:** dữ liệu `processed` là *volatility* (Parkinson), không phải giá raw. Câu hỏi "giá lên/xuống" cần OHLCV → đã dùng `raw/prices`.

## 3. Kết quả chính (trên 1851 event, 29 mã; SSB bị skip do quá ít event)

**Phân bố nhóm:** pos = **1044**, neu = **784**, neg = **23**.
→ Sentiment **lệch dương cực mạnh**. Chỉ 23 event tiêu cực, đến từ 14 mã (6 mã có ≥2: GAS/MWG/VIC mỗi mã 3). → Không đủ power cho H2.

### Trả lời H1–H5

| # | Câu hỏi | Kết quả |
|---|---------|---------|
| **H1** | Sentiment tích cực → giá lên? | **Yếu, không ý nghĩa.** Mean return nhóm pos dương (22→106 bp theo T+1→T+10) nhưng đây có thể là **drift thị trường chung**, không phải sentiment. Mann-Whitney pos vs neg: p_raw & p_demeaned **đều > 0.54** ở mọi horizon. |
| **H2** | Sentiment tiêu cực → giá xuống? | **KHÔNG.** Nhóm neg thực tế có return **cao hơn** pos (46→170 bp) — ngược kỳ vọng! Nhưng n=23, underpowered, **không ý nghĩa** (p>0.47). Không kết luận được. |
| **H3** | Horizon nào mạnh nhất? | **Không có** horizon nổi bật. Corr ≈ phẳng ~0.13 ở mọi k. |
| **H4** | Khác biệt giữa các mã? | **Yếu.** 21/29 mã có corr dương (HDB 0.41, PDR 0.39 mạnh nhất) nhưng |corr| phần lớn <0.3, n nhỏ → không ý nghĩa từng mã. |
| **H5** | Sentiment → return hay → volatility mạnh hơn? | **Return hơi mạnh hơn** volatility: mean \|ρ\| T+1 = 0.139 (ret) vs 0.127 (vol); T+5 = 0.134 vs 0.095. Cả hai đều yếu. |

### Go/No-Go (Bonferroni α=0.01, primary horizon T+5)
- Không có horizon nào đạt p < 0.01.
- Không có horizon nào có |spread pos−neg| ≥ 30 bp và ý nghĩa.
- **→ NO-GO.**

## 4. Figures & outputs
`results/2026-07-11_sentiment_price_eda/`:
- `fig_sentiment_distribution.png` — phân bố sentiment (thấy rõ lệch dương).
- `fig_mean_return_by_group.png` — mean forward return theo nhóm × horizon.
- `fig_per_ticker_corr_ret5d.png` — corr từng mã.
- `fig_lag_correlation.png` — box corr theo horizon.
- `fig_vol_vs_ret_corr.png` — ret vs vol.
- `events_all.csv`, `per_ticker_stats.csv`, `summary.json`.

## 5. Tại sao NO-GO — 3 nguyên nhân (trung thực)

1. **Dữ liệu sentiment quá thưa & lệch**: ~2% ngày có tin, gần như toàn tích cực. Thiếu mẫu tiêu cực → không thể kiểm định "tin xấu → giá xuống".
2. **Không điều chỉnh drift thị trường**: forward return chưa trừ benchmark thị trường → mean return dương có thể chỉ là xu hướng tăng chung của VN30 2006–2026, không phải tác động sentiment.
3. **Confounding theo mã**: neg events tập trung vài mã (GAS/MWG/VIC) → khác biệt return có thể là đặc thù mã, không phải sentiment.

## 6. Code review
Adversarial review (12 findings: 3 HIGH, 5 MED, 4 LOW) — **tất cả HIGH/MEDIUM đã fix hoặc tài liệu hóa trung thực**. Chi tiết: `baselines/2026-07-11_sentiment_price_eda/code_review/code_review_2026-07-11.md`. Test: **10/10 pass**.

## 7. Khuyến nghị (follow-up, chưa làm)

Nếu muốn kiểm tra kỹ hơn trước khi bỏ hẳn sentiment:
1. **Event study có abnormal return**: trừ return thị trường (VN-Index hoặc basket VN30) → loại drift. Đây là cải thiện quan trọng nhất.
2. **Mở rộng mẫu tin xấu**: crawl thêm tin historic hoặc dùng cửa sổ sentiment "đặc" (June 2026) — hiện quá ngắn cho T+5.
3. **Dùng sentiment làm auxiliary/noise feature** thay vì primary directional signal (vì corr ~0.13 vẫn có chút giá trị dự báo volatility, phù hợp bài toán volatility forecasting của project).

## 8. Files thay đổi (tất cả NEW, cô lập)
```
baselines/2026-07-11_sentiment_price_eda/
├── requirements/requirements.md
├── design/design.md
├── code/sentiment_price_eda.py
├── code_review/code_review_2026-07-11.md
└── test/test_smoke.py
results/2026-07-11_sentiment_price_eda/        (outputs)
docs/reports/2026-07-11_1125_summaryOfUpdate_report.md   (file này)
```
Không sửa file nào của baseline khác hay `src/` chung (hard isolation per §3.F).

## 9. Commands run (thật)
- `python baselines/2026-07-11_sentiment_price_eda/code/sentiment_price_eda.py` → NO-GO.
- `python -m pytest baselines/2026-07-11_sentiment_price_eda/test/ -v` → 10 passed.
- Coverage/diff-cover/ruff: **Not run** (chưa cài trong môi trường — tooling gap đã ghi trong CLAUDE.md).

## 10. DoD checklist
- [x] Code thỏa mãn request; không refactor không liên quan.
- [x] Surgical, cô lập.
- [x] Code review adversarial chạy + xử lý findings.
- [x] Smoke test pass (10/10).
- [x] Summary report (file này).
- [x] Impact analysis: thay đổi additive (folder mới), không ảnh hưởng baseline/src hiện có.
- [ ] Diff-coverage ≥80%: Not run (tooling gap).
- [ ] Lint: Not run (tooling gap).

---

# PHẦN BỔ SUNG — Phân tích mức THỊ TRƯỜNG (market-level)

**Câu hỏi:** Gom sentiment theo ngày thị trường (pool 30 mã) thì tương quan với **độ biến động** và **giá** thị trường thế nào?

**Code:** `baselines/2026-07-11_sentiment_price_eda/code/sentiment_market_eda.py` (tái dùng helpers script trước). Test: `test/test_market_smoke.py` (4 test). Outputs: `results/2026-07-11_sentiment_price_eda/market/`.

## Thiết kế
- **Market sentiment/day** (pool 30 mã): mean sentiment, **news_count** (attention), neg_ratio, dispersion.
- **Market basket** equal-weighted 30 mã: forward return T→T+k, |return|, avg Parkinson vol, và **forward realized vol** (std trả qua [T+1..T+22] — predictive).
- Lag-Spearman corr + event study (ngày có tin vs không) + detrend (first-difference để loại trend 2010-2025).
- Panel: 4889 ngày giao dịch, 1197 ngày có tin (24.5%).

## Kết quả

### 1. Tương quan sentiment → hướng giá thị trường: ≈ 0
| Measure | ret_1d | ret_5d |
|---------|--------|--------|
| news_count | 0.002 | 0.013 |
| sent_mean | 0.028 | 0.009 |
| neg_ratio | 0.000 | 0.044 |
Detrended (first-diff) sent_mean vs ret: 0.06–0.07 → **vẫn gần 0 sau khi loại trend**. → Sentiment không dự báo hướng giá thị trường.

### 2. Attention → forward volatility: **yếu và ÂM** (ngược kỳ vọng) ⭐
- corr(news_count, fwd_vol_22d) = **−0.093** (mọi lag).
- Event study: ngày có tin → forward vol tiếp theo **thấp hơn 13.5%** (ratio **0.865**, **p=1e-11**); |return| thấp hơn 13.6% (p=0.00014).
- → Nhiều tin đi kèm vol THẤP hơn, không cao. Ngược với "attention → uncertainty" kinh điển.

**Giải thích khả dĩ:** corpus tin dự án chủ yếu là **công bố định kỳ** ("Báo cáo cập nhật lợi nhuận", báo cáo phân tích) → **giải tỏa bất định** (info resolution) chứ không phải tín hiệu biến động. Hoặc tin được thu thập nhiều ở giai đoạn thị trường yên.

### 3. Sentiment sign → return thị trường: không (n_neg=14 ngày, quá ít).

## Verdict market-level
**Tương quan thực tế ≈ 0 ở mọi hướng dự báo.** Phát hiện duy nhất robust (p=1e-11) là *attention ↔ forward vol thấp hơn*, nhưng (a) magnitude yếu, (b) ngược kỳ vọng, (c) khả năng lớn là artifact của loại tin (định kỳ), không phải signal sentiment thật. **→ Sentiment vẫn không nên làm feature dự báo giá/vol chính.**

## Lưu ý method (đã fix trong review)
- Bản đầu dùng realized vol **trailing** (nhìn lùi) làm target "dự báo" → review bắt (HIGH), sửa sang **forward vol**. Kết quả direction **vẫn giữ** (robust).
- 80 correlations không hiệu chỉnh đa phép thử → treat as exploratory (caveat).
- Basket equal-weighted, composition drift theo thời gian (caveat).

## Commands run (market)
- `python .../sentiment_market_eda.py` → outputs ở `market/`.
- `python -m pytest .../test/ -v` → **14/14 pass**.

---

# PHẦN BỔ SUNG 2 — Phân tích theo LOẠI TIN (news-type) + SỬA LỖI MARKET

## ⚠️ Sửa lỗi quan trọng về kết quả market
Phát biểu trước đây ("ngày có tin → forward vol thấp hơn 13%, p=1e-11") **phần lớn là artifact coverage**. Ngày có tin tập trung giai đoạn post-2018 (yên hơn); ngày không tin tập trung pre-2018. Khi **match theo năm** (so sánh news vs no-news trong cùng năm), hiệu ứng **biến mất**:

| Loại tin | pooled ratio | **year-matched median** |
|----------|-------------|------------------------|
| all_news | 0.865 | **1.034** |
| rating_POS | 0.880 | **1.013** |
| earnings_update | 0.816 | **0.943** (residual nhẹ nhất) |

→ **Sau khi kiểm soát thời kỳ, news KHÔNG dự báo volatility.** (Bắt được nhờ adversarial review.)

## Phân tích news-type
**Code:** `code/sentiment_newstype_eda.py`, test `test/test_newstype_smoke.py` (8 test). Outputs: `results/.../newstype/`. Phân loại 1851 title → rating direction + type bằng keyword.

### Thành phần corpus
- **89% định kỳ**: khuyến nghị analyst (POS=971, NEU=292, NEG=**7**) + earnings updates (182).
- **event/shock: 6 tin (0.3%)** → không đủ test "tin sự kiện → vol lên".

### Kết quả
1. **Rating direction → return:** không testable. NEG=7 (quá ít + artifact mean-reversion: sell rating thường ban hành sau khi giá rớt). POS→return dương (24-60 bp) nhưng khả năng drift; NEU→âm nhẹ. Không ý nghĩa thống kê.
2. **Loại tin → vol:** sau year-match, mọi loại ≈ 1.0 → không loại tin nào dự báo vol. Chỉ earnings-update còn residual nhẹ (~6%).
3. **Không có tin sự kiện đủ** để test giả thuyết "event → vol lên".

## Verdict cuối cùng (cả 3 phân tích)
**Sentiment/news trong dataset này KHÔNG mang tín hiệu dự báo** cho hướng giá hay volatility. Phát biểu "attention→vol thấp hơn" đã bị bác bởi year-matching. Corpus chủ yếu là tin analyst/earnings định kỳ, không có tin sự kiện.

→ **Khuyến nghị: không dùng sentiment/news làm feature dự báo giá/vol.** Nếu vẫn muốn khai thác, cần nguồn tin khác (tin sự kiện thật, không phải analyst coverage) hoặc làm auxiliary noise feature yếu.

## Commands run (news-type)
- `python .../sentiment_newstype_eda.py` → outputs `newstype/`.
- `python -m pytest .../test/` → **24/24 pass**.


