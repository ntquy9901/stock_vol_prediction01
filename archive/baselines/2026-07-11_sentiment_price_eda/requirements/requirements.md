# Requirements — Sentiment ↔ Price EDA (2026-07-11)

## Mục tiêu
Xác định có hay không có mối quan hệ giữa **sentiment tin tức** và **biến động giá** (và volatility) của 30 cổ phiếu VN30, để quyết định **go/no-go** cho việc đưa sentiment làm feature vào model dự báo.

## Câu hỏi nghiên cứu (H1–H5)
- **H1**: Ngày T sentiment tích cực → forward return T+1/T+5 dương đáng kể?
- **H2**: Ngày T sentiment tiêu cực → forward return T+1/T+5 âm đáng kể?
- **H3**: Độ trễ nào (T+1, T+2, T+3, T+5, T+10) có signal mạnh nhất?
- **H4**: Signal khác biệt giữa các mã có không?
- **H5**: Sentiment tương quan với **volatility** mạnh hơn hay với **return** mạnh hơn?

## Input
- Giá: `data/raw/prices/<TICKER>_ohlcv.csv` (open/high/low/close/volume, 2006-11-21 → 2026-06-09, 30 mã).
- Sentiment: `data/sentiment_baseline/<TICKER>_sentiment.csv` (`date, sentiment_1d, news_count_1d, news_titles`). ~109 ngày có tin/mã.
- Volatility: `data/processed/vn30_only/<TICKER>_processed.csv` (parkinson_volatility).

## Output
- `results/2026-07-11_sentiment_price_eda/`: `events_all.csv`, `per_ticker_stats.csv`, `summary.json`, `fig_*.png`.
- Report: `docs/reports/<ts>_summaryOfUpdate_report.md`.

## Success criteria / Go-No-Go
- **Go (sentiment có giá trị)**: ít nhất 1 horizon có |mean return pos−neg| ≥ 0.3% VÀ p-value (Mann-Whitney) < 0.05 ở cả pooled và ≥1/3 số mã.
- **No-go**: không nhóm/horizon nào đạt ý nghĩa thống kê → sentiment không nên là feature chính (chỉ dùng làm noise/auxiliary).

## Giới hạn đã biết
- Tin rất thưa (~2.2% ngày giao dịch) → phân tích xoay quanh event days.
- Sentiment lệch dương (median ~0.5) → nhóm negative ít sample, sức mạnh test cho H2 hạn chế.
- Giá OHLCV có thể chưa adjust stock split → forward return winsorize 1/99% + flag outlier.
