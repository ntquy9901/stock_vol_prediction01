# Summary — Calendar-Augmented News Gate Baseline (Cách 1: calendar feature vào x_news)

**What changed:** thêm baseline mới `baselines/2026-08-01_calendar_news_gate_baseline/` — nối 10
cột calendar feature (day-of-week, month cyclical, Tết proximity, month/quarter-end, earnings-
window proxy) vào `x_news` của kiến trúc `per_ticker_news_gate_baseline` hiện có, KHÔNG đổi model
(tái dùng `PerTickerGatedNewsBaseline` không sửa, n_feat 146→156). Đây là "Cách 1" đã thống nhất
với user hôm nay (2026-08-01), thử nghiệm giả thuyết: tin tức có ảnh hưởng khác nhau theo mùa
BCTC/Tết hay không.

## Files

| File | Trách nhiệm |
|---|---|
| `code/calendar_features.py` | `compute_calendar_vector(date_str)` — hàm thuần, 10 feature, bảng Tết 2005-2027 hardcode |
| `code/dataset_calendar_news.py` | `MultiStockDatasetWithCalendarNews` — x_news = dual-group (sibling, read-only) ++ calendar (tính live, không cần panel/parquet riêng) |
| `code/train_calendar_news_gate.py` | copy-modify từ `per_ticker_news_gate_baseline`, chỉ đổi nguồn dataloader |
| `requirements/requirements.md`, `design/design.md` | spec + kiến trúc + giả định (A1-A4) + Simplicity/Anti-Abstraction gate |
| `code_review/code_review_2026-08-01.md` | tự review đối kháng — 0 HIGH, 1 MEDIUM (calendar-day proxy, có chủ đích) + 2 LOW |

## Tests + coverage

**35/35 pytest pass** (`baselines/2026-08-01_calendar_news_gate_baseline/test/`):
- 23 unit test cho `compute_calendar_vector` (mọi nhánh: Tết đúng ngày/xa Tết/biên cửa sổ, cuối
  tháng/quý thường+nhuận, earnings window + biên năm).
- 7 dataset smoke test (synthetic + **1 real-data-sample smoke**: slice thật `ACB_processed.csv` +
  `dual_group_news_panel.parquet` thật).
- 5 train smoke test (1 `train_epoch` thật end-to-end, gate debug/logging).

Diff-coverage: **Not run** (tool chưa cài, gap đã biết — xem CLAUDE.md "Tooling gaps").

## Code review

Tự thực hiện đối kháng (agent/PR tương tác bất khả thi, cùng lý do các baseline trước). **0 HIGH.**
1 MEDIUM (month/quarter-end dùng ngày lịch, không phải ngày giao dịch thật — quyết định giữ,
Simplicity Gate, đã nêu rõ requirements A3) + 2 LOW (bảng Tết chỉ phủ 2005-2027; ranh giới
earnings-window không đối xứng ở ngày quarter-end). Đã kiểm tra riêng rủi ro rò rỉ dữ liệu (bài học
từ bug `TRAIN_CUTOFF` trước đây) — **không áp dụng** ở đây vì `compute_calendar_vector` không có
bước fit nào, thuần là hàm của ngày.

## Kết quả thật (test set, 10 epoch, best-val checkpoint = epoch 6)

So sánh với `per_ticker_news_gate_baseline` (CÙNG 10 epoch, CÙNG panel dual-group, CÙNG model/loss
— chỉ khác 10 cột calendar), lấy từ `results/per_ticker_gate_2026-07-26_221920/results.json`:

| Metric | Không calendar (146 cột) | **Có calendar (156 cột)** | Diff |
|---|---:|---:|---:|
| Test DirAcc | 68.76% | 68.13% | **-0.63pp (xấu hơn)** |
| Test R² | 0.71587 | 0.71173 | -0.0041 (xấu hơn) |
| Test QLIKE | 0.54969 | 0.56602 | **+0.0163 (xấu hơn)** |
| Test RMSE | 0.002635 | 0.002654 | +0.000019 (xấu hơn) |
| Test MAE | 0.0007181 | 0.0007256 | +0.0000075 (xấu hơn) |
| Test MSE | 6.943e-6 | 7.045e-6 | +0.102e-6 (xấu hơn) |

**Cả 6/6 metric đều xấu đi nhẹ khi thêm calendar feature** — kết quả NO-LIFT (thậm chí hơi tiêu
cực), nhất quán với pattern đã thấy ở 12 lần thử tích hợp tin tức trước đó của project (chưa lần
nào vượt HAR-only 69.98% DirAcc).

**Diễn giải (không kết luận quá đà — per design.md §7 risk):** kết quả này KHÔNG chứng minh "tin
tức không bị ảnh hưởng bởi Tết/mùa BCTC" — chỉ cho thấy cách đưa vào ĐƠN GIẢN NHẤT (nối thẳng 10
cột vào x_news, gate vẫn tĩnh theo mã, không tương tác) không đủ để kiến trúc hiện tại (concat +
Linear + LSTM 156→64) tận dụng được tín hiệu này trong 10 epoch. Có thể tín hiệu bị "chìm" trong
156 chiều, hoặc thật sự không có tương tác đáng kể, hoặc cần Cách 2 (gate học theo thời gian) để
model chủ động khuếch đại calendar signal thay vì tự học từ concat.

**Gate học được** (per-ticker, `results.json.final_gate_values`) dao động 0.10 (VHM) → 0.87 (HDB) —
đa dạng nhưng CHƯA đối chiếu với ablation độc lập (ngoài scope báo cáo này).

## Impact analysis

Chỉ tạo file MỚI trong baseline folder riêng (hard isolation, CLAUDE.md §3.F) — không sửa
`per_ticker_news_gate_baseline`, `dual_group_news_embedding_baseline`, hay bất kỳ file `src/` nào.
Đã grep xác nhận: không có baseline nào khác import từ folder mới này (không có consumer khác cần
kiểm tra).

## DoD checklist

- [x] Requirements + Design (SDD đầy đủ: Specify/Clarify(AskUserQuestion)/Plan/Tasks/Implement
      test-first/Validate)
- [x] Test-first: viết `test_calendar_features.py` TRƯỚC, xác nhận FAIL (ModuleNotFoundError), rồi
      mới implement `calendar_features.py`
- [x] Code: 3 file mới, Anti-Abstraction Gate PASS (tái dùng model/loader không sửa)
- [x] Tests: 35/35 pass (bao gồm real-data-sample smoke)
- [x] Code review adversarial: 0 HIGH, findings đã ghi nhận là giới hạn có chủ đích
- [x] Train 10 epoch thật (cap Training policy), so sánh công bằng với sibling cùng epoch
- [x] Impact analysis
- [x] Summary report (file này)
- [ ] Diff-coverage: Not run (tooling gap đã biết)

## Đề xuất tiếp theo (chờ user quyết định)

1. **Không leo thang lên Cách 2 (gate theo thời gian) ngay** — kết quả Cách 1 no-lift, cần cân
   nhắc kỹ trước khi đầu tư kiến trúc phức tạp hơn cho cùng giả thuyết.
2. Nếu vẫn muốn kiểm tra giả thuyết Tết/BCTC, có thể thử tách riêng TỪNG nhóm feature (chỉ Tết,
   chỉ earnings, chỉ day-of-week) thay vì cả 10 cột cùng lúc — cô lập rõ hơn cột nào (nếu có) mang
   tín hiệu, cột nào chỉ thêm nhiễu.
3. Bảng ngày Tết (A1) nên được spot-check ít nhất vài mốc trước khi tin tưởng bất kỳ kết luận nào
   liên quan tới Tết.
