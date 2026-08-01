# Design (Plan) — Calendar-Augmented News Gate Baseline

## 1. Data flow

```
date string (từ window_dates, đã có sẵn trong pipeline dataset — KHÔNG cần nguồn dữ liệu mới)
        │
        ▼
calendar_features.py :: compute_calendar_vector(date_str) -> np.ndarray[10]   (PURE FUNCTION,
        │                không I/O, không phụ thuộc ticker — tính on-the-fly, KHÔNG cần build
        │                panel/parquet riêng vì chi phí tính gần như 0, khác hẳn macro_news_baseline
        │                cần PhoBERT+PCA nên phải cache thành parquet)
        ▼
dataset_calendar_news.py :: MultiStockDatasetWithCalendarNews (mirror dataset_macro_news.py pattern)
        x_news[t, ticker] = concat(dual_group_vec[ticker, date_t],   # 146, từ sibling loader
                                    calendar_vec(date_t))             # 10, tính live, GIỐNG NHAU
                                                                       # cho mọi ticker cùng ngày
        │
        ▼
train_calendar_news_gate.py (copy-modify sibling train_per_ticker_gate.py)
        model = PerTickerGatedNewsBaseline(n_feat=156, num_stocks=30)   # TÁI DÙNG KHÔNG SỬA,
                                                                          # import từ sibling
```

## 2. Vì sao KHÔNG cần panel/parquet riêng cho calendar (khác macro_news_baseline)

`macro_news_baseline` cần `build_macro_panel.py` vì phải chạy PhoBERT + PCA (tốn CPU/GPU, cache
lại là bắt buộc). Calendar feature là hàm thuần (sin/cos/so sánh ngày) — chi phí tính lại mỗi lần
là không đáng kể (10 phép tính số học/ngày). Cache ra parquet ở đây là abstraction thừa (vi phạm
Anti-Abstraction Gate) — tính trực tiếp trong `_create_sequences` khi build `day_feats`, đúng vị
trí nơi `window_dates` đã có sẵn.

## 3. File list

```
baselines/2026-08-01_calendar_news_gate_baseline/
├── requirements/requirements.md
├── design/design.md
├── code/
│   ├── __init__.py
│   ├── calendar_features.py          # compute_calendar_vector(date_str), CALENDAR_FEATURE_NAMES,
│   │                                   # TET_DATES lookup (2005-2027, xem requirements.md A1)
│   ├── dataset_calendar_news.py       # MultiStockDatasetWithCalendarNews + create_calendar_news_dataloaders
│   │                                   # import read-only: load_news_panel từ dual_group sibling
│   └── train_calendar_news_gate.py    # copy-modify train_per_ticker_gate.py; import
│                                       # PerTickerGatedNewsBaseline KHÔNG SỬA từ per_ticker_gate sibling
├── code_review/code_review_2026-08-01.md
└── test/
    ├── __init__.py
    ├── test_calendar_features.py      # unit test hàm thuần — KHÔNG cần data thật, exhaustive
    │                                   # (mọi nhánh: Tet đúng ngày, xa Tet, cuối tháng/quý,
    │                                   #  trong/ngoài earnings window)
    ├── test_dataset_smoke.py          # real-data-sample smoke (CLAUDE.md Testing rules): 1 lát
    │                                   # cắt nhỏ data thật + panel thật -> assert shape [.,.,156]
    └── test_train_smoke.py            # --smoke mode, 1-2 epoch, verify gate_history.json write
```

## 4. Model — KHÔNG tạo file mới

`PerTickerGatedNewsBaseline` từ `2026-07-26_per_ticker_news_gate_baseline/code/model_per_ticker_gate.py`
dùng thẳng, import read-only. `n_feat` là tham số constructor — không cần sửa gì để nhận input
156 chiều thay vì 146. Đây là bằng chứng cụ thể Anti-Abstraction Gate đã tuân thủ, giống hệt cách
`macro_news_baseline` tái dùng `DualGroupNewsBaseline` không sửa.

## 5. Simplicity Gate / Anti-Abstraction Gate

- **Simplicity Gate: PASS.** 10 cột feature mới, tính bằng numpy thuần (sin/cos/so sánh), không
  thêm dependency, không thêm panel/parquet không cần thiết (xem §2).
- **Anti-Abstraction Gate: PASS.** Tái dùng 100% model (`PerTickerGatedNewsBaseline`), tái dùng
  loader dual-group (`load_news_panel`), tái dùng toàn bộ debug/logging/plotting logic của
  `train_per_ticker_gate.py` (chỉ đổi nguồn dataloader). KHÔNG viết panel-building script mới vì
  không cần (§2 giải thích rõ vì sao macro_news khác).

## 6. So sánh công bằng (đúng biến cần đo)

So sánh CHÍNH: baseline này (156-dim x_news) vs. `per_ticker_news_gate_baseline` (146-dim, không
calendar) — CÙNG epoch (10), CÙNG panel dual-group gốc, CÙNG model class, CÙNG loss/optimizer
config. Biến duy nhất khác nhau: có/không 10 cột calendar. Đây là phép so sánh cô lập đúng
(requirements.md §2).

## 7. Risks

- **10 cột thêm vào 146 cột hiện có — tín hiệu nhỏ có thể bị "chìm"** trong không gian 156 chiều
  trước khi qua `NewsFeatureLSTM` (Linear 156→64) — nếu no-lift, không kết luận được "calendar
  không có ích" tuyệt đối, chỉ là "không đủ mạnh để nổi lên qua kiến trúc concat-Linear hiện tại"
  (ghi rõ trong summary, tránh kết luận quá đà).
- **Bảng ngày Tết gõ tay (A1)** — rủi ro sai 1-2 ngày ở 1 vài năm; ảnh hưởng cục bộ (chỉ lệch cửa
  sổ ±10 ngày quanh năm đó), không ảnh hưởng toàn bộ kết quả nếu có sai sót nhỏ.
- **earnings_window là proxy toàn thị trường (A2)** — không phân biệt được mã nào công bố sớm/muộn
  trong cửa sổ 20 ngày; nếu muốn chính xác hơn cần dữ liệu ngày công bố thật/mã (ngoài scope).
