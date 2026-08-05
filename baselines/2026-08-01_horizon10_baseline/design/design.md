# Design (Plan) — 10-Day-Ahead Volatility Forecast Baseline

## 1. Data flow

```
create_dual_news_dataloaders(..., forecast_horizon=10, ...)   # sibling function, UNCHANGED,
        │                                                       chỉ truyền kwarg khác (5 -> 10)
        ▼
MultiStockDatasetWithDualNews._create_sequences()              # sibling class, UNCHANGED
        target_idx = i + seq_length(22) + forecast_horizon(10) - 1   # = i + 31 (thay vì i + 26)
        y[ticker] = parkinson_volatility[target_idx]
        ▼
ParallelLSTMGNN (HAR-only)                    PerTickerGatedNewsBaseline (gated news)
        │                                              │
train_har_only_reference_h10.py               train_per_ticker_gate_h10.py
(copy-modify train_har_only_reference.py)     (copy-modify train_per_ticker_gate.py)
```

## 2. File list

```
baselines/2026-08-01_horizon10_baseline/
├── requirements/requirements.md
├── design/design.md
├── code/
│   ├── __init__.py
│   ├── train_har_only_reference_h10.py     # copy-modify sibling, +1 CLI arg
│   └── train_per_ticker_gate_h10.py        # copy-modify sibling, +1 CLI arg
├── code_review/code_review_2026-08-01.md
└── test/
    ├── __init__.py
    ├── test_target_shift.py                # PROVES horizon=10 shifts target correctly
    └── test_train_smoke_h10.py             # train_epoch smoke, both scripts, forecast_horizon=10
```

**KHÔNG có file dataset/model mới** — cả 2 script import `create_dual_news_dataloaders`,
`MultiStockDatasetWithDualNews` (từ `2026-07-25_dual_group_news_embedding_baseline`, read-only) và
`ParallelLSTMGNN` / `PerTickerGatedNewsBaseline` (từ `src/` và
`2026-07-26_per_ticker_news_gate_baseline`, read-only) — không sửa gì, không viết lại.

## 3. Thay đổi cụ thể so với 2 script gốc (diff tối thiểu)

Cả 2 script chỉ thêm:
```python
ap.add_argument("--forecast_horizon", type=int, default=10,
                help="days ahead to predict (this baseline: 10, vs. project default 5)")
...
train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = create_dual_news_dataloaders(
    data_dir=..., news_panel_path=..., graph_method=..., batch_size=...,
    config=config, forecast_horizon=args.forecast_horizon)   # <-- dòng MỚI duy nhất
```
Và đổi output dir prefix (`results/har_only_h10_<ts>/`, `results/per_ticker_gate_h10_<ts>/`) để
không đụng folder 5-ngày.

## 4. Test — chứng minh horizon thật sự đổi (không chỉ "chạy không lỗi")

`test_target_shift.py`: dựng `MultiStockDatasetWithDualNews` với dữ liệu synthetic (biết trước
toàn bộ giá trị `parkinson_volatility`), gọi với `forecast_horizon=10`, lấy `y` của window 0, so
với `parkinson_volatility[0 + seq_length + 10 - 1]` tính THỦ CÔNG bên ngoài dataset — PHẢI khớp.
Đồng thời assert nó KHÔNG khớp với công thức `+5-1` (bắt lỗi nếu code lỡ dùng nhầm default cũ).

## 5. Simplicity Gate / Anti-Abstraction Gate

- **Simplicity Gate: PASS.** Không thêm abstraction — dùng đúng tham số đã tồn tại sẵn trong
  hàm/class có sẵn.
- **Anti-Abstraction Gate: PASS.** Không viết dataset/model mới; 2 script train là copy-modify tối
  thiểu (đúng quy ước CLAUDE.md §3.F rule 3 "hard-isolated copy-modification").

## 6. So sánh công bằng

4 ô so sánh (2 kiến trúc × 2 horizon), CÙNG panel, CÙNG seed init mặc định của mỗi script, CÙNG
10 epoch, CÙNG optimizer/hyperparameter — biến duy nhất đổi là `forecast_horizon`.

## 7. Risks

- **Windows giảm nhẹ** (mất thêm 5 ngày cuối/split so với horizon=5) — không đáng kể (hàng nghìn
  window còn lại).
- **VolatilityNormalizer fit trên target 10-ngày** có thể có phân phối hơi khác target 5-ngày
  (biến động xa hơn thường "mượt" hơn do trung bình hoá theo thời gian) — đây là hiện tượng dữ
  liệu tự nhiên, không phải bug, sẽ phản ánh trong kết quả (vd RMSE khác thang đo tuyệt đối, nên so
  sánh ưu tiên R²/DirAcc/QLIKE hơn RMSE thô giữa 2 horizon).
