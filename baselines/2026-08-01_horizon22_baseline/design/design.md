# Design (Plan) — 22-Day-Ahead Volatility Forecast Baseline

## 1. Data flow

Identical to `2026-08-01_horizon10_baseline/design.md` §1, with `forecast_horizon=22`:
```
create_dual_news_dataloaders(..., forecast_horizon=22, ...)   # sibling function, UNCHANGED
        ▼
MultiStockDatasetWithDualNews._create_sequences()              # sibling class, UNCHANGED
        target_idx = i + 22 + 22 - 1 = i + 43
        ▼
ParallelLSTMGNN (HAR-only)        PerTickerGatedNewsBaseline (gated news)
train_har_only_reference_h22.py   train_per_ticker_gate_h22.py
(copy-modify h10 sibling)         (copy-modify h10 sibling, incl. resume support added today)
```

## 2. File list

```
baselines/2026-08-01_horizon22_baseline/
├── requirements/requirements.md
├── design/design.md
├── code/
│   ├── __init__.py
│   ├── train_har_only_reference_h22.py     # copy-modify horizon10 sibling, forecast_horizon default 22
│   └── train_per_ticker_gate_h22.py        # copy-modify horizon10 sibling (already has resume support)
├── code_review/code_review_2026-08-01.md
└── test/
    ├── __init__.py
    ├── test_target_shift_h22.py             # target_idx = i+43, window-count vs h5/h10
    └── test_train_smoke_h22.py              # train_epoch smoke + REAL full-universe window-count check (§3 risk)
```

No new dataset/model file — same reuse pattern as horizon10.

## 3. Real-data window-count check (design response to requirements.md §3)

Unlike horizon10 (only spot-checked 1 real ticker), this baseline's real-data test loads the
ACTUAL `common_stocks` list via `_load_raw_stock_data` + `_split_raw_data_by_date` (same functions
`create_dual_news_dataloaders` uses internally) and asserts every stock's train/val/test HAR
dataframe has `len(df) > seq_length + forecast_horizon` (44) BEFORE attempting to build the full
dataset -- catches a short-history ticker early with a clear assertion message, rather than a
confusing downstream `min()` over an empty/negative range.

## 4. Simplicity / Anti-Abstraction Gate

Same as horizon10 — PASS, no new abstraction, parameter already exists.

## 5. Risks

- **Window-count risk (§3) is the main new risk** — mitigated by the real-data test in §3 above,
  run BEFORE the real training command.
- Same overfitting-risk lesson from today's 5-day experiment applies: 10 epochs may show a
  still-improving trend, in which case the report will flag it (not silently extend without
  asking, per requirements.md §4).
