# Design (Plan) — 1-Day-Ahead Volatility Forecast Baseline

Identical mechanism to `2026-08-01_horizon10_baseline` / `2026-08-01_horizon22_baseline` — see
those `design.md` files for the full explanation (reused here, not re-derived).

## File list

```
baselines/2026-08-01_horizon1_baseline/
├── requirements/requirements.md
├── design/design.md
├── code/
│   ├── __init__.py
│   ├── train_har_only_reference_h1.py     # copy-modify h22 sibling, forecast_horizon default 1
│   └── train_per_ticker_gate_h1.py        # copy-modify h22 sibling (incl. resume support)
├── code_review/code_review_2026-08-01.md
└── test/
    ├── __init__.py
    ├── test_target_shift_h1.py             # target_idx = i+22
    └── test_train_smoke_h1.py              # smoke + real-data window-count check
```

No new dataset/model file — same reuse as horizon10/horizon22.

## Simplicity / Anti-Abstraction Gate

PASS — identical to the 2 prior horizon baselines, no new abstraction.
