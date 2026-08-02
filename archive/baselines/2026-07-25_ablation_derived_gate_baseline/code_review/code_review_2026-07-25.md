# Code Review — Ablation-Derived Gate Baseline (2026-07-25)

**Method:** Self-directed adversarial review (same rationale as the three sibling baselines
today — `/code-review` expects a GitHub PR, not applicable to uncommitted local work).

## Findings

No new correctness issues — this is a mechanical variant of
`2026-07-25_selective_news_gate_baseline` (mask mechanism already reviewed/tested there) and
`2026-07-25_top3_news_gate_baseline` (identical pattern, only the ticker allowlist source
changed: internal LSTM-GNN ablation instead of HGB/XGBoost EDA).

- **4/4 tests pass**, including exact-zero-contribution checks.
- **Ticker list correctly sourced** from `2026-07-25_news_usefulness_ablation`'s
  epoch-matched (10-vs-10), QLIKE-based delta computation — verified the 11-ticker set in
  `model_ablation_gate.py` matches `results/ablation_derived_ticker_classification.json`'s
  `news_on_tickers` exactly.

## Result finding (not a code bug — an empirical result)

Of the three ticker-gating attempts today, this one shows the result most consistent with the
hypothesis: NEWS_ON tickers averaged 50.47% test DirAcc vs. NEWS_OFF's 47.33% (+3.1pp, correct
direction) — compared to the 22-ticker EDA gate (OFF beat ON by 5.3pp, contradicted) and the
3-ticker EDA gate (a tie). QLIKE (0.5623) is close to both the all-ON model (0.5652 @10ep) and
the HAR-only reference (0.5623 — coincidentally near-identical), so this variant is not a clear
win on the primary QLIKE criterion either, despite the more promising DirAcc split. See summary
report for full discussion, including the still-substantial per-ticker noise (e.g. HPG, one of
the 11 ON tickers, scored only 35.58% test DirAcc individually, vs. VHM's 77.91% — an over-40pp
spread among the ON group itself).

## Summary

0 code findings. Result is the most encouraging of the three gating attempts today but still
modest and noisy — reported honestly, not oversold.
