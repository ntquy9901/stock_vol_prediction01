# Canonical results table — remaining live baselines (2026-08-03)

Scope: the still-live baselines beyond the 2 already covered in
`docs/reports/2026-08-03_final_paper_readiness_report.md` §1 (HAR-only backbone vs news-fusion
per-ticker-gate). Live list confirmed via `ls baselines/ | grep -v archive`:
`2026-07-25_dual_group_news_embedding_baseline`, `2026-07-25_expand_news_cache_baseline`,
`2026-07-25_macro_news_baseline`, `2026-07-26_spillover_qlike_baseline`,
`2026-08-01_calendar_news_gate_baseline`, `2026-08-01_horizon1_baseline`,
`2026-08-01_horizon10_baseline`, `2026-08-01_horizon22_baseline`
(`2026-07-26_per_ticker_news_gate_baseline` is the already-covered headline baseline;
`2026-07-15_objective_news_baseline` was closed pre-training per the readiness report §3.7).

## Headline determination

None of these baselines can be **cleanly** inference-only re-evaluated. Every saved checkpoint
predates the P1.2 cross-stock date-alignment fix, and every one of these baselines' dataloaders
depends on the exact helper functions that P1.2 rewrote. Re-running inference under the current
(fixed) pipeline would feed each model a structurally different test panel than it was trained on,
producing a misleading number rather than an honest comparison. Per the task's own guidance (P1.2
named as the canonical exclusion example), these are marked **excluded — needs fresh training run**,
not re-evaluated. No retraining was performed (out of scope).

### Evidence for the exclusion (applies to all model baselines below)

- P1.2 fix commit `6672ffa` — dated **2026-08-02 17:30** — rewrote `remove_outliers()` (now
  winsorizes in place instead of dropping rows), `_load_raw_stock_data()`, `_split_raw_data_by_date()`,
  and added `_reindex_to_common_dates()` in `src/lstm_gat_hybrid/dataset_with_graph_method.py`.
- Every news-fusion baseline builds its dataloaders via
  `create_*_dataloaders(...)` which imports and calls exactly
  `_load_raw_stock_data`, `_split_raw_data_by_date`, `_generate_har_for_split` from that file
  (verified in `dataset_dual_news.py:195-206`; the macro/spillover/calendar/horizon baselines all
  reuse `create_dual_news_dataloaders`).
- All checkpoints under `models/` for these baselines are dated **before** 2026-08-02 17:30
  (07-25/07-26 and 08-01 runs). No post-P1.2 checkpoint exists for any of them.
- Consequence: the winsorize-vs-drop change and `_reindex_to_common_dates()` alter which windows
  exist, their date alignment, and the per-ticker normalizer statistics — so the model weights
  (fit to pre-P1.2 windows/normalizers) are not compatible with the current test panels. The
  readiness report §2 quantifies the material impact of P1.2 indirectly: per-ticker-gate QLIKE
  moved from ~0.553 (pre-P1.2) to ~0.464 (post-P1.2) on the headline pipeline.
- Separately, the DirAcc formula bug was already surfaced at run time for the news-fusion baselines
  (their `results.json` store both the flatten-biased `directional_accuracy` and the corrected
  `directional_accuracy_per_stock`), so the corrected DirAcc is available — but it too was computed
  on pre-P1.2 (misaligned) data and is therefore not paper-comparable.

## Table

Metrics below are the **historical values recorded at each run's own (pre-P1.2) time** — shown for
traceability only, **not comparable** to the post-fix headline table and **not for paper citation**.
DirAcc column reports the corrected per-ticker value (`directional_accuracy_per_stock`) where the
run recorded it; the flatten-biased headline value is given in parentheses.

| Baseline | Architecture summary | QLIKE | RMSE | MAE | R² | DirAcc (corrected; flatten-biased) | Epochs / seed | Checkpoint | Confidence |
|---|---|---|---|---|---|---|---|---|---|
| dual_group_news_embedding | `DualGroupNewsBaseline` — ParallelLSTMGNN backbone + dual-group (market/ticker) news embedding fusion, n_feat=146, d_news=64 | 0.5598 | 0.002651 | 0.000717 | 0.712 | 47.47% (68.25%) | 10 / 42 | `models/dual_group_news_2026-07-26_192414/best.pt` (07-26) | Excluded — pre-P1.2 checkpoint, needs fresh run |
| macro_news | `MacroNewsBaseline` (DualGroupNews, wider n_feat=212) — adds macro news channel, d_news=64 | 0.5634 | 0.002662 | 0.000717 | 0.710 | 45.71% (68.63%) | 10 / 42 | `models/macro_news_2026-07-26_020954/best.pt` (07-26) | Excluded — pre-P1.2 checkpoint, needs fresh run |
| spillover_qlike | `DualGroupNewsBaseline` + directed spillover graph + QLIKE-weighted loss, n_feat=146, d_news=64 | 0.5622 | 0.002647 | 0.000722 | 0.713 | 47.22% (68.23%) | 10 / 42 | `models/spillover_qlike_2026-07-26_192749/best.pt` (07-26) | Excluded — pre-P1.2 checkpoint, needs fresh run |
| calendar_news_gate | `PerTickerGatedNewsBaseline` with calendar-augmented x_news (Tet/earnings/generic calendar features), n_feat=156, d_news=64 | 0.5660 | 0.002654 | 0.000726 | 0.712 | 45.65% (68.13%) | 10 / 42 | `models/calendar_gate_2026-08-01_073829/best.pt` (08-01) | Excluded — pre-P1.2 checkpoint, needs fresh run |
| horizon1 — HAR-only ref | `ParallelLSTMGNN_HAR_only_reference`, forecast_horizon=1 | 0.5099 | 0.002428 | 0.000675 | 0.758 | — (72.35%) | not recorded / 42 | `models/har_only_h1_2026-08-01_103548/best.pt` (08-01) | Excluded — pre-P1.2 checkpoint, needs fresh run |
| horizon1 — per-ticker gate | `PerTickerGatedNewsBaseline`, horizon=1, n_feat=146, d_news=64 | 0.4834 | 0.002420 | 0.000651 | 0.759 | 33.16% (72.39%) | 10 / 42 | `models/per_ticker_gate_h1_2026-08-01_104140/best.pt` (08-01) | Excluded — pre-P1.2 checkpoint, needs fresh run |
| horizon10 — HAR-only ref | `ParallelLSTMGNN_HAR_only_reference`, forecast_horizon=10 | 0.5732 | 0.002689 | 0.000726 | 0.704 | — (67.80%) | not recorded / 42 | `models/har_only_h10_2026-08-01_090759/best.pt` (08-01) | Excluded — pre-P1.2 checkpoint, needs fresh run |
| horizon10 — per-ticker gate | `PerTickerGatedNewsBaseline`, horizon=10, n_feat=146, d_news=64 | 0.5733 | 0.002690 | 0.000735 | 0.704 | 48.81% (67.39%) | 20 / 42 | `models/per_ticker_gate_h10_2026-08-01_095135/best.pt` (08-01) | Excluded — pre-P1.2 checkpoint, needs fresh run |
| horizon22 — HAR-only ref | `ParallelLSTMGNN_HAR_only_reference`, forecast_horizon=22 | 0.5938 | 0.002750 | 0.000739 | 0.705 | — (66.38%) | not recorded / 42 | `models/har_only_h22_2026-08-01_101237/best.pt` (08-01) | Excluded — pre-P1.2 checkpoint, needs fresh run |
| horizon22 — per-ticker gate | `PerTickerGatedNewsBaseline`, horizon=22, n_feat=146, d_news=64 | 0.5943 | 0.002759 | 0.000750 | 0.703 | 42.89% (67.17%) | 10 / 42 | `models/per_ticker_gate_h22_2026-08-01_102011/best.pt` (08-01) | Excluded — pre-P1.2 checkpoint, needs fresh run |
| expand_news_cache | Data-cache building only (`build_incremental_cache.py`) — no model, no training loop | — | — | — | — | — | — | none | N/A — not a model baseline (no checkpoint) |

Notes on the table:
- Seed = 42 for all runs (`torch.manual_seed(42)` / `np.random.seed(42)` in each train script).
- Epoch counts taken from each run's `loss_history.json` length or learning-curve filenames; the
  HAR-only horizon references did not persist a loss history, so their exact epoch count is "not
  recorded" (the shared training scripts use `--epochs` default 10 with early stopping
  `min_epochs=min(20, epochs)`).
- QLIKE/RMSE/MAE/R²/DirAcc read directly from each run's `results/<dir>/results.json`
  (`test_metrics`).
- HAR-only horizon runs store per-ticker DirAcc under a separate `per_ticker_test_metrics` key (not
  a single scalar in `test_metrics`), so the corrected DirAcc scalar column is left as "—".

## Baselines that could NOT be cleanly re-evaluated, and why

1. **dual_group_news_embedding, macro_news, spillover_qlike, calendar_news_gate,
   horizon1/10/22 (both HAR-only-ref and per-ticker-gate variants)** — checkpoint predates the P1.2
   date-alignment structural fix (`6672ffa`, 2026-08-02 17:30). Their dataloaders call
   `_load_raw_stock_data` / `_split_raw_data_by_date` / `_generate_har_for_split` from
   `dataset_with_graph_method.py`, all rewritten by P1.2 (winsorize-in-place instead of row-drop, plus
   `_reindex_to_common_dates`). Running inference now would evaluate a model trained on the old
   (misaligned, row-dropped) panels against the new (realigned, winsorized) panels with re-fit
   normalizers — a distribution mismatch that yields a misleading number, not an honest comparison.
   No post-P1.2 checkpoint exists for any of them. Excluded from the clean comparison; each would
   need a fresh training run under the fixed pipeline to be paper-comparable. (Retraining was not
   performed — out of scope.)

2. **expand_news_cache** — not a model/training baseline at all; its `code/` contains only
   `build_incremental_cache.py`, a data-cache builder. There is no model class, training loop, or
   checkpoint to re-evaluate. N/A rather than a gap.

## What a fresh run would cost (for the user's decision)

If the paper needs any of these baselines comparable to the post-fix headline table, each requires a
retrain under the current pipeline (same pattern as the two headline runs in the readiness report:
20 epochs, seed=42). The news-fusion and horizon baselines already have working, isolated train
scripts under their own `baselines/<name>/code/`, so a fresh run is a straightforward invocation —
it was deliberately not triggered here because retraining was out of scope for this task.
