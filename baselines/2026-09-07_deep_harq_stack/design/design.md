# Design — deep + HARQ stack

## Data flow
1. `recompute_harx(market, h)` — rebuild the canonical panel + folds (delivered `build_enriched_panel` /
   `make_folds`), and per fold fit HAR-X (`har5`) and HAR-X-Q (`har5 + harq`) OLS on TRAIN, predict on BOTH
   val and test, keyed by `(ticker, date)` (`_pred_dict` keys by node index → mapped via `panel.tickers`).
   Reuses `harq_walkforward.quarticity_panel` and `_ols_predict` — identical HAR-X-Q as the committed baseline.
2. `load_volga(market, h, split)` — VolGA forecasts from the canonical `qlike_anchor` cells (5-seed ensemble).
3. `_fit_w` — weight on VolGA minimising pooled VALIDATION QLIKE (grid 0..1). Val only — no test peeking.
4. `run` — evaluate `stack = w·VolGA + (1−w)·HAR-X-Q` on TEST for w∈{0.5 (primary), w_fit (control)}; metrics
   + date-clustered DM vs HAR-X. Also reports HAR-X / HAR-X-Q / VolGA test QLIKE for context.

## Key decisions
- **Equal weight is the headline, not the fitted weight.** 1/N combination is robust out-of-sample; the
  val-fit weight overfits (documented control). This avoids any suspicion of test-tuned weights: w=0.5 is
  chosen a priori.
- **Reuse the reviewed HAR-X-Q.** The stack does not re-implement HARQ; it imports `harq_walkforward`, so the
  HAR-X-Q leg is byte-identical to the committed, reviewed baseline; HAR-X reproduces canonical exactly.
- **Alignment on `(ticker,date)`.** Only cells present in all of HAR-X / HAR-X-Q / VolGA are scored, so all
  three models are compared on an identical set; the QLIKE floor is shared.

## Gates (SDD)
- Simplicity: a fixed-weight average of two frozen forecasts; the only estimated quantity (the control
  weight) is explicitly shown to overfit and is not the reported result.
- Anti-abstraction: reuses the delivered panel/folds/OLS/metrics/DM and the committed HARQ baseline directly.
- Performance: vectorised numpy; per market×horizon a linear OLS refit loop (seconds); no GPU.

## Caveats
- The `run()` test path calls `_require_overlap` to fail loud if the VolGA cell dump is stale/misaligned and
  the 3-way `(ticker,date)` intersection drops most cells (rather than silently comparing on a truncated set).
- The val-fit control uses `.update()` over folds; walk-forward val windows can repeat a `(ticker,date)` so
  the pooled val panel is last-fold-wins. This affects ONLY the `val_fit` overfitting control (reported to
  show 1/N is more robust); test cells are disjoint across folds, so the primary equal-weight result is exact.

## Files
- `code/deep_harq_stack.py` — recompute_harx / load_volga / _align / _fit_w / run / main.
- `test/test_stack.py` — alignment, node-index→ticker mapping, val-fit direction, blend endpoints, real-data
  smoke (equal stack beats HAR-X AND both members at h5, significantly).
