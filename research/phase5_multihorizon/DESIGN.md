# Phase 5 Design: Bug Fix (per-ticker windowing/scaling) + Multi-Horizon Forecast (+1, +10 days)

**Date:** 2026-08-01
**Branch:** `global-benchmark`

---

## 0. Bug fix design — `src/common/multi_ticker_dataset.py`

**Why a shared module (revises the "no shared abstraction" call in the original design):** the 3 bugs
in §0 of REQUIREMENTS.md exist identically in both `train_enhanced.py` and `cross_market_experiment.py`
— both build a multi-ticker `VolatilityDataset` the same broken way. Fixing it in 2 places would
duplicate non-trivial, correctness-critical logic (exactly what the VN30 lesson in CLAUDE.md §"LSTM-GNN
Normalization Failure" warns about — normalization bugs are easy to reintroduce). One shared, tested
module is justified here (unlike the target-column-naming decision in §3 below, which stays a 1-liner
per file).

**Reuses, does not reinvent:** `temporal_split_dataframe()` from `src/common/temporal_split.py` already
does correct chronological single-series splitting (sort by date, slice by ratio) — the new module
calls this ONCE PER TICKER instead of once on a concatenated multi-ticker frame (that "once per ticker"
call is the actual fix for bug #1; the existing utility itself was never buggy, it was simply never
invoked per-ticker).

```python
# src/common/multi_ticker_dataset.py

class WindowedSeriesDataset(Dataset):
    """Sliding-window dataset over ONE already-scaled series. Never spans >1 ticker
    because the caller only ever passes one ticker's array."""
    def __init__(self, features_scaled: np.ndarray, target_scaled: np.ndarray, seq_length: int):
        self.features, self.target, self.seq_length = features_scaled, target_scaled, seq_length

    def __len__(self):
        return max(0, len(self.features) - self.seq_length)

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.target[idx + self.seq_length - 1]
        return torch.tensor(x), torch.tensor(y)


def build_per_ticker_datasets(ticker_dfs: dict[str, pd.DataFrame], feature_cols: list[str],
                               target_col: str, seq_length: int = 22,
                               train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                               date_column: str = "date"):
    """
    For each ticker: temporal_split_dataframe (per-ticker, not pooled) -> fit
    feature_scaler + target_scaler on that ticker's TRAIN split ONLY -> .transform()
    (not refit) on val/test -> build WindowedSeriesDataset per split.

    Returns: dict[ticker] -> {"train": ds, "val": ds, "test": ds, "target_scaler": scaler}
    Caller pools "train"/"val" across tickers via ConcatDataset for training;
    evaluates "test" per-ticker (own scaler for inverse_transform), then concatenates
    the ORIGINAL-SCALE (true, pred) arrays across tickers for pooled metrics.
    """
```

**Bug #1 fix (window boundary):** windows are built from ONE ticker's array at a time — structurally
impossible to cross into another ticker's rows, since `WindowedSeriesDataset` never sees more than one
ticker's data.

**Bug #2 fix (pooled scaler):** `feature_scaler`/`target_scaler` are instantiated and fit fresh **inside
the per-ticker loop**, on that ticker's train split only.

**Bug #3 fix (val/test refit):** val/test slices call `scaler.transform(...)`, never
`scaler.fit_transform(...)` — same scaler object fit once on train, reused read-only for val/test.

**Test evaluation detail:** because each ticker has its own `target_scaler`, pooling final test metrics
across tickers requires inverse-transforming each ticker's predictions with ITS OWN scaler before
concatenating true/pred arrays — done in the caller (`train_enhanced.py`/`cross_market_experiment.py`),
not inside the shared module, since "how to aggregate across tickers for reporting" is caller policy,
not a dataset-building concern.

**Follow-up fix (2026-08-01, same-day follow-up session):** while debugging the horizon-1 DirAcc
anomaly (RESULTS.md §4.1 — confirmed NOT a bug, a genuine data characteristic), found and fixed a
4th, separate, real bug: `evaluate_pooled`/`evaluate_train_market_split`/`evaluate_full_series`
concatenated multiple tickers' predictions before computing `directional_accuracy`, so `np.diff()`
computed a spurious "change" at each ticker-boundary seam. Fixed via
`src/common/evaluation.py::directional_accuracy_grouped`/`evaluate_predictions_grouped` (per-ticker
diffs, never crosses a boundary), test-first. Verified impact on real data is negligible
(`n_tickers - 1` spurious diffs out of 1600+ samples) — real, worth fixing, but not the explanation
for the horizon-1 anomaly.

**Known limitation (code review finding, not fixed this phase):** `build_full_series_datasets` fits
the test-market scaler on that market's ENTIRE date range, so an early window's normalization can use
statistics computed from chronologically later dates within the same series — a mild look-ahead in the
scaling step (not in the model's predictions, which never see future data). This is inherited from the
ORIGINAL `cross_market_experiment.py` (its `test_ds` also fit a scaler on the full `test_df`), not a
regression introduced by this phase's fix. A proper walk-forward test-market scaler would need its own
design effort — out of scope here (see REQUIREMENTS.md §5 "no model/architecture change").

---

## 1. Why only 1 parameter change (horizon)

Both `train_enhanced.py`'s `VolatilityDataset` and `cross_market_experiment.py`'s `VolatilityDataset`
already take a `target_col` string — the horizon is fully determined by which column is read, not by
any model/loss/split code. This mirrors the VN30 project's finding (report §7.1): "chỉ cần đổi 1 tham
số" — `forecast_horizon` only affects the line that computes the target column, everything downstream
(model, loss, temporal split, evaluation) is horizon-agnostic already.

## 2. Two independent call sites — no shared abstraction

`feature_merger.py` (Experiment A) and `cross_market_experiment.py` (Experiment B) each compute their
own target column today (`target_5d = parkinson_volatility.shift(-5)`), duplicated, because Experiment
B intentionally does NOT depend on `feature_merger.py` (it loads raw processed CSVs directly, HAR-only,
no market/sentiment merge — see `cross_market_experiment.py:76-110`).

**Decision: keep them independent.** Each gets its own `horizon` param and its own one-line
`shift(-horizon)`. Introducing a shared `create_target_column(df, horizon)` helper for 2 near-identical
1-liners would be premature abstraction for this change's scope (Simplicity Gate below).

## 3. Column naming

`target_5d` → generalized to `target_{horizon}d` (e.g. `target_1d`, `target_10d`). Chosen over a fixed
generic name (`target`) because:
- `feature_merger.py` already writes one CSV per ticker to a single shared `processed_sp500_enhanced/`
  directory — a horizon-named column lets a re-run at a different horizon add a column to the same file
  without clobbering other horizons' targets (useful if a future run wants h=1 and h=10 pre-computed in
  one CSV, though this phase only asks for one horizon per script invocation).
- Matches the exact naming already used in `project-context.md`'s documented (but never-implemented)
  `create_forecast_targets()` pattern (`target_1d`, `target_5d`, `target_10d`, `target_22d`) — reusing
  a naming convention already blessed in this project's own docs rather than inventing a new one.

## 4. Change per file

### `src/common/feature_merger.py`
- `merge_features(..., horizon: int = 5)` — new param, default preserves current behavior.
- `df[f"target_{horizon}d"] = df["parkinson_volatility"].shift(-horizon)` (replaces hardcoded `target_5d`).
- `df.dropna(subset=[f"target_{horizon}d"])` (was `["target_5d"]`).
- CLI: `--forecast_horizon` (default 5) added to `__main__` argparse block, passed through `merge_all_tickers`.

### `src/experiments/sp500/train_enhanced.py`
- CLI: `--forecast_horizon` (choices `[1, 5, 10]`, default 5).
- `target_col = f"target_{args.forecast_horizon}d"`.
- Replace the current "concat all tickers -> one positional 70/15/15 split -> one `VolatilityDataset`"
  block with: build a `{ticker: df}` dict (one df per ticker, loaded independently, NOT concatenated),
  call `build_per_ticker_datasets(ticker_dfs, feature_cols, target_col, seq_length)`, `ConcatDataset`
  the per-ticker `"train"`/`"val"` splits for `train_loader`/`val_loader`, and evaluate `"test"` per
  ticker (own `target_scaler`) before pooling (true, pred) arrays for the final `evaluate_predictions`
  call — same pooled-metric behavior as before, now scaled/windowed correctly.
- Results dir: `results/sp500_enhanced_h{horizon}_{timestamp}/` (was `sp500_enhanced_{timestamp}/`) so
  horizon-1/5/10 runs don't overwrite each other; `results.json` gains a `"forecast_horizon"` field.
- Requires re-running `feature_merger.py` with `--forecast_horizon 1`, `5`, and `10` first to produce
  `target_1d`/`target_5d`/`target_10d` columns in the enhanced CSVs (all 3 regenerated — see §0).

### `src/experiments/sp500/cross_market_experiment.py`
- CLI: `--forecast_horizon` (choices `[1, 5, 10]`, default 5).
- `load_market_data(market, tickers=None, horizon=5)` — returns `{ticker: df}` dict instead of one
  concatenated DataFrame (the concatenation was itself part of bug #1 — removed, not just relocated).
  `df[f"target_{horizon}d"] = df["parkinson_volatility"].shift(-horizon)` per ticker df.
- `run_experiment` calls `build_per_ticker_datasets` for the train-market dict (train/val split) and
  again for the test-market dict (test split only, its own per-ticker scalers — cross-market means the
  test market's tickers never appear in train, so there is no leakage risk in fitting test-market
  scalers on test-market data; this mirrors what the current code already does, just per-ticker instead
  of pooled).
- Results dir: `results/cross_market_h{horizon}_{timestamp}/`; `results.json` gains `"forecast_horizon"`.

## 5. Window-count risk check (per VN30 report §7.5 precedent)

Horizon 10 needs `seq_length(22) + horizon(10) = 32` minimum rows per split; horizon 1 needs 23 — both
less than horizon 5's already-proven 27, so no new minimum-row risk versus what Phase 3/4 already
handled. VN30's own audit at horizon 22 (44 rows min) found zero at-risk tickers even with much longer
minimums; 32 rows is not a new risk. No additional guard code needed (Simplicity — don't add a check
for a risk that doesn't exist at this scope).

## 6. Simplicity Gate

- No new classes, no new model, no new abstraction layer. 3 files touched, each gets 1 new CLI arg and
  1 generalized target-column line.
- **Pass.**

## 7. Anti-Abstraction Gate

- Reuses `VolatilityDataset`, `EnhancedLSTM`/`SimpleLSTM`, `evaluate_predictions`, temporal-split logic,
  `feature_merger.merge_features` — all unchanged. No new wrapper around PyTorch/pandas/sklearn.
- **Pass.**

## 8. Data flow

```
Experiment A (feature-set comparison):
  processed_sp500/{ticker}.csv
        │ feature_merger.merge_features(horizon=1|5|10)
        ▼
  processed_sp500_enhanced/{ticker}_enhanced.csv   (gains target_1d/target_5d/target_10d columns)
        │ train_enhanced.py --forecast_horizon {1,5,10} --feature_set {har,full}
        │   -> {ticker: df} dict -> build_per_ticker_datasets() -> ConcatDataset(train/val across tickers)
        │   -> per-ticker test eval -> pooled (true,pred) -> evaluate_predictions()
        ▼
  results/sp500_enhanced_h{1,5,10}_*/results.json

Experiment B (cross-market):
  processed_sp500/*.csv, processed/*.csv (VN30)
        │ cross_market_experiment.py --forecast_horizon {1,5,10}
        │   load_market_data() -> {ticker: df} dict (target_{h}d computed inline, no CSV write)
        │   -> build_per_ticker_datasets() for train-market and test-market separately
        ▼
  results/cross_market_h{1,5,10}_*/results.json
```

## 9. Test plan (test-first)

`tests/test_sp500/test_multi_ticker_dataset.py` (bug-fix correctness, written+run to FAIL first):
1. **Boundary test:** 2 synthetic tickers, distinct known value ranges (ticker A: 0.01-0.02, ticker B:
   100-200) — build datasets with `seq_length=5`, assert every window's raw (pre-scale) values fall
   within exactly one ticker's known range. Run against a version calling the OLD concat-then-window
   logic first to confirm it fails (cross-boundary window has mixed-range values), then against the new
   `build_per_ticker_datasets` to confirm it passes.
2. **Scaler-reuse test:** ticker with known train-split mean/std; assert val/test split's transformed
   values equal `(raw - train_mean) / train_std` (computed by hand), NOT `(raw - val_mean) / val_std`.
3. **Split-per-ticker test:** 2 tickers with different date ranges; assert each ticker's train/val/test
   boundary is computed from ITS OWN row count (via `temporal_split_dataframe` per ticker), not from a
   position in a concatenated multi-ticker frame.

`tests/test_sp500/test_multihorizon_target.py` (horizon correctness):
1. Synthetic DataFrame with known `parkinson_volatility` sequence (e.g. `[0.1, 0.2, ..., 1.0]`) →
   `merge_features(..., horizon=1)` produces `target_1d[i] == parkinson_volatility[i+1]`; same for
   `horizon=10`. Written and run to confirm FAIL against current code (which only ever writes
   `target_5d`) before the implementation change.
2. Same check reimplemented against `cross_market_experiment.load_market_data`'s per-ticker shift logic.
3. After implementation: all pass; existing `tests/test_sp500/test_feature_merger.py` (horizon=5
   default) still passes unchanged (with `target_5d` still present) — regression check.
