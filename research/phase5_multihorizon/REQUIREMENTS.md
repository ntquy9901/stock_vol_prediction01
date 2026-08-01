# Phase 5 Spec: Bug Fix (per-ticker windowing/scaling) + Multi-Horizon Forecast (+1, +10 days)

**Date:** 2026-08-01
**Branch:** `global-benchmark`
**Status:** Spec confirmed with user (fix 2 structural bugs FIRST, then multi-horizon; both experiments
A+B; 10 epoch/run; horizon-5 must be RETRAINED under the fixed pipeline — see §0)

---

## 0. Scope change — why horizon 5 gets retrained, not reused

While diagnosing why S&P 500 DirAcc (50.89%) is far below VN30 DirAcc (68.42%) on the same HAR-only
5-day setup, code review of `train_enhanced.py` and `cross_market_experiment.py` found 3 structural
bugs (not present in VN30's `ParallelLSTMGNN`, which keeps stocks on a separate array dimension):

1. **Sliding windows cross ticker boundaries.** Multiple tickers are `pd.concat`-ed row-wise into one
   DataFrame; `VolatilityDataset.__getitem__` then slices `idx:idx+seq_length` across that combined
   frame with no awareness of where one ticker's history ends and the next begins. Near a boundary, a
   22-day window mixes two unrelated stocks.
2. **`StandardScaler` fit on the pooled multi-ticker split, not per ticker.** Different stocks have
   different volatility scale; fitting one scaler over concatenated rows blends those scales.
3. **`train_ds`/`val_ds`/`test_ds` each fit their OWN independent scaler** (`VolatilityDataset.__init__`
   calls `fit_transform`, not `transform`, for every split) instead of fitting once on train and
   reusing `.transform()` for val/test — a train/inference distribution mismatch, found while fixing #2.

User decision: fix all 3 before adding `forecast_horizon` (see conversation). Consequence: the
existing horizon-5 numbers in `research/phase3_training/RESULTS.md` and
`research/phase4_crossmarket/RESULTS.md` were produced by the buggy pipeline and are **no longer a
valid baseline** to compare fixed-pipeline horizon-1/10 results against (that would confound "bug
fixed" with "horizon changed"). Horizon 5 is therefore **retrained** under the fixed pipeline alongside
horizon 1 and 10 — 3 horizons × 2 sub-experiments, all under identical (fixed) conditions.

## 1. Goal

1. Fix the 3 structural bugs in `train_enhanced.py` and `cross_market_experiment.py` (shared root
   cause → shared fix in a new `src/common/multi_ticker_dataset.py` helper, see DESIGN.md).
2. Add a parameterized `forecast_horizon` (replacing the hardcoded 5-day target) to both scripts.
3. Train horizon {1, 5, 10} × both sub-experiments (A: feature-set comparison, B: cross-market) under
   the fixed pipeline, and report whether the VN30-observed trend (shorter horizon = easier to
   forecast) holds for S&P 500 once the pipeline bugs are fixed.

Modeled on VN30 report §7 (`docs/report_2026-08-01/BAO_CAO_TONG_HOP.md`, other worktree): same
technique (parameterize the horizon, no model/loss/metric changes), same test-first verification of
target-shift correctness before training.

---

## 2. Input/Output

### Input
- Experiment A: `data/processed_sp500_enhanced/{AAPL,MSFT,GOOGL}_enhanced.csv` (existing, from Phase 3)
- Experiment B: `data/processed_sp500/*.csv` (S&P 500, 257 tickers) + `data/processed/*.csv` (VN30, 32 tickers)

### Output
- `results/sp500_enhanced_h{1,10}_{timestamp}/results.json` (Experiment A, 4 runs)
- `results/cross_market_h{1,10}_{timestamp}/results.json` (Experiment B, 4 runs)
- `research/phase5_multihorizon/RESULTS.md` — comparison tables (horizon × 6 mandatory metrics)

---

## 3. Acceptance Criteria

- [ ] New `src/common/multi_ticker_dataset.py` builds windows strictly within one ticker's own rows
      (never crossing into another ticker), fits `StandardScaler` once per ticker on that ticker's
      TRAIN split only, and reuses (`.transform()`, not refit) for that ticker's val/test.
- [ ] Test proves a window built near a ticker boundary contains ONLY that ticker's rows (synthetic
      2-ticker data, known values) — written and run BEFORE the fix, confirmed to FAIL against current
      `train_enhanced.py`/`cross_market_experiment.py` logic, then confirmed to PASS after the fix.
- [ ] Test proves val/test features are transformed with the TRAIN split's fitted scaler, not their own
      independently-fit scaler (compare transformed values against manual calculation).
- [ ] `--forecast_horizon` argument added to `feature_merger.py`, `train_enhanced.py`,
      `cross_market_experiment.py`; default unchanged (5) — existing calls remain backward compatible.
- [ ] Test proves `target_{h}d` column equals `volatility.shift(-h)` for h=1 and h=10 (synthetic data,
      known values) — written and run BEFORE the target-generation code is changed (test-first).
- [ ] 6 Experiment-A runs (10 epoch each: horizon {1,5,10} × feature_set {har,full}) complete and save
      `results.json` with all 6 mandatory metrics, under the FIXED pipeline.
- [ ] 6 Experiment-B runs (10 epoch each: horizon {1,5,10} × direction {sp500→vn30, vn30→sp500})
      complete and save `results.json` with all 6 mandatory metrics, under the FIXED pipeline.
- [ ] `RESULTS.md` reports all 3 horizons for both experiments under a clearly-labeled "post-bugfix
      pipeline" heading, and explicitly notes the old Phase 3/4 horizon-5 numbers are superseded (not
      comparable) due to the bug fix — not silently overwritten, so the regression is documented.
- [ ] `/code-review` run on all changed/new files; HIGH/MEDIUM findings fixed before done.

---

## 4. [NEEDS CLARIFICATION] — resolved

1. ~~Which experiments get multi-horizon?~~ → Both A (feature-set) and B (cross-market). (User choice)
2. ~~Which horizons?~~ → 1 and 10 (not 22). (User choice, narrower than VN30's full 1/5/10/22 sweep)
3. ~~Epoch budget?~~ → 10 epoch/run, matches CLAUDE.md experimentation cap and VN30 report's default.
4. ~~Fix bugs before or after multi-horizon?~~ → **Before** (User choice, this session). Consequence:
   horizon 5 also retrained (§0), so scope is now 3 horizons × 2 experiments = 12 runs, not 8.

---

## 5. Scope (In/Out)

### In Scope
- New `src/common/multi_ticker_dataset.py`: per-ticker temporal split (reuses existing
  `temporal_split_dataframe` from `src/common/temporal_split.py`) + per-ticker scaler fit-once-reuse +
  boundary-safe windowing.
- `--forecast_horizon` param in `feature_merger.py`, `train_enhanced.py`, `cross_market_experiment.py`.
- Rewire `train_enhanced.py` and `cross_market_experiment.py` to use the new shared dataset builder
  instead of their current (buggy) inline `VolatilityDataset`/split logic.
- Target-shift + windowing + scaler-reuse correctness tests (test-first).
- 12 training runs (10 epoch each): horizon {1,5,10} × {A: feature_set har/full, B: direction
  sp500→vn30/vn30→sp500}.
- Horizon comparison tables, clearly labeled as post-bugfix (not comparable to old Phase 3/4 numbers).

### Out of Scope
- Horizon 22 (explicitly excluded by user).
- Any model/architecture change (LSTM stays as-is in both scripts) — only data pipeline is fixed.
- Expanding ticker count beyond current scope (3 tickers for A, existing 257 SP500 / 32 VN30 for B).
- Retroactively "fixing" the numbers already published in Phase 3/4 RESULTS.md — those files are left
  as historical record; Phase 5 RESULTS.md documents that they are superseded.

---

## 6. Files to Create/Modify

### New Files
| File | Purpose |
|------|---------|
| `research/phase5_multihorizon/REQUIREMENTS.md` | This file |
| `research/phase5_multihorizon/DESIGN.md` | Design + gates |
| `research/phase5_multihorizon/RESULTS.md` | Comparison tables (post-bugfix) |
| `src/common/multi_ticker_dataset.py` | Shared per-ticker split/scale/window builder (bug fix) |
| `tests/test_sp500/test_multi_ticker_dataset.py` | Boundary/scaler-reuse correctness (test-first) |
| `tests/test_sp500/test_multihorizon_target.py` | Target-shift correctness (h=1, h=10) |

### Modified Files (Backward Compatible — default horizon=5 preserved)
| File | Change |
|------|--------|
| `src/common/feature_merger.py` | Add `horizon` param to `merge_features`; column name `target_{h}d` |
| `src/experiments/sp500/train_enhanced.py` | Use `multi_ticker_dataset` builder; add `--forecast_horizon` |
| `src/experiments/sp500/cross_market_experiment.py` | Use `multi_ticker_dataset` builder; add `--forecast_horizon` |

---

## 7. Success Metrics

- All 12 new runs produce valid `results.json` with 6 mandatory metrics (no NaN/crash).
- `RESULTS.md` table lets a reader compare DirAcc/R²/QLIKE/RMSE/MSE/MAE across horizon {1,5,10} for
  both HAR-only and Full feature sets (Experiment A), and both transfer directions (Experiment B).
- Bug-fix verified: DirAcc at horizon 5 (post-fix) is reported next to the old buggy 50.89%/49-51% —
  expectation (not guaranteed) is a measurable improvement, since near-random DirAcc was the symptom
  that triggered this investigation. If it does NOT improve, that itself is a reportable finding (bugs
  fixed but a different, deeper cause dominates — e.g. genuinely harder market, not just less mature
  pipeline).
