# Summary of Update — dataset.py normalization + split-first leakage fix

Date: 2026-08-03
Scope: `src/lstm_gat_hybrid/dataset.py`, `train.py`, `train_parallel.py` (the HAR-only
comparison baseline pipeline behind `train.py` / `train_parallel.py`).

## What changed and why

Two bugs in `MultiStockDataset` / `create_multi_stock_dataloaders` were fixed by
reusing the already-tested split-first helpers from the news-fusion lineage
(`dataset_with_graph_method.py`) — imported, not duplicated.

### Bug 1 — Leakage (HAR + normalizer fit over full unsplit series)
`create_multi_stock_dataloaders` previously built three FULL `MultiStockDataset`
instances (each computing HAR rolling features and fitting `VolatilityNormalizer`
over the whole series) and then positionally `Subset`-sliced them. HAR rolling
means and normalizer statistics therefore leaked future/val/test data into
training. Rewritten to: load raw → `_split_raw_data_by_date` (chronological) →
`_generate_har_for_split` (HAR per split) → fit normalizers on the TRAIN split only,
then copy (not refit) the fitted objects into val/test. This also inherits the P1.2
cross-ticker date-alignment fix (`_reindex_to_common_dates`) for free. `Subset`
slicing removed (each dataset instance now holds only its own split).

### Bug 2 — Dead normalization (fitted scalers never applied)
`MultiStockDataset.__getitem__` fitted normalizers but never called `.transform()`,
so training/eval ran on raw ~1e-3 volatility values (the exact "fit a scaler then
forget to use it" incident documented in CLAUDE.md). `__getitem__` now applies
per-stock feature/target `.transform()` and clips the normalized target to
`[-10, 10]`, mirroring `dataset_presplit.py::MultiStockDatasetWithPreSplitData.__getitem__`.

### Trainers
`validate()` in both `train.py` and `train_parallel.py` now takes a `dataset=`
param and inverse-transforms predictions AND targets back to raw volatility scale
per stock (flatten order `i % n_stocks`) before computing the 6 business metrics;
`avg_loss` stays on the normalized scale for learning-curve/early-stopping
comparability. Call sites wired to pass `datasets[1]` (val) / `datasets[2]` (test).
The `loss = loss * 10.0` hack in `train.py`'s `train_epoch`/`validate` was REMOVED —
with real StandardScaler-style normalization the MSE loss is naturally O(1) (a
constant loss multiplier only rescales gradients ≡ effective LR; unnecessary once
normalization is applied). `train_parallel.py` had no such hack.

### Backward compatibility
`MultiStockDataset`'s legacy direct-`data_dir` path is preserved unchanged (new
`precomputed_raw_data`/`precomputed_har_data` params gate the new path). The
standalone `test_phase1_implementation.py` still runs.

## Files (path → purpose)
- `src/lstm_gat_hybrid/dataset.py` — new precomputed-data constructor path;
  `__getitem__` normalization; `create_multi_stock_dataloaders` rewritten split-first.
- `src/lstm_gat_hybrid/train.py` — `validate(dataset=)` + inverse-transform; removed `*10` hack; wired call sites.
- `src/lstm_gat_hybrid/train_parallel.py` — `validate(dataset=)` + inverse-transform; wired call sites.
- `tests/lstm_gat_hybrid/test_dataset_normalization_fix.py` — new tests (7).

## Tests + verification
- `pytest tests/lstm_gat_hybrid/test_dataset_normalization_fix.py` → 7 passed.
- `pytest tests/lstm_gat_hybrid/` (full dir, incl. date-alignment + diracc) → 22 passed.
- `python test_phase1_implementation.py` (legacy direct-load path) → PASS.
- End-to-end smoke (epochs capped to 2 at runtime; no committed config change):
  - `train.py`: train loss 1.07→0.92 (normalized, O(1), decreasing); test RMSE 0.003488,
    MAE 0.000912, MSE 0.000012, R² 0.6796, QLIKE 0.7209, DirAcc 49.52% (per-ticker).
  - `train_parallel.py`: val loss ~1.06→1.00 (normalized); test RMSE 0.003179,
    MAE 0.000880, MSE 0.000010, R² 0.7339, QLIKE 0.5138, DirAcc 46.34% (per-ticker).
  - Both: normalized-scale loss O(1) (confirms normalization active), business
    metrics back on raw ~1e-3 volatility scale (confirms denormalization), non-constant
    predictions, DirAcc via the already-correct `n_stocks=` per-ticker path.

Coverage gate (diff-cover C0=100%): Not run — `diff-cover`/`pytest-cov` not installed
in this environment (documented tooling gap in CLAUDE.md §Per-project setup). New
behavior is covered by the 7 targeted tests above (transform applied, clip, train-only
normalizer fit + shared objects, no date overlap, normalized-scale batch, real-data smoke).

## Code review
Inline adversarial self-review performed (the orchestrating session waived external
review/approval). Checks: flatten-order `i % n_stocks` matches `y.reshape` row-major
and `stock_names` ordering (identical sorted keys across all 3 datasets); denorm
fallback when `target_normalizers` empty (normalize=False); no leftover debug/temp
code; legacy path untouched. No blocking findings.

## Risks / follow-ups
- Other ad hoc debug scripts under `src/lstm_gat_hybrid/` (`train_simplified.py`,
  `sanity_constant_baseline.py`, `debug_corrupted_val_batches.py`, `check_vhm_normalizer.py`)
  also import `create_multi_stock_dataloaders`; they get the unchanged return shape but
  their own `validate()` (if any) does not denormalize. Out of scope (not the reported
  pipeline); noted for awareness.
- Reported HAR-only baseline numbers in prior results tables were produced under the
  buggy (un-normalized + leaky) pipeline and should be re-generated with a full run
  before being cited.
