# Summary: Per-ticker DirAcc fix extended to the 3 remaining HAR-only pipeline files

## Context

Commit `fccaf6a` fixed `evaluate_predictions()` in `src/common/evaluation.py` to accept an
optional `n_stocks` parameter that computes `directional_accuracy` correctly per-ticker on a
day-major, ticker-interleaved flattened multi-stock array (see
`docs/report_2026-08-01/DIRACC_ISSUE_NOTE.md`), instead of the old behavior that silently
compared different tickers' same-day values. That commit applied `n_stocks=` at ~22 caller
sites across the news-fusion baseline lineage (`baselines/`) but not to 3 files that are part
of the original, non-news "HAR-only" pipeline. This change closes that gap.

## Files changed

- `src/lstm_gat_hybrid/train.py` — `validate()`: `evaluate_predictions(all_targets, all_predictions)`
  → `evaluate_predictions(all_targets, all_predictions, n_stocks=num_stocks)`. `num_stocks` reused
  from the existing `batch_size, num_stocks = y.shape` loop variable (in scope after the loop).
- `src/lstm_gat_hybrid/train_parallel.py` — identical one-line fix in its `validate()`.
- `src/lstm_gat_hybrid/train_parallel_enhanced.py` — `validate()` has two branches:
  - Normalizer branch (the one used by every real run in this file, since datasets always carry
    `target_normalizers`): moved the pre-existing `n_stocks = len(actual_dataset.stock_names)`
    computation (previously computed only for a separate manual per-stock block) to before the
    `evaluate_predictions(...)` call, and passed `n_stocks=n_stocks`.
  - No-normalizer fallback branch: added `n_stocks=num_stocks`, reusing the `num_stocks` captured
    from the dataloader loop (`batch_size, num_stocks = y.shape`).

All three files already reported `metrics['directional_accuracy']` directly to
console/`results.json` as the headline value, so the fix changes what gets reported, not just an
internal value nothing reads.

## Ordering contract verified before applying the fix

Confirmed (read `src/lstm_gat_hybrid/dataset.py::MultiStockDataset._create_sequences` and
`src/lstm_gat_hybrid/dataset_presplit.py::MultiStockDatasetWithPreSplitData._create_sequences`)
that both dataset classes build `y` per window in a fixed `stock_names` order, append sequences
in increasing (chronological) window order, and every DataLoader in `dataset.py` and
`dataset_with_graph_method.py` uses `shuffle=False` for train/val/test. This guarantees the
flattened `[batch, num_stocks]` → `.reshape(batch_size * num_stocks)` arrays used in all 3 files
are day-major, ticker-interleaved (`index i` → ticker `i % n_stocks`, window `i // n_stocks`) —
exactly the contract `directional_accuracy_per_ticker()` requires. No ordering mismatch found.

## Tests

New file: `tests/lstm_gat_hybrid/test_diracc_per_ticker_fix.py` (pytest, `pytestmark =
pytest.mark.smoke`, follows the house style of `test_date_alignment_fix.py`).

- `TestEvaluatePredictionsKnownDivergence` — pure-helper check: a hand-constructed 2-ticker x
  5-window array where per-ticker DirAcc is a known 0% (predictions move opposite the true
  direction every window, every ticker) while the flattened DirAcc is a known 100% (dominated by
  the ~100x level gap between tickers). Confirms `evaluate_predictions(..., n_stocks=2)` returns
  0.0 as `directional_accuracy` and 100.0 as `directional_accuracy_flat_biased`.
- `TestTrainPyValidateIntegration` / `TestTrainParallelPyValidateIntegration` — I/O-level test:
  drives the real `validate()` function from each file with a dummy queue-based model and a plain
  list-of-batches "dataloader" (no real DataLoader needed since these files don't shuffle),
  reproducing the known-divergence scenario end-to-end and asserting the returned
  `metrics['directional_accuracy']` is the correct per-ticker value.
- `TestTrainParallelEnhancedPyValidateIntegration` — same, covering both branches: with a fake
  dataset carrying identity normalizers (the production code path) and with `dataset=None`
  (fallback path).

**Regression check:** before finalizing, the 3 source fixes were `git stash`-ed (test file kept)
and the integration tests were re-run — all 4 integration tests failed against the pre-fix code
(returned the inflated 100.0 instead of the correct 0.0), confirming the tests actually exercise
the bug. Fix was then restored (`git stash pop`) and all 15 tests in `tests/lstm_gat_hybrid/`
pass.

Commands run:
```
python -m py_compile src/lstm_gat_hybrid/train.py src/lstm_gat_hybrid/train_parallel.py src/lstm_gat_hybrid/train_parallel_enhanced.py
python -m pytest tests/lstm_gat_hybrid/ -v        # 15 passed
```

## End-to-end smoke run (COMPLETED)

`src/lstm_gat_hybrid/train_parallel_enhanced.py` has a built-in `--quick_test` flag (5 epochs,
within CLAUDE.md's 5-10 epoch experimentation cap) — ran as a real training run against
`data/processed` (`python src/lstm_gat_hybrid/train_parallel_enhanced.py --quick_test
--graph_method knn`, ~4.5 minutes, CPU, 33 stocks, knn graph). Completed successfully (exit 0),
5 epochs, best epoch 4, results saved to
`results/parallel_lstm_gnn_knn_2026-08-03_002158/training_results.json`.

`training_results.json`'s `test_metrics.directional_accuracy` = **48.14%** (the corrected,
per-ticker value — this is what's now reported as headline DirAcc in console + JSON, confirmed
identical to `directional_accuracy_per_stock`).

`directional_accuracy_flat_biased` (the pre-fix formula's value) is computed internally by
`evaluate_predictions()` once `n_stocks` is passed, but this training script only prints/saves
`directional_accuracy` to console/JSON (not the biased companion value) — so, to get an explicit
before/after number from this exact run rather than relying only on the synthetic-data proof, the
saved checkpoint (`best_parallel_model.pth`) and the same test dataloader were reloaded in a
throwaway script (not committed) that called `validate()` again and printed the full metrics dict:

```
mse: 8.682940460857935e-06
rmse: 0.002946682958999481
mae: 0.0008301305933855474
r2: 0.771415114402771
qlike: 0.46505415439605713
directional_accuracy: 48.14008346942479          <- corrected headline (per-ticker), AFTER fix
directional_accuracy_flat_biased: 71.56774309940465  <- old formula, BEFORE fix (what would have
                                                        been reported as headline previously)
r2_per_stock: -0.10455963760614395
directional_accuracy_per_stock: 48.14008346942479    <- matches directional_accuracy exactly, confirms fix wiring
```

**Real-run confirmation: 48.14% (corrected) vs 71.57% (old, inflated) — a 23.4 percentage-point
gap**, consistent with the project's previously documented 20-40pp inflation pattern
(`docs/report_2026-08-01/DIRACC_ISSUE_NOTE.md`). Note the corrected 48.14% is close to/below random
chance (50%), matching the pattern already seen in the news-fusion baselines once the same fix was
applied there.

`train.py` and `train_parallel.py` have **no quick/smoke CLI flag** — their `if __name__ ==
'__main__':` blocks call the full training function directly with `config.num_epochs` = 70 / 50
respectively, on CPU (`config.device == 'cpu'`, 30 stocks). Running these to even 5 epochs would
require code changes beyond this fix's scope (adding a quick-test mode) or an external monkeypatch
script, and CLAUDE.md's training policy caps unattended experimentation at 5-10 epochs with no
standing approval to add new CLI surface for this narrow fix. Not run end-to-end; verified instead
via the `validate()`-level integration tests above, which exercise the exact same code path
(`evaluate_predictions` call site) with real tensor shapes and the real function objects imported
from the modules — the only untested part is the surrounding data-loading/model-forward
machinery, which this change does not touch.

## Code review

Not run as a separate `/code-review` pass — change is a mechanical, one-line-per-call-site
parameter addition identical to the already-reviewed pattern from commit `fccaf6a`, reusing
existing in-scope variables, with the ordering contract independently re-verified against the
dataset source before applying (see above). Self-assessed low risk given the mechanical nature and
the regression-tested integration tests.

## Constraints honored

- Did not touch `dataset_with_graph_method.py`, `dataset.py`, `dataset_presplit.py` (concurrent
  work).
- Did not touch anything under `archive/`.
- Changes limited to the `n_stocks=` parameter addition (plus reordering one pre-existing line in
  `train_parallel_enhanced.py` so `n_stocks` is computed before its new use) and explanatory
  comments — no other refactoring.

## DoD checklist

- [x] Code satisfies the exact request; no unrelated refactor.
- [x] Tests written, cover the changed behavior (I/O-level, not just pure helper), pass; failure
      against pre-fix code confirmed.
- [x] `py_compile` run on all 3 changed files.
- [ ] `diff-cover` C0/C1 gate — **Not run**: `diff-cover`/`pytest-cov` not installed in this
      environment (documented pre-existing tooling gap in CLAUDE.md).
- [ ] Lint (`ruff`) — **Not run**: not installed (documented pre-existing tooling gap).
- [x] Smoke test (pytest `smoke` marker) — pass (`tests/lstm_gat_hybrid/test_diracc_per_ticker_fix.py`).
- [x] Live end-to-end run for `train_parallel_enhanced.py` — **done**, 5-epoch `--quick_test`,
      confirmed real-run DirAcc changed from 71.57% (old formula) to 48.14% (corrected).
- [ ] Live end-to-end run for `train.py`/`train_parallel.py` — **Not run**, no quick-test mode
      exists for these 2 files and adding one is out of this fix's scope (see above).
- [x] Impact analysis — grepped for other `evaluate_predictions(` callers outside `baselines/`
      (see below) to confirm no other in-scope caller was missed.
- [x] Similar-pattern check — see below.

## Impact / similar-pattern check

`grep -rn "evaluate_predictions(" src/` (excluding `archive/`) found several other **live,
genuinely multi-stock** callers still passing a flattened array without `n_stocks` — these were
**out of scope for this task** (user named exactly 3 files) and were **not modified**, but per
CLAUDE.md's "similar check" rule (don't fix 1/N copies silently) they are listed here as follow-up:

- `src/lstm_gat_hybrid/sanity_constant_baseline.py:155` — `n_stocks = len(stock_names)` is already
  computed at line 106 (same pattern as the 3 fixed files); a one-line `n_stocks=n_stocks` addition
  would apply directly.
- `src/lstm_gat_hybrid/validate_fixed.py:127,144` — same flattened multi-stock pattern
  (`batch_size, num_stocks = y.shape` at line 46), but this file is **not imported anywhere**
  (`grep -rl "validate_fixed"` only matches itself) — appears to be a reference/scratch file, not
  live code.
- `src/lstm_har_gat_hybrid/train_hybrid.py:254` — a different model family (LSTM-HAR-GAT hybrid,
  `num_stocks=30` default), flattens `(batch, num_stocks, 1)` predictions/targets before calling
  `evaluate_predictions`; ordering not verified against its `custom_collate_fn` — would need the
  same ordering-contract check as this task did before adding `n_stocks=`.
- `src/timesnet_baseline/train.py:239` — dataloader yields `(x_har, x_temporal, y)` directly with
  no `batch_size, num_stocks` reshape step visible in `validate()`, suggesting this pipeline may
  train per-ticker (single stock at a time) rather than jointly — **unconfirmed**, would need
  checking its dataset class before deciding whether `n_stocks` even applies.

Single-ticker pipelines (`har_baseline`, `lstm_baseline`, `lstm_har_baseline`, `lstm_har_enhanced`,
`cryptomamba_baseline`, `experiment/*`, `timesfm_baseline`) were not flagged — `evaluate_predictions`
without `n_stocks` is the correct call for those (one ticker's chronological sequence).
