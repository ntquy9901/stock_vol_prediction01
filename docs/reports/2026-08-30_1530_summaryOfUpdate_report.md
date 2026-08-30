# Summary of update — walk-forward HAR-X vs no-graph LSTM (VN100 h1)

## What changed
New gated baseline `baselines/2026-08-30_walkforward_harx_lstm/` (5 sub-folders, SDD): an
expanding-window, periodic-retrain (K=66) walk-forward of HAR-X vs the delivered no-graph LSTM over the
same 454-date VN100 h1 OOS region, to test whether periodic retraining changes the fixed-split verdict.

## Result (headline)
- Fixed split: LSTM QLIKE 0.5784 vs HAR-X 0.5115, DM p=1.14e-3 → LSTM significantly WORSE.
- Walk-forward: LSTM (5-seed ensemble) QLIKE **0.4965** vs HAR-X 0.5074, DM p=**0.372** → no
  significant difference (point estimate marginally favours LSTM; LSTM also wins MSE/MAE/R²).
- **Verdict changes:** periodic retraining removes HAR-X's significant advantage; it was a
  single-training-window artifact, not structural. Full write-up:
  `docs/reports/2026-08-30_walkforward_harx_lstm.md`.

## Files
- `code/wf_folds.py` — expanding-window fold construction + `assert_no_leakage` guards.
- `code/wf_panel.py` — unsplit VN100 5-feature panel (reuses `masked_rich` helpers read-only) +
  `pack_fold` with TRAIN-ONLY per-node scalers.
- `code/run_walkforward.py` — pooled runner (HAR/HAR-X OLS + reused no-graph `train_masked_rich`
  LSTM), date-clustered DM, over/under-fit evidence, GPU-politeness poll, CLI.
- `test/test_walkforward_folds.py`, `test/test_walkforward_panel.py`, `test/test_walkforward_runner.py`
  — 27 tests (fold/leakage guards, train-only-scaler leakage, stubbed runner, GPU helpers).
- `requirements/requirements.md`, `design/design.md`, `code_review/code_review_2026-08-30.md`.
- `results/walkforward_harx_lstm/walkforward_vn100_h1.json` — full metrics/DM/evidence.

## Tests + coverage
- 27 passed under `.venv_gpu_encode`. Diff-scope coverage: **100% line + 100% branch** on the three
  changed modules (`--cov-branch`). Real-data-sample smoke included (8-ticker VN100 slice).

## Code review
- Adversarial 3-lens + dedicated leakage lens (`code_review/code_review_2026-08-30.md`). No HIGH/MAJOR
  findings. One documented MINOR: train↔val tail contiguity (val is early-stop-only; the scored
  forecast region is fully purged from train and val) — accepted per the approved design spec.

## Performance
- LSTM trains batched `[B,102,10,5]` (batch 32) on GPU via the reused `train_masked_rich` (no batch=1,
  no per-item main-thread loop); HAR/HAR-X vectorised OLS. GPU politeness poll held the run until
  another agent's GPU job finished (util<15, VRAM<1200MiB, 3 consecutive samples), then ran in 1243 s
  on one RTX 4060. Folds are sequential by necessity (expanding window + single shared GPU).

## Data-quality gate
- N/A (no data change) — this baseline only READS the delivered processed VN100 panel + raw OHLCV;
  no crawl/append/reprocess.

## Over/under-fit evidence
- Per fold: seed-ensembled train/val/test metrics + `classify_fit` verdict + per-seed learning curves,
  plus a pooled `fit_summary`. 6/7 folds "ok" for every model; fold 1 flagged "overfit" for ALL three
  models (HAR/HAR-X/LSTM) — an early-OOS regime effect, not LSTM-specific.

## Risks / follow-ups
- DM is date-clustered but not fully HAC-corrected across fold boundaries (stated caveat); p-values
  approximate. Single market/horizon. The equivalence claim (not "LSTM wins") is the defensible one.

## DoD checklist
- [x] SDD artifacts (requirements/design) written before code
- [x] 5 sub-folders present; code runs; tests pass (pytest)
- [x] Leakage test asserts train-only scaler + purged forecast dates
- [x] 100% line / 100% branch diff-coverage; ruff --select F clean
- [x] Adversarial 3-lens review done, findings addressed
- [x] Over/under-fit evidence captured
- [ ] Commit + push through pre-push gate (next)
