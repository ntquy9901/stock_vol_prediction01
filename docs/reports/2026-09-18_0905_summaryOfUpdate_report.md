# Summary of update — LSTM-feature / GBME-blend baseline (2026-09-18)

## What changed
New falsification baseline `baselines/2026-09-18_lstm_gbme_blend/` testing the transferable germ of
arXiv:2505.23084 ("GBDT + LSTM for Investment Prediction") — a deep sequential model (+) tree ensemble —
adapted to this repo's variance-QLIKE thesis on HOSE. A small per-fold LSTM reads the OWN-8 (+earnings)
own-history feature sequence and predicts the h-step-ahead log-variance; that prediction is handed to the
champion gamma-GBM as one extra causal feature (GBME+lstmfeat) and also blended convexly with GBME. Direction
run to CLOSE it concretely on QLIKE, not to fabricate a win.

## Files (path -> purpose)
- `code/lstm_blend_config.py` — single source of tunable constants (SEQ/HIDDEN/LR/WD/BATCH/EPOCHS/PATIENCE/
  MIN_EPOCH/VALID_LEN/LSTM_SEEDS/INNER_K/BLEND_GRID/HORIZONS/MIN_ROWS/PRED_CAP/kill criterion/SPIKE_WINDOWS).
- `code/lstm_feat.py` — `SeqLSTM`, batched GPU train + chunked inference, causal sequence builder, temporal
  early-stop split, leakage-safe inner-block OOF cross-fitting (`oof_train_feat`/`test_feat`), variance clip.
- `code/run_lstm_blend.py` — walk-forward driver: GBME vs GBME+lstmfeat vs val-fit blend, date-clustered DM,
  fit diagnostics, learning curves, standalone-LSTM<->GBME error-correlation, per-fold + spike robustness,
  atomic per-fold checkpoint. Output `results/gamma_gbm/lstm_blend_hose_h<h>.json`.
- `requirements/requirements.md`, `design/design.md`, `code_review/code_review_2026-09-18.md`,
  `test/test_lstm_blend.py`, `test/conftest.py`.
- Results: `results/gamma_gbm/lstm_blend_hose_h{1,5,10,22}.json`.

## Results (HOSE, 8 folds each, seed-ensembled GBM, 5 metrics; QLIKE below)
| h  | n       | GBME QLIKE | GBME+lstmfeat | gain% | DM p  | blend QLIKE | blend gain% | DM p  | err-corr | pred-corr | LSTM-only QLIKE | fit |
|----|---------|-----------|---------------|-------|-------|-------------|-------------|-------|----------|-----------|-----------------|-----|
| 1  | 399,040 | 1.5719    | 1.5727        | -0.05 | 0.225 | 1.5684      | +0.22       | 0.050 | 0.161    | 0.785     | 190.5           | ok  |
| 5  | 397,428 | 1.6479    | 1.6503        | -0.14 | 0.140 | 1.6467      | +0.07       | 0.737 | 0.133    | 0.746     | 183.3           | ok  |
| 10 | 395,413 | 1.6856    | 1.6854        | +0.01 | 0.838 | 1.6870      | -0.08       | 0.841 | 0.105    | 0.710     | 158.7           | ok  |
| 22 | 390,577 | 1.7270    | 1.7264        | +0.04 | 0.486 | 1.7317      | -0.27       | 0.594 | 0.084    | 0.683     | 155.1           | ok  |

Spike-robust (COVID/2022/Apr-2025 excluded): GBME+lstmfeat gain stays within [-0.21%, +0.08%], never
DM-significant. All folds n_folds==8 (verified).

## Verdict — NO-GO (pre-registered kill: beat GBME DM-sig at both h1 and h5, spike-robust)
`GBME+lstmfeat` does not beat `GBME` at any horizon (gain within +/-0.14%, all DM p >> 0.05); it collapses onto
GBME (identical to ~0.05%), i.e. the tree ignores the LSTM feature. The convex blend is marginally positive
only at h1 (+0.22%, p=0.050 borderline) and negative at h10/h22 — no reliable lift. Pre-registered success =
False. Direction CLOSED on HOSE QLIKE.

Mechanism / why null: the standalone LSTM is a strictly-dominated own-history re-encoder — its central
predictions track GBME (prediction-correlation 0.68–0.79, confirming redundancy), while its thin-market tail
occasionally hits the variance cap, giving a catastrophic standalone QLIKE (155–190) and a low error-
correlation (0.08–0.16). Either way it carries no signal orthogonal to the gamma-GBM. This matches, on QLIKE,
the prior repo findings that deep/GNN (+) GBM blends add nothing beyond own-history.

## Tests + coverage
- `.venv_gpu_encode -m pytest baselines/2026-09-18_lstm_gbme_blend/test/` — 27 passed.
- Diff-coverage equivalent (per-module `--cov-branch`): `lstm_feat.py` and `run_lstm_blend.py` both C0=100% /
  C1=100% (0 missed statements, 0 partial branches on changed lines).
- Fake-trainer driver tests + real-torch `train_lstm` unit tests + one real-torch end-to-end smoke; causality
  (mutating future rows leaves fold-0 metrics bit-identical), OOF coverage + fail-loud gap, DM degeneracies,
  clip guards, blend weight, err-corr, spike branches.

## Code review
Three-lens adversarial review documented in `code_review/code_review_2026-09-18.md`. Findings: B1 whole-set
LSTM forward OOM (fixed with chunked `_forward_batched`, re-verified); B2 stacking-leakage handled by OOF (fit
gate confirms pooled val->test gap ~0.03–0.17, status ok); B3 sequence + outer-fold causality proven by test.
No critical/major left open. Minor/no-action: one `1e-6` config-hardcode WARN (numerical tolerance) and E702
semicolons in tests (house-style, WARN-only).

## Performance / batching
LSTM trains on GPU in BATCH-sized steps (no batch=1); inference is chunked so a ~936k-row HOSE train window
does not OOM (an initial whole-set forward tried to allocate ~20 GiB and was replaced by bounded chunks;
standardisation kept in float32 to avoid a float64 copy that exhausted CPU RAM at the largest fold). Runtime:
~8 folds/horizon at ~60–120 s/fold.

## Quality gate
- Overfit-evidence gate on all 4 result JSONs: OK (train/val/test + learning_curves present, learned model
  not over/under-fit).
- config-hardcode whole-file scan on code: 0 BLOCK. ruff --select F: clean.
- Data-quality (Pandera/Evidently): N/A (no data change — reuses existing enriched HOSE panel + crawled
  earnings; no raw ingestion).

## Risks / follow-ups
- SP500 not run (scope was HOSE); the mechanism (LSTM re-encodes own-history) is market-agnostic but SP500
  could be run with the same driver (`run_lstm_blend.py sp500`) if a cross-market check is wanted.
- No git actions performed here — the coordinator pushes.
