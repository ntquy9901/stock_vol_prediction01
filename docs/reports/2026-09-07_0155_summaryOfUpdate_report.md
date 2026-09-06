# Summary — qlike_anchor: P1/P2 review fixes + train_deep made testable (no QG_SKIP)

## What changed
Fixed all 4 findings from two converging adversarial reviews (coordinator agent + external read-only review):
- **P1 leakage-safe OOF:** warm-up cells (no OOF) are masked OUT of the training loss (no in-sample fallback);
  the residual target is fully OOF. In-sample HAR-X (`tr_base`) is used only as the forecast base for the TRAIN
  split's fit-evidence.
- **P1 QLIKE floor consistency:** one `split_objective(y,f,mask,loss,floor)` helper threads cfg.qlike_floor
  (1e-8) through the batch loss, early-stop, and learning curve (was 1e-12 vs 1e-8).
- **P2 learning curves** record the actual training objective (QLIKE when loss=qlike).
- **P2 EDA dedup:** `assert_unique_cells` fails loud on duplicate cell rows before the pivot.
- **P2 train fit-evidence basis (round 3):** the result JSON now carries `train_metrics_note` stating the train
  metrics use the in-sample HAR-X base over all train cells (incl. warm-up excluded from the OOF loss) and are
  not same-basis as val/test — so the fit-evidence is not misread as an OOS comparison.
- qa_config single-source OOF params; anchor=harx provenance (oof_splits/oof_warmup_frac); empty-val-mask fail-loud.

## Gate resolution (the right way, not a skip)
The pre-push pragma-logic guard first blocked because the P1 fixes added 19 logic lines to `train_deep`, a
`# pragma: no cover` GPU trainer. Rather than QG_SKIP, `train_deep` was made genuinely testable: a CPU
integration test (`test_qa_train_integration.py`) runs it end-to-end on tiny synthetic data for all four
(loss, anchor) configs (torch falls back to CPU), and the function-level pragma was removed. Only three
nondeterministic early-stop glue lines keep a line-level pragma. Result: qa_train.py is 100% C0/C1, the guard
no longer flags train_deep (now WARN-only: 4 lines in run(), 1 in _load_test_cells, both < BLOCK), and no
QG_SKIP is needed.

## Tests
`pytest baselines/2026-09-07_qlike_anchor/test/ scripts/eda/test_lstm_harx_cell_gap.py` -> 20 passed. qa_train.py
100% line+branch. Smoke re-verified (vn30 h1 qlike/harx) end-to-end.

## DoD
Code + tests (incl. executing integration test) + 3-round adversarial code review + this report + gate pass
(no skip). Results (VN100 qlike matrix) to be re-run on the fixed code next.
