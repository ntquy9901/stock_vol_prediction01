# Adversarial code review — LSTM-feature / GBME-blend baseline (2026-09-18)

Three-lens adversarial pass (Blind Hunter / Edge-Case Hunter / Acceptance Auditor) over `lstm_feat.py`,
`run_lstm_blend.py`, `lstm_blend_config.py` and the test suite, with a performance lens (train/inference code).
Scope excludes `archive/`. All findings triaged; no critical/major left open.

## Blind Hunter (hidden bugs)

- **B1 (fixed before first run) — whole-set LSTM forward OOM.** The initial `torch_trainer` / early-stop eval
  forwarded the ENTIRE prediction/eval set through the LSTM in one call; on a real HOSE train window this tried
  to allocate ~20 GiB on an 8 GiB GPU (`CUDA out of memory`). Fixed with `_forward_batched` (chunked inference
  in `C.BATCH` steps) used for Xva, Xtr(record) and Xpred. Training was already batched. Re-verified: real HOSE
  smoke and full run complete without OOM. Performance-lens finding — resolved.

- **B2 — stacking-leakage / in-sample train feature.** A naive design would feed the in-sample train LSTM
  prediction to the GBM, inflating train fit and tripping the overfit gate. Mitigated by OUT-OF-FOLD inner
  temporal K-fold for the train feature (`oof_train_feat`) + a full-train LSTM only for the test feature
  (`test_feat`). Verified by `test_oof_coverage_and_causality` (train dates disjoint from the block each row is
  predicted on) and by the pooled full-run `fit_diagnostics=ok` (val->test gap 0.075, not the 0.47 seen in a
  single-fold smoke).

- **B3 — sequence causality.** `build_sequences` builds each row's window from its own ticker's PAST feature
  rows only (`block[lo:j+1]`, left-padded), so no future feature enters a sequence. Outer-fold causality is
  additionally proven by `test_outer_fold_causality` (mutating rows at/after the test end leaves fold-0 metrics
  bit-identical).

## Edge-Case Hunter

- **E1 — degenerate early-stop split.** `_group_pred` raises `ValueError` when the train window has too few
  dates to hold out a validation slice (covered: `test_group_pred_raises_on_degenerate_val`).
- **E2 — OOF coverage gap.** `oof_train_feat` fails loud (`RuntimeError`) if any train row is left unembedded
  (covered by monkeypatching `inner_blocks` to drop dates).
- **E3 — DM degeneracies.** `_safe_dm` returns p=1.0 for identical loss series (tree ignores lstmfeat -> loss
  series equal) and for the h>=n_dates HLN failure (covered).
- **E4 — exp/link overflow.** `_to_variance` double-clips (log-space then variance-space) to `[FL, PRED_CAP]`
  so an extreme log-variance cannot overflow `exp` (covered: `test_to_variance_clips_both_sides`).
- **E5 — undefined correlation.** `_err_corr` returns 0.0 when either series is (near-)constant (covered).
- **E6 — all-test-in-spike.** `spike_robustness` is omitted when the exclusion mask removes every test row
  (covered: `test_run_hose_all_in_spike`).
- **E7 — has_earn on/off.** Both the earnings and no-earnings feature paths are exercised
  (`test_run_hose_with_earnings...` vs sp500 fake loader).

## Acceptance Auditor

- Meets requirements: GBME vs GBME+lstmfeat vs convex val-fit blend, all 4 horizons, date-clustered DM,
  spike-robust, err-correlation reported, pre-registered kill criterion in config.
- Gate-compliant evidence: learned model name `GBME+lstmfeat` matches `overfit_check._LEARNED_PATTERNS`
  ('lstm'); result JSONs carry `metrics`/`train_metrics`/`val_metrics` (all 5) + `fit_diagnostics` +
  `learning_curves`. Non-learned `GBME`/`blend` are correctly exempt from the fit gate.
- Config isolation: uniquely-named `lstm_blend_config.py` (NOT bare `config.py`) — avoids the sibling
  sys.modules collision; OWN-8 loaded by path (`paper_models_config`) for the same reason.
- Single-source-of-truth: every tunable lives in `lstm_blend_config.py` with inline comments; GBM hyper-params
  + FL reused from `full_matrix`. config-hardcode scan: 0 BLOCK (1 WARN = the `1e-6` early-stop tolerance, a
  local numerical guard, matching the sibling embed.py).
- Tests: 27 pass; diff-cover-equivalent C0=100% / C1=100% on both new modules (0 missed statements, 0 partial
  branches). Fake-trainer driver tests + one real-torch smoke + real-torch unit tests for `train_lstm`.

## Verdict
No critical/major findings open. B1 (OOM) fixed and re-verified; B2 (leakage) handled by OOF and confirmed by
the pooled fit diagnostics. Minor/no-action: the `1e-6` WARN (numerical tolerance), and E702 semicolons in the
test file (house-style, WARN-only per CLAUDE.md).
