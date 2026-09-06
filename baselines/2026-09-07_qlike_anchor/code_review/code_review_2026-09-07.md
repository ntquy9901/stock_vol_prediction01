# Code review — qlike_anchor (2026-09-07)

Scope: qa_train.py (trainer + helpers), run_qlike_anchor.py (driver), qa_config.py. Layers: correctness,
leakage, fairness, isolation.

## Findings
- **Leakage (clean):** anchor train target uses OOF HAR-X (oof_harx, fit strictly on earlier anchors); val/test
  use the fold's train-fit HAR-X. assert_no_leakage(purge=horizon) enforced. Features causal (delivered panel).
  Warm-up in-sample fallback affects only train-cell residual targets, never val/test.
- **Fairness (clean):** identical folds/seeds/scaler/positivity floor as edge_hmatched; QLIKE floor = training_config.
  Smoke reproduces HAR/HAR-X to the delivered values (HAR 0.4781). Same QLIKE definition (RMR._metrics/_dm_all).
- **Positivity (clean):** linear output + inverse-transform + shared floor (no Softplus/ReLU collapse). anchor=harx
  clips |z|<=c so a spike cannot collapse far below HAR-X; z=0 -> HAR-X.
- **Isolation (§3.F, clean):** delivered modules imported read-only; no other baseline/src modified.
- **Coverage:** pure helpers 100% C0 (7 tests); train_deep/run/main are # pragma: no cover GPU/driver glue,
  smoke-verified end-to-end on vn30 h1 for (qlike,harx) and (mse,none) — both produce positive floored forecasts,
  DM, and cell logs.

## Status
No critical/major defects. Smoke-verified. Full VN100 matrix + honest GO/NO-GO to follow.

## Round 2 — external read-only review findings (all fixed)
Two independent adversarial reviews (coordinator agent + external reviewer) converged. Findings + fixes:
- **P1 OOF fallback (leakage-safe claim):** the warm-up in-sample fallback `np.where(isfinite(oof), oof, harx["tr"])`
  made ~30% of the residual training target the in-sample fit. FIX: keep NaN warm-up OOF and MASK those cells
  out of the loss (`valid_tr` in qa_train.train_deep); the in-sample HAR-X (`tr_base`) is now used only as the
  forecast base for the TRAIN split's fit-evidence, never as a training target. Residual target is fully OOF.
- **P1 QLIKE floor inconsistency:** train loss + early-stop used 1e-12 while the final metric used
  cfg.qlike_floor=1e-8. FIX: single `split_objective(y,f,mask,loss,floor)` helper threads cfg.qlike_floor
  through the batch loss, early-stop, and learning curve — train == select == report.
- **P2 learning curve objective:** curves recorded MSE even when loss=qlike. FIX: train_curve/val_curve now
  record the training objective (QLIKE when loss=qlike) via split_objective.
- **P2 EDA silent dedup:** scripts/eda/lstm_harx_cell_gap.py pivoted with aggfunc='first'. FIX: assert_unique_cells
  raises on duplicate (model,ticker,date) before the pivot.
- Also: qa_config no longer redeclares OOF split/warmup (they live in xgb_config, single source); anchor=harx
  runs now record oof_splits/oof_warmup_frac in provenance; empty val-mask now fails loud.
All re-verified: 16 tests pass; smoke (vn30 h1 qlike/harx) runs end-to-end with the fixes.

Note: `.venv_gpu_encode` works on this machine (pytest + smoke ran); the external reviewer's "no Python at
Python310" was an environment issue on their side, not a repo defect.
