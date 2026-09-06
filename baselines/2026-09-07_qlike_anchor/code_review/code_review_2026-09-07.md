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
