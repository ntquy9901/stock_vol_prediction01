# Summary of update — retrain-on-(train+val) variant + robustness CI reporting

## What changed

Two parallel robustness studies on the volatility-GAT leave-one-out ablation, both scored on the existing
held-out test with no leakage:

1. **Retrain-on-(train+val) variant (Task 1).** Merge train+validation into one training set, keep the
   same held-out test, retrain all six rungs (HAR, FULL, minus_graph, minus_gate, minus_news,
   LSTM_only) with a fixed 9-epoch budget (no early stopping, no test-based selection), and re-score on
   test. Diebold-Mariano (HLN, HAC lag h−1) FULL-vs-each-rung across horizons 1/5/10/22, seed 42.
2. **Per-year / per-ticker / block-bootstrap CI reporting (Task 2).** No training; reuses the existing
   5-seed test dumps to report per-year metrics, per-ticker HAR-vs-FULL distribution, and moving-block
   bootstrap 95% CIs for FULL−HAR.

## Files

| Path | Purpose |
|---|---|
| `baselines/2026-08-15_volatility/code/run_retrain_trainval.py` | Retrain-on-(train+val) runner (fixed epochs, HAR refit on train+val, test-only eval) |
| `baselines/2026-08-15_volatility/test/test_retrain_trainval.py` | Smoke test for the runner |
| `baselines/2026-08-15_volatility/code/dm_retrain.py` | DM (HLN) for the retrain dumps; reuses `dm_report` loss families |
| `baselines/2026-08-15_volatility/test/test_dm_retrain.py` | Unit test (reads retrain dirs, seed-ensemble) |
| `baselines/2026-08-15_volatility/code/robustness_report.py` | Per-year / per-ticker / block-bootstrap over existing dumps |
| `baselines/2026-08-15_volatility/test/test_robustness_report.py` | 5 unit/smoke tests |
| `docs/paper/explainers/retrain_trainval_report.md` | Task 1 report (metrics + DM + reading + caveats) |
| `docs/paper/explainers/robustness_test_report.md` | Task 2 report (per-year/ticker/CI) |
| `results/volatility_retrain_h{1,5,10,22}_seed42_2026-08-15_182005_retrain/` | Raw metrics + per-obs test dumps |

## Key results (held-out test)

**Retrain regime (seed 42), FULL vs HAR by loss family (DM sign: negative favors FULL):**

| Horizon | QLIKE | SE (MSE/RMSE/R²) | AE (MAE) |
|---|---|---|---|
| h1 | FULL wins (p<.01) | FULL wins (p=.02) | FULL wins (p<.01) |
| h5 | tie (p=.36) | tie (p=.12) | FULL wins (p=.02) |
| h10 | HAR wins (p<.01) | tie (p=.74) | tie (p=.44) |
| h22 | HAR wins (p<.01) | tie (p=.15) | FULL wins (p=.02) |

**Change vs the primary train-only regime:** at h1, folding validation into training makes FULL beat
HAR on all three loss families (the primary 5-seed study had the FULL−HAR CI including zero at h1).
HAR keeps its QLIKE edge at h10/h22; LSTM_only remains the strongest deep configuration at h5. No
single model dominates HAR across all horizons — the parsimony reading holds, with the h1
qualification.

**Robustness CIs (5-seed, primary regime):** every FULL−HAR block-bootstrap 95% CI includes 0 (no
horizon significant); FULL-lower-QLIKE ticker share declines with horizon (h1: 20/33 → h22: 8/33).

## Commands run

- Training (GPU venv): `run_retrain_trainval.py 2026-08-15_182005_retrain cuda 42 9 {1 5 | 10 22}` —
  two parallel processes; basis built cleanly (train≈73k obs = train+val), no Traceback.
- DM: `dm_retrain.py 2026-08-15_182005_retrain {h} 42` for h in 1/5/10/22.
- Tests: `pytest test/test_dm_retrain.py` → 2 passed; `pytest test/test_retrain_trainval.py` → 2 passed;
  `pytest test/test_robustness_report.py` → 5 passed.
- Lint: `ruff check` on new files → clean.

## Code review

- Retrain runner and DM script were self-reviewed and mirror the already-reviewed `run_ablation.py` /
  `dm_report.py` patterns (same rung spec, same DM machinery, same shared QLIKE floor, no test-based
  selection). `/code-review` (3-layer adversarial) not run in this session — flagged as a follow-up
  along with `diff-cover` C0/C1 (diff-cover not installed in this env).

## Risks / follow-ups

- **Single-seed retrain.** DM p-values are seed-42 only. The h1 FULL-over-HAR reversal is large
  (dm ≈ −5.4) but a 5-seed retrain (42/123/2026/7/2024) is the appropriate confirmation before citing
  it as a headline. Follow-up if desired.
- **Fixed 9-epoch budget** (no held-out set to tune epochs); avoids leakage but is untuned.
- `/code-review` + `diff-cover` C0/C1 pending (tooling gap).

## DoD checklist

- [x] Code satisfies request (retrain variant + CI reporting), surgical (isolated new files).
- [x] Tests written + pass (2 + 2 + 5).
- [x] Lint clean (ruff).
- [x] Real measured results captured (test dumps + DM + CI).
- [x] Summary report (this file).
- [x] Push after task (branch code pushed; docs pushed to master — see commit).
- [ ] `/code-review` 3-layer + diff-cover C0/C1 — follow-up (tooling / not run this session).
