# Code review — Refined SP500 champion (2026-09-11)

Adversarial 3-layer review (Blind Hunter / Edge Case Hunter / Acceptance Auditor), run by an independent reviewer
agent on `code/gbm_refined_walkforward.py` + `test/test_gbm_refined.py` + requirements/design.

## Headline verdict: SAFE TO COMMIT
**No CRITICAL or MAJOR leakage or correctness bug.** ruff `--select F` PASS; pytest PASS (8/8 after the fix below).

## Correctness verified (the risky items)
1. **Dual index-space wiring in `_design_refined`** — CORRECT. `run()` passes `anchors=panel.anchors[fold.*]`
   (date-axis `[T,N]`) for HAR/extras/estimators/spike and positional `pos=fold.*` (anchor-axis `[A,N]`) for
   earnings. Traced through `pack_fold`: `est[anchors][i]`, `spk[k][anchors][i]`, `asym[k][pos][i]` all line up
   row-for-row with `har5` row `i` (row-major `anchor*N+ticker`). No off-by-one, no wrong-axis, no future pull.
2. **Estimators aligned to ORIGIN t** (not target t+h) — CORRECT; columns present in real data (no silent all-NaN).
3. **Spike features causal** — trailing rolling ops, per-ticker columns, NaN warm-up handled natively by HistGBM.
4. **Asymmetric earnings forward-looking** — distances from target t+h to SCHEDULED dates known at t; identical
   mechanism to the committed 2026-09-09 champion.
5. **Overfit-evidence train QLIKE** — `ptr`/`ytr` masked by the same `mtr`; formula matches `per_obs_qlike`.
6. **No leakage into floors/fit** — train-only scalers (`t_mean`), GBM fit on train rows only, HAR-X OLS train-only,
   `assert_no_leakage` invoked, all test models share one `nf_te` floor (apples-to-apples DM).

## Findings and disposition

### MAJOR (test adequacy) — FIXED
The dual index-space (the one advertised risk) was only shape-tested; an `anchors`<->`pos` swap for the
estimator/spike block would have passed. **Fixed:** added `test_design_alignment_values` — builds distinguishable
per-(index,ticker) values and asserts `_design_refined` places the ORIGIN-date estimator/spike value and the
positional earnings value into the correct columns; an anchors/pos swap now fails the test. pytest now 8/8.

### MINOR — accepted (documented, not changed)
- **Train vs test QLIKE use different prediction floors** (train floored at `QLIKE_FLOOR`, test at the higher
  per-node `POS_FLOOR_FRAC*t_mean+eps`). This marginally biases the `train_to_test_gap_pct` DIAGNOSTIC only; it is
  NOT a leakage/correctness issue (all *test* models share `nf_te`, so the go/no-go QLIKE + DM are unaffected).
  Not changed: a fix would require a full 4-horizon re-run to keep the committed JSONs consistent with the code,
  for a marginal diagnostic that does not affect the decision. Accepted.
- **Earnings-date provenance** (yfinance realized dates assumed scheduled/known at t) — inherited, identical to the
  committed champion, explicitly accepted in requirements.md §Leakage constraints.
- **No auto aggregate go/no-go** — the criteria inputs (per-model QLIKE, `dm_vs_HARX`, `dm_vs_GBMearn`,
  `overfit_evidence`) are all emitted per horizon; the cross-horizon verdict is a manual read of 4 JSONs (matches
  the committed pattern). Accepted.

## Result (all inputs to the go/no-go, from the committed JSONs)
GBM+refined beats GBM+earn at all 4 horizons: h1 +1.96% (p=.000), h5 +1.94% (p=.000), h10 +1.68% (p=.020),
h22 +1.70% (p=.009); beats HAR-X +21.4/+13.6/+12.3/+14.9% (all p=.000); overfit gap −10.0%..+3.3% (no overfit).
Meets the requirements.md GO criteria (beats GBM+earn at 4/4 with DM p<0.05 at all 4; no horizon worsened; gap
< 15%; beats HAR-X everywhere).

## Verdict: GO — safe to commit after the added alignment test (done).
