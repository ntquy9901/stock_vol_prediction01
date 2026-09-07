# Code review — HARQ walk-forward baseline (2026-09-07)

Adversarial 3-lens review (subagent) of `code/harq_walkforward.py` + `test/test_harq.py`, cross-checked
against the delivered `wf_enriched_panel.pack_fold`, `run_walkforward._har_ols_preds`, `run_masked_rich`
(`_pred_dict/_metrics/_dm_all`) and `stats.date_clustered_dm`.

## Verdict: NO CRITICAL. The "HAR-X-Q beats HAR-X" claim is not invalidated by any leakage/alignment/floor bug.
Verified CLEAN:
- **HARQ causality/alignment:** `quarticity_panel` uses backward `rolling(QWIN, min_periods=1)` on `pk^2`
  (days ≤ t); `feats[:,:,0] == panel.pk` (channel 0 copied before volume imputation), so the HARQ daily term,
  `har_daily`, and target `pk[t+h]` share the same series; `harq[panel.anchors[fold.train]]` extracts at the
  exact anchors `pack_fold` uses for `har5` — no off-by-one.
- **Row-order/mask parity:** `har5_tr.reshape(-1,5)[mtr.reshape(-1)]` and `hq_tr.reshape(-1)[mtr.reshape(-1)]`
  flatten in the same anchor-major/node-minor order and apply the identical `tmask_tr`; the HAR-X leg is
  algebraically identical to canonical `_har_ols_preds` (verified byte-for-byte design/lstsq/reshape).
- **Floor parity:** both models use the same per-node `POS_FLOOR_FRAC*mean+EPS` and the same QLIKE floor.
- **DM orientation:** `_dm_all(HAR-X-Q, HAR-X)` → `favors="A"` iff `mean_diff<0` = HAR-X-Q better. Correct.

## Findings and resolutions
| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| 1 | MAJOR (test guardrail) | No test proved the HARQ column actually changes predictions — `HAR-X-Q <= HAR-X + 1e-6` passes even if HARQ were zeroed/scrambled | **Fixed**: (a) new unit `test_appended_harq_column_changes_predictions` — on synthetic data where the target depends on the HARQ column, asserts HAR-X-Q predictions differ from HAR-X AND fit better; (b) smoke now asserts `HAR-X-Q < HAR-X` STRICTLY and `dm mean_diff < 0` (both fail if the column were inactive). |
| 2 | MINOR (test fidelity) | Smoke ran `folds_target=3` but asserted against the canonical 7-fold HAR-X | **Fixed**: smoke now runs `folds_target=7` (exact reproduction) and asserts `abs(HAR-X − 0.5607) < 0.002`. |
| 3 | MINOR (note) | Canonical `0.5607` embedded as a literal | Kept as a tight tolerance at the 7-fold cadence where reproduction is exact by construction; acceptable (a test expectation, not a pipeline tunable). |

## Post-fix verification
- Tests 6/6 pass; **100% line + 100% branch** coverage on `code/`.
- ruff `--select F`: clean. No CRITICAL/unresolved MAJOR remain.
