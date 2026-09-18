# Adversarial code review — Literal GBME + leaf-cooccurrence graph (Option A)

Date: 2026-09-18 · Reviewer: adversarial subagent (cynical / edge-case / acceptance lenses) + author triage.
Scope: `code/hgbr_leaf.py`, `code/leaf_graph.py`, `code/run_gbme_leafgraph.py`, `code/gbme_lg_config.py`,
`test/test_gbme_leafgraph.py`. Reused shared modules (`full_matrix`, `vn_gbm_graph_stage1`, `stats`, `metrics`,
`overfit_check`) treated as correct (out of scope). `archive/` out of scope.

Special scrutiny requested: private-API fragility, leakage, correctness, test quality.

## Outcome
No HIGH. One MAJOR (test quality) and five MINOR — all resolved. Re-verified: 24 tests pass, diff-cov
C0=100% / C1=100% on changed lines, ruff-F clean, config-hardcode gate clean.

## Findings and resolutions

### MAJOR
- **M1 — `test_fit_alpha_picks_min` was vacuous.** The only assertion `a in grid and isfinite(q)` passes even if
  `fit_alpha` ignored QLIKE. `fit_alpha` decides whether the graph is used at all, so its argmin had no real
  coverage. **Fixed:** replaced with `test_fit_alpha_picks_grid_argmin` (asserts the returned alpha equals the
  independently-recomputed per-grid QLIKE argmin, `q == min(qs)`, and `alpha != 0` on a case where the graph
  genuinely helps) plus `test_fit_alpha_tie_breaks_to_smaller` (verifies the documented tie-break keeps the
  smaller alpha).

### MINOR
- **M2 — clip/maximum asymmetry broke graph-isolation at alpha=0.** The smoothed arm was `np.clip(., FL,
  PRED_CAP=1.0)` while the base GBME used `np.maximum(., FL)` (no cap), so at `alpha=0` the two arms differed
  for any prediction > 1.0 — the alpha->0 NO-GO could show a nonzero gain from the cap, not the graph. In
  practice the target is variance (~1e-4) so it stayed inert, but the arms were not identical-by-construction.
  **Fixed:** the smoothed arm now uses `np.maximum(smoothed, FL)` (same floor as base, no upper cap; the
  smoothed value is a convex combination of >=FL gamma predictions and HGBR predict is bounded, unlike an XGB
  exp-link). `PRED_CAP` removed (orphan constant), design.md updated. `alpha==0` is now exactly GBME.
- **M3 — reconstruction guard proves leaf-VALUE sum, not leaf-INDEX identity.** The graph consumes indices; the
  guard checks values. Collision-safe only because gamma leaf values are effectively unique floats. **Fixed
  (doc):** `leaf_matrix` docstring now states index-correctness is guaranteed by the exact `num_threshold`
  traversal and the guard is a defensive cross-check.
- **M4 — `_traverse_tree` ignores `missing_go_to_left` (NaN routing).** A NaN feature would mis-route (NaN goes
  right) vs HGBR, tripping the reconstruction guard and aborting (fail-loud, acceptable). **Fixed (doc):**
  documented the NaN-free precondition (panel dropna) and the fail-loud behavior.
- **M5 — `test_leaf_matrix_reconstruction_ok` "independent recompute" wording overstated.** It recomputes via
  the same `_traverse_tree`, but compares against sklearn's `_raw_predict` ground truth (so a traversal bug is
  still caught). **Fixed (comment wording).**
- **M6 — `test_fold_predictions_causal_slicing` checked only lengths.** A te/va offset swap in `sl()` would
  preserve equal lengths. **Fixed:** distinct block sizes (30/200/70) + per-row content check that each split's
  base equals `predict_gbme` on that exact source frame.

## Verified clean (no finding)
- **Leakage:** GBME + leaf-matrix fit on `trf_e` only (val excluded); `combo=[tef,trf_e,vaf]` order matches the
  `sl()` offsets consistently for `dates`, `X`, `p_gbme`, `leaves`, and the `yy`/`dts` accumulation; alpha fit
  on the `va` slice only and frozen for all splits; `val_dates` = last `VALID_LEN` train dates, with an embargo
  gap to test; per-day smoothing is contemporaneous (grouped by unique date, no cross-day mixing).
- **Private-API extraction:** raw `num_threshold` `<=` traversal matches HGBR numeric routing; reconstruction to
  ~1e-15 (link space) verified on real fits (sklearn 1.7.2); fail-loud guard each fold pins against layout drift.
- **Correctness:** `_traverse_tree` active-mask loop terminates and lands every sample at a leaf (incl.
  single-leaf root); `day_similarity` per-tree id offset makes `(S@Sᵀ)/T` exact Hamming similarity; kNN
  self-exclusion + singleton short-circuit; `_safe_dm` degenerate (identical loss / too-few dates) handling;
  `_pool_doc` per_fold/spike/keep-empty branches.
- **Evidence gate:** `train_metrics`/`val_metrics`/`fit_diagnostics` emitted per model.

## Performance lens
Leaf traversal vectorised over samples (no per-item Python loop); leaf-matrix extracted from seed[0] only;
per-day smoothing via sparse one-hot matmul; `alpha==0` identity fast path; one horizon per process; per-fold
arrays freed (`del fp`). No batch=1 anti-pattern in the hot path.
