# Code Review — leaf-graph FINAL paper-grade baseline (2026-09-18)

Adversarial 3-layer review (Blind Bug Hunter + Edge Case Hunter + Acceptance Auditor) of the NEW code in
`baselines/2026-09-18_leaf_graph_paper/` only, compared against the committed, validated proof-of-concept
`baselines/2026-09-18_gbm_leaf_graph` (947296f1). The committed baseline was NOT edited.

## Verdict
No HIGH/MAJOR defects. The baseline is a faithful adaptation of the validated method with the model set
restricted to `{XGB, XGB+leafgraph}` (GBME removed), 3 seeds, and a full/no-earn feature-set switch. All
high-risk acceptance criteria verified:

- **Model set** — `FM.gbm`/`GBME` appear only in docstrings/comments, never called; `ORDER == ["XGB",
  "XGB+leafgraph"]` (test-asserted, `GBME` absent from `doc["metrics"]`).
- **Causality** — booster fit on `trf_e` (train minus the trailing `VALID_LEN`-date val slice); alpha fit on
  OOS val QLIKE, frozen for test; `smooth_all` groups strictly by date (no cross-day mixing, no future rows).
  `test_smoothing_is_per_day_causal` verifies the per-day invariant.
- **`_Stream` sufficient statistics** — `ss_tot = syy - sy^2/n`, `r2 = 1 - sse/ss_tot`;
  `test_stream_matches_direct_pooled_metrics` confirms all 5 metrics equal `_metrics5` to rtol 1e-9.
- **Feature-set switch** — `FM.panel` dropna is on `OWN + ["y"]` only, so full vs noearn share identical rows;
  `resolve_cols` drops all 4 EARN cols for noearn (3 tests). Graph-isolation invariant added:
  `test_fold_predictions_shares_base_and_alpha0_isolates_graph` asserts XGBLG reuses the XGB base and is
  bit-identical at alpha=0.
- **Evidence gate** — both models are `looks_learned` (contain "xgb"); doc carries train/val/test blocks for both.
- **Numerical guards** — floor `FL` + `PRED_CAP` clip on every prediction path; zero-variance r2 -> 0.0 in both
  `_Stream` and `_metrics5`; degenerate/ValueError DM -> p=1.0.
- **Performance** — no batch=1 anti-pattern; XGBoost trains on the full matrix; `day_similarity` is a sparse
  one-hot inner product (not an m x m x T broadcast); streaming accumulator bounds train-array memory.

## Findings and resolution

| Sev | Location | Problem | Resolution |
|-----|----------|---------|-----------|
| MINOR | run:264 embargo | `int(h*1.6)+5` hardcoded tunable constants inline (single-source-of-truth rule). Inherited from the reference. | FIXED: added `EMBARGO_MULT=1.6`, `EMBARGO_BUFFER_DAYS=5` to config; driver references `C.` |
| MINOR | main success print | `success()` needs both `KILL_HORIZONS=(1,5)`; under the one-horizon-per-process workflow the console always printed `False` (JSON unaffected). | FIXED: print the success line only when `set(KILL_HORIZONS) <= set(docs)` |
| NIT | `_Stream` ss_tot | Pooled `syy - sy^2/n` is cancellation-prone vs `_metrics5`'s stable `Σ(y-ȳ)²`; ~1e-12 relative error on real data, negligible. | No change (tested equal to rtol 1e-9; not a practical problem) |
| NIT | evidence smoke | Asserts blocks EXIST, not `fit=='ok'` (tiny synthetic folds may legitimately overfit; real-data fit is enforced by the push gate). | Accepted limitation; real-data JSONs verified to pass `check_overfit_evidence` |
| NIT | seed-0 double fit | Seed-0 booster is fit once in `predict_xgb` and again for `leaf_matrix`. | Kept for parity with the validated reference (leaves must match the seed-0 ensemble member) |
| NIT | `code/` package name | Shadows stdlib `code`; run tests from repo root (conftest seeds paths) as the pre-push gate does. | Shared repo idiom; no change |

## Post-fix verification
- `.venv_gpu_encode` pytest: 33 passed.
- Diff-coverage on new code: C0 line = 100%, C1 branch = 100% (entry `main()` + path bootstrap are `# pragma:
  no cover`).
