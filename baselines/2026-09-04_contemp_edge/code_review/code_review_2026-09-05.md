# Code review — contemporaneous-edge ablation (2026-09-05)

Adversarial review (general-purpose subagent) of `run_contemp.py` (new edge logic + scoring) and the
paper v3/v4 contemporaneous-edge section against the source JSONs. `archive/` out of scope.

## Findings and resolution

**MAJOR — paper prose contradicted its own VN30 table (RESOLVED).** v3 stated HAR/HAR-X hold the best
QLIKE "at every horizon" on VN30; false at h10 where the no-graph LSTM has the lowest QLIKE (0.6371 vs
HAR-X 0.6385). Fixed in v3/v4: "at three of the four horizons (the no-graph LSTM edges narrowly ahead
at h10, not significant)."

**MINOR — wrong favoured-model direction, VN100 h10 QLIKE (RESOLVED).** v3 prose said the no-graph LSTM
is favoured at h10/h22; per the JSON VolGA is favoured at h10 (p=0.221), only h22 favours LSTM
(p=0.849). Both non-significant, so the verdict is unchanged. Fixed in v3/v4.

**MINOR — positivity floor hardcoded instead of imported (RESOLVED).** `run_contemp.py:92` used the
literals `1e-2 * D.t_mean + 1e-12` for the HAR/HAR-X floor while the deep models floor with
`pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS`. Numerically identical today
(`pc.POS_FLOOR_FRAC=1e-2`, `pc.POS_FLOOR_EPS=1e-12`) so results are unaffected, but a latent H2
floor-mismatch risk. Fixed: import `pipeline_config as pc` and use the constants (single source of
truth). Results unchanged (identical values).

## Checks that passed (no finding)
- **Leakage:** `_fold_adj`'s `last_tr_row = last train anchor + horizon` and `build_contemp_adj`'s
  `sqrt_pk[:last_tr_row+1]` mirror the delivered edge convention (`wf_enriched_panel.py`); the `+horizon`
  is the last train *target* row (supervised at train time), and `assert_no_leakage` guards it. Edge uses
  train rows only. Unit test `test_train_only_no_lookahead` confirms post-cutoff rows do not affect A.
- **Floor consistency in scoring:** all four models scored with the same `fl = cfg.qlike_floor` in
  `_metrics` and both `_dm_all` calls.
- **Boundary (past bug H1):** `make_folds`/`assert_no_leakage`/`pack_fold` are reused, not
  reimplemented; boundary is global via `frozen_universe`.
- **DM basis:** qlike/se/ae computed on the same key-intersected pooled ensemble preds; favors "A"=first
  model, matching the paper's convention.
- **All table numbers correct:** every cell of tab:edgemetrics(-vn30) and tab:dm-edge(-vn30) matches the
  JSONs; per-column/horizon bolds are the true minima; DM bold exactly when p<0.05.
- **Config consistency:** all 8 contemp JSONs have n_folds=7, seeds=[42,123,2026].

## Tests
`baselines/2026-09-04_contemp_edge/test/test_contemp_adj.py` — 7 tests (shape/self-loop, top-k count,
picks correlated sources, train-only no-lookahead, NaN-safe, both `_fold_adj` branches). 7/7 pass under
the GPU venv. `run()` is a GPU training driver (marked `# pragma: no cover`), exercised by `--smoke`.
