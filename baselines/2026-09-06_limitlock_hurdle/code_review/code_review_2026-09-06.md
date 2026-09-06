# Code review — Limit-lock hurdle on HAR-X (2026-09-06)

Adversarial 3-layer review (Blind Hunter / Edge-Case Hunter / Acceptance Auditor). Scope: the three
baseline modules under `code/`. `archive/` and delivered baselines (imported read-only) are out of scope.

## Blind Hunter (hidden bugs)
- **Leakage in the classifier.** Features `ll[anchor]` are causal (prefix-invariance unit-tested). The
  classifier target `zero_range_flag[anchor+h]` uses `wf_folds.make_folds` train/val/purge/forecast
  slices (purge length = horizon), so a train anchor's target never reaches the forecast region. The
  logistic standardization (`mu`/`sd`) is fit on TRAIN rows only, applied to test. No leak found.
- **Floor parity (H2 lesson).** Both compared models apply the identical per-node positivity floor
  `nfloor = POS_FLOOR_FRAC*t_mean + POS_FLOOR_EPS` via `np.maximum`, and the same `qlike_floor` in
  every metric/DM call. On an overridden cell the forecast IS `nfloor` (the floor), so the basis is
  identical. Verified empirically: `qlike_robust` (non-lock cells) is byte-identical across HAR-X and
  every hurdle variant (0.4585), i.e. the override touches only lock cells.
- **Index safety.** `lock[anchor+horizon]` is safe because `build_enriched_panel` sets
  `anchors <= T-horizon-1`, so `anchor+horizon <= T-1`.

## Edge-Case Hunter
- **Single-class train target.** `logistic_fit_predict_proba` returns the constant base rate (0.0/1.0)
  when the train target has <2 classes (no crash) — unit-tested. Graceful no-op.
- **Empty lock subset.** `_metrics_full` returns `qlike_lockdays=None`/`qlike_robust=None` when the
  respective subset is empty (unit-tested), avoiding a metric call on an empty array.
- **NaN returns / flags.** `_clean_returns`/`_clean_flags` map non-finite to 0 (no-move / not-locked);
  off-file dates reindex to NaN then `nan_to_num` in the driver — masked out anyway. Features stay
  finite (unit-tested + real-data smoke on FPT).
- **Identical prediction series -> DM.** When the classifier never fires, the hurdle series == HAR-X,
  so `date_clustered_dm` raises "non-positive long-run variance" (zero differential); caught by
  `_dm_all` and reported as an error/`p_value=None`. Correct behaviour, documented in the report.

## Acceptance Auditor
- Requirements go/no-go criteria are all computed: QLIKE(all)/qlike_robust/qlike_lockdays per model,
  DM vs plain HAR-X, classifier confusion (TP/FP/FN/TN) per threshold, the 2025-04-10 concrete case,
  and an oracle (perfect-classifier) ceiling to separate "idea" from "classifier".
- CPU-only: OLS (`np.linalg.lstsq`) + `sklearn.LogisticRegression`; no torch/GPU import. Vectorised
  (flattened `[A*N,k]` design; one logistic fit per fold) — no per-observation Python loop in the hot
  path. Performance gate satisfied for a CPU OLS/logistic baseline.
- Config: `LIMIT_FRAC`/`NEAR_LIMIT_MULT`/`LOCK_THRESHOLDS` centralised in `limitlock_config.py` (the
  baseline's canonical config module, marked `# config-ok`); windows/floors sourced from
  `pipeline_config`. No scattered magic numbers on pipeline lines.

## Findings
No critical/major findings. Minor (accepted, not blocking): `import metrics as M` sits inside the
driver's case block (runs at most once, under `# pragma: no cover`).

## Verification
- `pytest baselines/2026-09-06_limitlock_hurdle/test/ -v` -> 18 passed, 0 skipped.
- Coverage C0 = 100% on all non-`pragma` lines of the three modules (term-missing: 0 missed).
- `ruff check --select F baselines/2026-09-06_limitlock_hurdle/` -> clean.
