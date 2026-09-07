# Code review — regime-conditional blend (2026-09-07)

Adversarial 3-lens review (Blind Hunter + Edge Case Hunter + Acceptance Auditor) of
`code/regime_blend.py`, `code/run_regime_blend.py`, `test/test_regime_blend.py`, cross-checked against the
reused `submission/soict_lstm_gat/metrics.py` and `baselines/2026-08-21_har_anchored_residual/code/stats.py`.

## Core correctness — verified clean (no CRITICAL)
- **Causality / no leakage.** `features()` reads only the `harx`/`deep` forecast columns, never realized `y`
  (test `test_features_are_causal_columns_only`). Per-fold isolation: `blend_walkforward` groups by fold,
  fits θ + standardization on that fold's val only, applies to the same fold's test.
- **QLIKE floor identical** across deep/harx/blend and both DM loss series (`FLOOR = pc.QLIKE_FLOOR = 1e-8`).
- **DM orientation correct.** `date_clustered_dm` `mean_diff < 0` favours A; `_dm_qlike(blend, harx, …)`
  puts blend as A, consistent with the driver's `gain = (harx − blend)/harx`.
- **Numerics guarded.** sigmoid clipped ±30; logs floored; constant-feature σ→1; non-finite objective falls
  back to init θ; NaN handled by `dropna` after pivot.

## Findings and resolutions
| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| 1 | MAJOR | Result JSON lacks train/fit evidence → feared pre-push overfit-gate BLOCK | **Refuted by running the gate**: `check_overfit_evidence.py` on `regime_blend_result.json` + `regime_blend_spike.json` exits 0 — the post-hoc blend has no top-level `metrics`/`metrics_per_seed`/`masked` design, so it is correctly classified non-learned and skipped. No change needed. |
| 2 | MAJOR | Smoke test asserted the empirical outcome (`harx qlike > blend qlike`) → a data finding could turn the gate red | **Fixed**: removed the directional assertion; smoke keeps structural/finite/`0≤w≤1`/`p∈[0,1]`/`qlike>0` checks only. The beat-HAR-X claim lives in the report/driver, not the code gate. |
| 3 | MAJOR | `thetas.nunique() >= 1` is vacuous (always true) | **Fixed**: asserts `== 2` (two folds → two independent fits). |
| 4 | MINOR | Two-sided DM p used for a directional WIN* claim | **Documented**: `verdict` docstring now states the p is the two-sided HLN value (conservative-to-mildly-liberal gate). |
| 5 | MINOR | Pooled test metrics/DM assume disjoint fold test windows | **Verified disjoint** (no `(ticker,date)` duplicate within a split/model in the anchored walk-forward); noted in `load_folds`. |
| 6 | MINOR | `pivot_table` silently averages duplicate cells | **Documented**: explicit `aggfunc="mean"` + comment that cells are unique per (fold,ticker,date,model), so mean == identity. |
| 7 | MINOR | `val_metrics.blend_qlike` could be misread as held-out | **Fixed**: renamed to `blend_qlike_insample` with a comment; the claim uses test only. |

## Post-fix verification
- Tests: 12/12 pass; **100% line + 100% branch** coverage on `code/` (driver `main()` / `__main__` marked
  `# pragma: no cover`).
- ruff `--select F`: clean. Overfit-evidence gate: exit 0 on both result JSONs.
- No CRITICAL/unresolved MAJOR remain.
