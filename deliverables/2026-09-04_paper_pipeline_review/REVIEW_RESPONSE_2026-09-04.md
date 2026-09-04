# Response to the independent code review (No-go verdict) — 2026-09-04

Each finding was verified with evidence (not accepted or dismissed blindly). Verdicts: 2 valid
(fixed), 1 valid-doc (fixed), 1 partially-valid (test added), 1 refuted-with-evidence, plus the
environment note.

## Finding 1 — pooled h5/h10 not matching claims/summary status. **VALID → FIXED.**
`RESULTS_SUMMARY.md` said the ablation was "h1 only". At review time h5 and h10 had completed.
Updated the ablation section to show h1/h5/h10 (VolGA row) with the consistent pattern (QLIKE n.s.,
AE significant at all three) and marked h22 as running. Verified against
`results/pooled_transfer_vn30/pooled_vn30_h{1,5,10}.json`. (h22 has since completed; the full
4-horizon fill happens in the consolidation pass.)

## Finding 2 — "config VOLUME_ZSCORE_WINDOW is 22 but the paper results used 20". **REFUTED (evidence).**
The delivered VN30/VN100 results used **22**, consistent with the config:
- `git log -S 'VOLUME_ZSCORE_WINDOW'` → the 20→22 change landed at commit `fef4899`, **2026-08-30 23:11**.
- The VN100 result JSON was committed **2026-08-31 23:42** (`7154624`) and the VN30 result **2026-09-01
  06:48** (`c2abd13`) — both AFTER the config became 22.
- The reader `wf_enriched_panel._feature_cols()` selects the column `volume_zscore_{pc.VOLUME_ZSCORE_WINDOW}`
  at run time, i.e. `volume_zscore_22`. No result JSON references `volume_zscore_20`.
- The new paper draft states "22-day volume z-score".
The likely source of the "20" reading is that the enriched CSVs carry BOTH `volume_zscore_20` and
`volume_zscore_22` columns (the 20 is kept only for backward-repro), and the result JSONs do not record
which window was used. **Action:** record the volume window explicitly in future result JSONs to remove
the ambiguity; no number needs changing.

## Finding 3 — MANIFEST points to `submission/soict_lstm_gat/masked_rich.py`, which does not exist. **VALID → FIXED.**
Confirmed: that path does not exist; `masked_rich.py` lives ONLY at
`baselines/2026-08-21_har_anchored_residual/code/masked_rich.py`. The readers add both
`submission/soict_lstm_gat` and that dir to `sys.path`, so `import masked_rich` resolves to the
existing file (the code runs correctly; only the manifest path was wrong). MANIFEST corrected with a
note about the resolution.

## Finding 4 — isolation test does not verify the subset is actually VN30. **PARTIALLY VALID → TEST ADDED.**
`test_isolation.py` uses `score_idx=[0,1,2]` (a positional stand-in) — it proves the isolation
*mechanism* (Arm 0 predictions invariant to non-VN30 features) but not that the real scored subset is
the VN30 universe. Added `test_build_score_set_is_the_actual_vn30_universe` to `test_build_smoke.py`:
it calls the real `_build`, screens the actual VN30 universe, and asserts
`scored_tickers == frozen_VN30 ∩ VN100_panel` and `scored_tickers ⊆ frozen_VN30`. **3 passed**
(`pytest tests/test_build_smoke.py`).

## Finding 5 — could not run pytest/coverage (no Python runtime in the review environment). **ENVIRONMENT — evidence provided.**
Not a code defect; the reviewer's snapshot lacked a Python runtime. Test evidence executed here
(GPU venv, CPU-forced where needed):
- VolGA walk-forward: `pytest baselines/2026-08-31_walkforward_volga/code/tests` → 17 passed.
- Pooled ablation: `pytest baselines/2026-09-04_pooled_transfer_vn30/code/tests` → 11 passed
  (10 + the new VN30-universe test); diff-cover C0=100 / C1=100 on changed lines; ruff-F clean.
- PatchTST baseline: `pytest baselines/2026-09-04_patchtst_gat/code/tests` (from the `code/` dir so
  conftest pins MKL threads) → 14 passed; ruff-F clean.
Reproduce commands are in `REPRODUCE.md`.

## Net
- Fixed: Findings 1, 3 (docs), 4 (test added). Refuted with evidence: Finding 2. Environment:
  Finding 5 (evidence supplied).
- No source-code correctness defect was found; the No-go was driven by doc/manifest drift + the
  reviewer's inability to run tests. After these fixes and the consolidation pass (fill h22 + full
  ablation numbers), the package is ready for a re-review.
