# Code review — gamma-GBM + earnings + calm-calibration (2026-09-09)

Adversarial 3-lens review (subagent) of `code/gbm_earnings_walkforward.py` + `test/test_gbm_earnings.py`,
cross-checked against the committed gamma-GBM (`gamma_gbm_walkforward`), `wf_enriched_panel.pack_fold`,
`run_masked_rich`, and `stats.date_clustered_dm`.

## Verdict: 1 CRITICAL (fixed) + 1 CRITICAL test-gap (fixed) + 1 MINOR. Correctness now holds.

### CRITICAL (FIXED) — earnings feature indexed in the wrong index space
`earnings_panels` returns `[A, N]` arrays indexed by anchor POSITION (0..A-1), but `_design` indexed them with
`panel.anchors[fold.*]` = DATE-indices into the `[T,N]` axis (which start at ~30). This misaligned the earnings
columns row-for-row against `har5`/target for every fold, and IndexError'd on later folds (date-index ≥ A).
The base GBM path was unaffected (never touches earnings), but every `GBM+earn`/`GBM+earn+cal` number was
invalid. **Fix:** `_design(har5, extras, earn, anchors, pos, use_earn)` — HAR/extras use `anchors` (date-index,
via `G._design`), earnings use `pos` = `fold.train/val/forecast` (positional). Call sites updated.
After the fix, walk-forward GBM+earn = 0.4008 h5 / 0.4278 h10 (both DM p<0.001 vs GBM), a large valid gain.

### CRITICAL (FIXED) — test gap let the indexing bug pass CI
`test_design_column_counts` used `anchors=[0,1]` (position == date-index, so the confusion was invisible) and
all-zero earnings (a no-op passes). **Fix:** `test_design_earnings_aligned_by_position_not_dateindex` uses
`anchors=[30,31]` with `pos=[0,1]` and a distinct value per (position,node), asserting the earnings column at
row `i*N+j` equals `earn[pos[i], j]` (positional), which fails if indexed by the date-index anchors.

### Verified CORRECT (no finding)
- **Earnings forward-looking / no leakage:** distance from `panel.target_dates` (= date at anchor+horizon,
  i.e. the label's own date) to the scheduled earnings calendar (public, known ahead) — no realized-y leakage,
  no off-by-one vs `pk[anchor+horizon]`.
- **Calibration no-peek:** `_calibrate` fits `c*=mean(y/f)` (QLIKE-optimal multiplier) per forecast bin on VAL
  only; test uses val-derived edges; test realized never enters. Correct math.
- **Clean ramp:** `prox=max(0,1-dist/RAMP)` is 0 beyond RAMP days → no far-cell noise (this is what made h10
  significant, p=0.000, vs the raw-distance probe's p=0.10).
- **Floors / DM orientation:** all four models share the per-node nfloor + QLIKE floor; `_dm_all(model, base)`
  favors A iff model better; dm_vs_HARX and dm_vs_GBM both correctly oriented.

### MINOR (documented) — val/test floor basis for calibration
Val forecasts fed to `_calibrate` are floored at scalar `fl`; test at per-node `nf_te`. Slight basis mismatch
in calm bins; no-peek and low priority. Moot in practice: calibration is a NEGATIVE control (it worsens QLIKE:
h5 0.4125 > 0.4008, h10 0.4481 > 0.4278) and is not the shipped model.

## Post-fix verification
- Tests 8/8 pass (incl. the positional-alignment regression); **100% line + 100% branch** coverage. ruff -F clean.
- Walk-forward re-run after the fix: earnings is a large significant win at h5 AND h10; calibration is a
  documented negative. No unresolved CRITICAL/MAJOR.
