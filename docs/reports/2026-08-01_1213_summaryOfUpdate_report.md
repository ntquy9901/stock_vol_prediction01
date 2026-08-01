# Summary: Bug Fix (per-ticker windowing/scaling) + Multi-Horizon (1/5/10-day), S&P 500 Experiments

**Date:** 2026-08-01
**Branch:** `global-benchmark`
**Scope:** `research/phase5_multihorizon/`

---

## What changed

Triggered by a user question — why is S&P 500 DirAcc (50.89%, Phase 3) far below VN30's (68.42%) on
the same HAR-only 5-day setup? Code review found 3 structural bugs in `train_enhanced.py` and
`cross_market_experiment.py` (both pool multiple tickers' rows before windowing/scaling — VN30's
`ParallelLSTMGNN` avoids this by keeping stocks on a separate array dimension). User decided to fix the
bugs first, then add the requested `+1`/`+10` day horizons (existing `+5` retrained too, since a fixed
pipeline can't be compared against the old buggy horizon-5 baseline).

### Files
| File | Purpose |
|---|---|
| `src/common/multi_ticker_dataset.py` (new) | Per-ticker split/scale/window builder — the bug fix |
| `src/common/feature_merger.py` | Added `horizon` param; `target_5d` → `target_{h}d` |
| `src/experiments/sp500/train_enhanced.py` | Rewired to use the fixed builder; added `--forecast_horizon` |
| `src/experiments/sp500/cross_market_experiment.py` | Same rewire; added `--forecast_horizon` |
| `tests/test_sp500/test_multi_ticker_dataset.py` (new) | Boundary/scaler-reuse/empty-split correctness |
| `tests/test_sp500/test_multihorizon_target.py` (new) | Target-shift correctness (h=1, h=10) |
| `research/phase5_multihorizon/{REQUIREMENTS,DESIGN,RESULTS}.md` | Spec, design, results |

### The 3 bugs fixed (see `DESIGN.md` §0 for detail)
1. Sliding windows built over row-wise-concatenated multi-ticker frames could span 2 different tickers.
2. `StandardScaler` fit on the pooled multi-ticker split, blending different stocks' volatility scales.
3. Val/test splits fit their OWN independent scaler instead of reusing train's — train/inference
   distribution mismatch.

Fix: `build_per_ticker_datasets()` / `build_full_series_datasets()` — one dataset per ticker (windows
never cross ticker boundaries), scaler fit once per ticker on train only, `.transform()` (not refit) for
val/test. Reuses the project's existing `temporal_split_dataframe()` utility per-ticker rather than
reinventing split logic.

---

## Tests + coverage

Test-first: wrote `test_multi_ticker_dataset.py` + `test_multihorizon_target.py` against the not-yet-
existing module/params, confirmed 7/9 relevant tests failed (red state) before implementation, then
implemented and confirmed all pass.

```
python -m pytest tests/test_sp500/ -v
29 passed in 2.7s
```

Includes a regression test (`TestBoundarySafety::test_old_concat_then_window_logic_would_mix_tickers`)
that demonstrates the old bug directly, and an edge-case test (`TestEmptySplit`) added after a real
0-row-split crash was hit during the actual training runs (one VN30 ticker's `test_ratio=0.0` split
rounded to 0 rows; `StandardScaler.transform()` rejects 0-sample arrays — fixed in `_scale_and_window`).

**Coverage gate:** `diff-cover` not set up in this repo (documented gap in CLAUDE.md) — **Not run**.
C0/C1 not measured numerically; all new/changed lines are exercised by the 9 new tests (traced
manually: `WindowedSeriesDataset`, `_scale_and_window` fit/no-fit/empty branches, `build_per_ticker_datasets`,
`build_full_series_datasets`, `feature_merger.merge_features` horizon param, `load_market_data` horizon
param — all covered by at least 1 test).

**Smoke tests:** `train_enhanced.py --epochs 2` and `cross_market_experiment.py --epochs 2` both ran
end-to-end without crashing before committing to the full 10-epoch sweep.

---

## Training runs (10 epochs each, all under the fixed pipeline)

12 runs: Experiment A (feature-set comparison, AAPL/MSFT/GOOGL) × horizon {1,5,10} × {har,full} = 6;
Experiment B (cross-market SP500↔VN30) × horizon {1,5,10} × {sp500→vn30, vn30→sp500} = 6. Full tables
in `research/phase5_multihorizon/RESULTS.md`.

**Bug-fix verdict (§3 of RESULTS.md):** comparing old (buggy) vs new (fixed) at horizon 5 — DirAcc
barely moves (±1.4pp, within single-seed noise), but **QLIKE improves 4-4.6× on 3 of 4 comparable
cells**. The bug fix mattered for calibration (QLIKE), not for the near-random directional accuracy —
**the original "why is S&P 500 DirAcc so much lower than VN30" gap is NOT primarily explained by these
3 bugs.** It's more likely the maturity gap already discussed (3 tickers vs 30, no GAT/spatial branch,
no per-ticker gate, 10 epochs, 1 day of iteration vs VN30's ~1 month) — a finding worth stating plainly
rather than implying the bug fix "solved" the accuracy gap, which it did not.

**New open finding:** horizon-1 DirAcc is anomalously LOW (31-36%, below random) across all 4
sub-experiments, while QLIKE/R² are BEST at horizon 1 — the opposite of VN30's pattern (horizon 1 =
easiest on every metric). Reproducible across both Experiment A and B, so unlikely to be single-run
noise. Not investigated further — flagged in RESULTS.md §4 as follow-up.

---

## Code review

Self-performed adversarial review (the `/code-review` skill requires direct user invocation via
`/code-review`, not available via the Skill tool in this session — noted so the user can re-run it
independently if a second pass is wanted). 2 findings, both PLAUSIBLE severity:

1. **Fixed during review:** `_scale_and_window` silently dropped a 0-row split with no warning — added
   a `[WARN]` print so an excluded ticker is visible in logs, not just discoverable by reading source.
2. **Documented, not fixed (out of scope):** `build_full_series_datasets` (cross-market test-side) fits
   its scaler on the test market's ENTIRE date range, so an early window's normalization can use later
   dates' statistics — a mild look-ahead in scaling only (not in predictions). Inherited from the
   ORIGINAL `cross_market_experiment.py` (not a regression from this fix); a proper walk-forward
   test-market scaler is a larger redesign, out of this phase's scope. Documented in `DESIGN.md` §0.

Also removed 1 orphaned variable (`criterion_for_eval`, unused, introduced then unused during the
rewrite) and 1 pre-existing dead-code pattern (`patience_counter` in `train_enhanced.py` that was
tracked but never checked against a threshold — noted, not silently smuggled in: this was already
non-functional in the original code, dropped as part of rewriting the same training-loop block).

---

## ⚠️ Important — unrelated to this change, found incidentally

**`src/experiments/sp500/` (containing `train_enhanced.py`, `cross_market_experiment.py`,
`download_sp500.py`) is gitignored and has NEVER been committed**, despite Phase 1/3/4 commit messages
claiming these files were added. Root cause: `.gitignore` line 79 has an unanchored `experiments/`
rule (originally meant for some other scratch directory) that also matches `src/experiments/sp500/` —
`git check-ignore -v` confirms `.gitignore:79:experiments/` as the matching rule, and
`git log --all -- src/experiments/sp500/train_enhanced.py` returns empty history.

**Consequence:** all of this session's edits to these 2 files exist ONLY on disk right now. A plain
`git add -A && git commit` would silently skip them — the multi-horizon + bug fix work in
`train_enhanced.py`/`cross_market_experiment.py` would NOT be captured in git history. I did not modify
`.gitignore` or force-add these files myself since that's a policy change outside this task's scope —
flagging for the user to decide (e.g. anchor the rule to `/experiments/`, or add
`!src/experiments/`).

---

## Definition of Done checklist

- [x] Code satisfies the request (bug fix + horizon 1/10, horizon 5 retrained under fixed pipeline)
- [x] Tests: 9 new tests, test-first (confirmed red before fix), all pass; no regressions (29/29 total)
- [x] Checks run: pytest run and shown above; no lint tool configured in this repo (documented gap)
- [x] Code review: self-performed adversarial review, 2 findings (1 fixed, 1 documented as known limit)
- [x] Summary report: this file
- [x] Smoke test: 2-epoch runs of both scripts passed before the full sweep
- [x] Impact analysis: grepped for other consumers of `target_5d`, `VolatilityDataset`,
      `processed_sp500_enhanced` — confirmed no external callers outside the 2 files rewired
- [x] Similar-pattern check: N/A — no other file in the repo pools multiple tickers this way
- [ ] Coverage gate (`diff-cover --fail-under=100`): **Not run** — tool not installed in this repo
      (documented gap in CLAUDE.md "Tooling gaps")
