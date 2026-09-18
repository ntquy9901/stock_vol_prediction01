# Code review — expected-schedule (leakage-safe) earnings baseline

Date: 2026-09-19. Scope: `code/expected_schedule.py`, `code/run_expected.py`,
`test/test_expected_schedule.py`, and the shared committed helper `scripts/eda/earnings_pit_cadence.py`.
Method: adversarial 3-layer (Blind Hunter + Edge-Case Hunter + Acceptance Auditor), plus one prior external
review. `archive/` and `deliverables/` mirror copies out of scope per project rules.

## Purpose
Replace realized (actual) earnings dates with a strictly leakage-safe **expected schedule** predicted from a
firm's OWN past reporting cadence, so the earnings feature carries no future-date foreknowledge. HOSE first
reduces the multi-filing disclosure stream to one quarterly event per (ticker, fiscal-year, quarter),
earliest across accounting scopes.

## Findings and resolutions

### CRITICAL (external review) — anchor passthrough leaked future actual dates — FIXED
`predict_schedule` returns the first `MIN_HISTORY=3` events as the RAW actual dates (anchors, no prediction).
`full_matrix.panel` consumes the schedule's *next* date after each forecast target as a forward-looking
feature (`earn_prox/earn_soon/earn_pre` via `_signed`). So for any forecast origin before a ticker's 3rd
actual release, the "next earnings" pointed at a realized future date — a point-in-time leak, even though the
prediction formula was causal.
- **Fix:** `expected_schedule` now exposes ONLY the causally-predicted dates
  (`predict_schedule(dates)[MIN_HISTORY:]`); the actual anchors are never emitted. Tickers with
  `<= MIN_HISTORY` observed releases expose an empty schedule (features stay neutral).
- **Regression guard:** new panel-level integration test
  `test_panel_no_future_actual_date_leaks_before_history` builds `full_matrix.panel` with an origin before
  history, asserts forward features are 0, and adds a contrast case feeding the raw actual dates that must
  light up. Confirmed empirically to FAIL on the pre-fix construction.

### C1 (prior) — off-by-one look-ahead in the cadence median — FIXED
`predict_schedule` used `median(gaps[:i])`, which includes `gaps[i-1] = d[i]-d[i-1]` (the gap TO the future
event). Corrected to `gaps[:i-1]` (strictly prior gaps) in both `expected_schedule.py` and
`scripts/eda/earnings_pit_cadence.py`. Docstrings in both were still stating `gaps[:i]`; corrected to
`gaps[:i-1]` (MEDIUM #2 from the external review).

### M1 (prior) — causality test could not distinguish causal from leaky — FIXED
The test now uses gaps `[30,90,90]`: causal `median([30,90])=60` vs leaky `median([30,90,90])=90`, so it
genuinely separates the two. Verified.

### M2 (prior) — `expected_load_earn` untested — FIXED
`run_expected.run_leaf_graph_paper` import moved lazily into `main()`; added
`test_expected_load_earn_dispatch`.

### MEDIUM (new) — `pit_cadence` still exposes anchors + misleading docstring — FIXED (docstring)
`scripts/eda/earnings_pit_cadence.py::pit_cadence` returns the full anchor-inclusive schedule and its
docstring advertised it "for the panel builder", inviting a re-introduction of the CRITICAL leak. Verified it
is NOT wired into any model run (grep: only its own unit tests reference it; `main()` uses `pit_vs_actual`
for the discrepancy CSV only). Docstring corrected (module + function) to state it is discrepancy-analysis
ONLY and NOT leakage-safe for panel features, pointing to `expected_schedule` as the model path. Behavior
left unchanged (dead for modeling), so committed discrepancy outputs are unaffected.

### MEDIUM (new) — silent degradation if earnings keys don't match frame tickers — RESOLVED (evidence)
`full_matrix.panel` uses `edates.get(tk)` and would silently produce all-zero earnings features (read as a
null result) on a ticker-key mismatch. Measured coverage of the exposed schedule against the enriched-CSV
frame stems:
- **SP500: 495/497 frames (100%)** matched a non-empty exposed schedule.
- **HOSE: 247/405 frames (61%)** matched a non-empty exposed schedule.
Coverage is healthy, so the near-inert HOSE earnings gain reflects genuine signal weakness (thin 2025+
archive → few causally-predicted dates per ticker, late in the test window), NOT a join failure. The `panel`
join is pre-existing shared code (not modified here); the coverage numbers are recorded so a zero-overlap run
would be distinguishable from a genuine null.

### LOW (new) — precomputed schedule, not per-origin — DOCUMENTED
The schedule is built once over full history. This stays leak-safe only while `h` ≪ inter-release gap
(quarterly ~91d vs max h=22 trading days). Added a docstring note to `expected_schedule`; no change needed
for the current horizon grid.

### LOW (new) — `MIN_HISTORY` duplicated across two modules — FOLLOW-UP
`MIN_HISTORY=3` is defined independently in `expected_schedule.py` and `earnings_pit_cadence.py` (different
dirs: baseline vs scripts/eda). Both carry explanatory comments. Left as follow-up to avoid inverting the
dependency (scripts/eda → baseline); noted here for single-source awareness.

## Blast radius (does the anchor leak move the headline numbers?)
Data date-range vs the 2015→ walk-forward folds (test starts 2022-07):
- **SP500** earnings back to 1999 → only 10/495 tickers have their 3rd anchor in-test; dropping anchors
  barely changes SP500 earnings numbers.
- **HOSE** archive only 2025-01→2026-08 → all 252 multi-event tickers have anchors in-test; dropping anchors
  makes HOSE earnings essentially neutral (consistent with the prior "HOSE earnings inert" finding). The HOSE
  headline lever is the leaf-graph smoothing (earnings-independent), so the paper story is preserved.

## Verdict
Core leak fix SOUND (verified end-to-end through `run_expected.main` → `_load_earn` → `FM.panel`). All
CRITICAL/MEDIUM handled; two LOW are documented/follow-up. Tests: 9 (expected_schedule) + 9
(earnings_pit_cadence) = 18 pass. ruff `--select F` clean; postgen config-hardcode gate pass.
Leak-free result JSONs regenerated with the fixed schedule (single clean run, after killing overlapping
orphan runs that had corrupted the prior output).
