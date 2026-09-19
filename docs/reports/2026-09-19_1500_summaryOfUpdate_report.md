# Summary of Update — v6 content edits (author review pass) + event-study test

Date: 2026-09-19 15:00

## Scope
A batch of author-requested edits to the SOICT submission `soict_2026-09-19_1430_v6_review5.tex`,
plus one verified factual correction (HOSE earnings-cadence error) and a new unit test.

## Changes
1. **Removed §5.2 "Forecast Errors across Volatility Regimes" + Fig 3 (2x2 diagnostics).**
   The section's diagnostic (storm-decile concentration, HAR-lag dominance) was real, but its
   concluding claim that a cadence-estimated earnings feature is "the natural lever for this residual"
   was not demonstrated (earnings gain is a modest +2.7-3.2%, HOSE-inert). Author chose to remove it.
   Fig 3 was its only anchor; the orphaned `fig:diagnostics` ref in Limitations and the now-uncited
   `breiman2001` bibitem were removed (21 -> 20 refs, all cited).
2. **Ticker counts stated once.** The 497 S&P 500 / 405 HOSE universe sizes now appear only in the
   Data-sources paragraph; removed from abstract, introduction, Table 2 caption, §5.2, and conclusion.
   (The "~218 tickers" HOSE earnings news-feed coverage is a distinct quantity and is retained.)
3. **Fig 2 (event study): removed the baseline line.** The "baseline (ticker median)" was a constant
   at y=1 (values are normalized by each ticker's median), redundant with the ">1 = elevated" axis
   label and hidden under the HOSE median curve. Dropped the `axhline` + legend entry in
   `scripts/eda/earnings_event_study.py`; regenerated `fig_earnings_event_study.pdf/.png`
   (peaks unchanged: S&P 500 2.75, HOSE 1.08; n=42,780 / 5,256).
4. **Removed the GARCH estimable-subset caveat everywhere** (abstract, Baselines methods, Table 1
   caption, the "GARCH sample coverage" Limitations paragraph, and the Design-and-evaluation clause).
5. **Corrected the HOSE earnings-cadence error: 32 days -> 3 days (verified).** The 32-day figure came
   from the noisy multi-filing diagnostic (`earnings_pit_cadence` over `hose_earnings_combined.parquet`,
   which carries parent/consolidated/combined filings per quarter). The headline model uses the clean
   one-per-quarter `expected_schedule` (earliest disclosure per fiscal quarter), whose median prediction
   error is 3.0 days (mean 17.3; within-3d 55%, within-7d 75%), against 2.0 days on the S&P 500.
   Rewrote the Limitations sentence: the median schedule is nearly as clean as the S&P 500 (heavier tail
   only), so the earnings lever's non-transfer reflects HOSE's muted announcement-day volatility response
   (event study, Fig. 2), not a noisy schedule; the "no full-history null" caveat now rests on the
   limited disclosure coverage.
6. **"option-implied volatility/signals" -> plain wording** ("volatility measured from intraday prices
   or implied by option prices") in Limitations and Conclusion.

## Verification
- Cadence numbers recomputed with the headline module (`expected_schedule.summarize(expected_vs_actual(...))`):
  HOSE median_abs_days = 3.0 (n=757, 252 tickers); S&P 500 = 2.0 (n=41,821, 495 tickers).
- Compile: 14 pages total; 0 Overfull \hbox; 0 undefined refs/citations; 20 bibitems all cited.
  §5.2 removal pulled the last analysis subsection to page 11, so the body remains <= 12 pages.
- Residual scan: 0 "32 days", 0 "option-implied", 0 rendered GARCH-subset caveat (one provenance
  comment remains, not rendered); cadence "error is 3 days" present.

## Tests
- `tests/test_earnings_event_study.py` (NEW) — `event_curve` alignment + per-ticker-median
  normalization (announcement-day median == 5x synthetic spike; calm offset == 1.0 baseline) and the
  fail-loud guard when no events align. 2 passed.
- `baselines/2026-09-19_expected_schedule/test/test_earnings_dm.py` — still 2 passed (unchanged).

## Code review
- Author-driven edits; the one factual claim (cadence) was verified against the headline pipeline before
  changing the number and the surrounding narrative. §5.2 removal keeps the paper's conclusions intact.

## Data-quality gate
- N/A (no data change) — paper text, one figure-builder line, and one test; no data/feature/manifest touched.

## Additional author-requested edits (same pass)
7. **Walk-forward start rationale.** Added why training begins 2015-01-01 (not 2000): both markets
   need a liquid daily cross-section over the training span; HOSE's broad listing and earnings
   coverage are recent, so an earlier start would leave the Vietnamese panel sparse.
8. **Embargo / floor / cap clarified.** Stated that the embargo grows with horizon $h$ because a
   longer-horizon target reaches further past the fold boundary (dropping rows whose $h$-step target
   would fall inside the test window prevents look-ahead leakage); the $10^{-8}$ floor keeps QLIKE
   (which divides by the forecast) finite and the $1.0$ cap is applied identically so no model gains
   from different clipping.
9. **Removed HAC/Newey-West and HLN small-sample-correction wording** from the DM methodology (author
   request) and dropped the now-uncited `neweywest1987` and `hln1997` bibitems (20 -> 18 refs); the DM
   test is still described as date-clustered on the daily loss-differential series.
10. **Slides:** subtitle "Tóm tắt cho phản biện" -> "Tóm tắt bài báo khoa học".

## Gate fix (why the first push attempt was blocked)
The initial push (f9f9d4e6) was blocked because a new `tests/test_earnings_event_study.py` shared its
basename with the pre-existing tracked `scripts/eda/test_earnings_event_study.py`, triggering pytest's
prepend-import duplicate-basename collision (ModuleNotFoundError at collection). Fix: removed the new
duplicate (the existing scripts/eda test is more comprehensive) and made that existing test import
self-contained via an explicit `sys.path.insert` (so it no longer relies on pytest's prepend). Both
markets' event-study tests (5) pass; full suite passes with no collision. (The "ruff=fail" in that gate
run was informational E702 semicolons = house style, not a blocking F-code.)

## Compile (final)
- Paper: 14 pages, body <= 12, 0 Overfull, 0 undefined refs/citations, 18 bibitems all cited,
  0 residual HAC/Newey-West/HLN. Slides: 11 pages, compiles clean.

## Follow-ups (low priority)
- Independent visual PDF QA of Fig 2 (baseline removed) before EasyChair upload.
