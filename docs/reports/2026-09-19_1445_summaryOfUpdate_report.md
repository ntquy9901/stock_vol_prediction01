# Summary of Update — Review-5 (advisor) fixes → paper v6 + earnings-DM regression test

Date: 2026-09-19 14:45

## Scope
Fifth advisor review (8.5/10) of the SOICT submission plus a second AI-agent review of v5.
Applied the advisor's mandatory + should-fix items to a new paper version and hardened the
earnings-DM audit trail with a numerical regression test.

## What changed
- `docs/paper/soict_2026-09-19_1430_v6_review5.tex` (NEW, from v5) — review-5 fixes.
- `baselines/2026-09-19_expected_schedule/test/test_earnings_dm.py` — added a numerical
  regression test guarding `results/gamma_gbm/expected_schedule/earnings_dm_sp500.json`.

## Verified code facts (used to write the fixes, not the reviewer's guesses)
- COVID contradiction (advisor's one serious item): the walk-forward test period starts
  2022-07-01, but the spike-robustness exclusion text named "COVID (Feb-Apr 2020)". No test
  observation lies in 2020 (`SPIKE_WINDOWS` COVID window matches 0 test rows), so the 57,466
  excluded observations are only the within-test windows (2022 H2 drawdown + April 2025 tariff).
  Fix = describe only the in-test windows; drop COVID. This is a description bug, not a result bug.
- GNNHAR adjacency (reproducibility): `S1.build_graph` builds a top-10 nearest-neighbour graph
  from Pearson correlations of log Parkinson variance over the training window, keeps only
  positive correlations, excludes self-correlations (diagonal −inf), and row-normalizes each
  node's weights to sum to one; frozen for validation and test. The paper now states this exactly
  (the reviewer's guessed "absolute / self-loops added / symmetric D^-1/2 A D^-1/2" was wrong).

## Review-5 fixes applied (v6)
1. COVID/stress-window text corrected in both places; 57,466 stated as the excluded count;
   Limitations notes the stress windows are pre-specified and lie within the 2022-onward test.
2. Table placement: `[!t]` + fallback to top-of-next-page (Table 1 → p9, Table 2 → p10).
   `\FloatBarrier` interleaving would push the bibliography to page 14 (breaks the 12-page body
   limit), so the reviewer's explicit fallback placement is used.
3. "four external baselines" → "three external baselines (GARCH, HAR, GNNHAR), VolTree...".
4. "exact p-values ... below 0.001" → "emphasized component comparisons ... yield p<0.001;
   Table 2 uses a p<0.05 marker for compactness".
5. §3.7 "remove ... cross-sectional dependence" → "account for ... and obtain one loss
   differential per trading day".
6. GNNHAR adjacency accurate construction added (§3.4).
7. Causal wording softened (§5.1/§5.2: "drives" → "is associated with"; event study framed
   as ex-post realized dates motivating the cadence-estimated forecast-time features).
8. "matching VolTree" → "matching the rounded VolTree values at h1 and h5 because the HOSE
   earnings block is effectively inert".
9. Grammar: "data are obtained from Yahoo Finance"; "405 HOSE-listed Vietnamese stocks";
   "three main findings".
10. "conditionally safe" → "leakage-safe with respect to the evaluated forecast origins".
11. Under-three-releases behavior stated: "the four earnings features are set to zero".
12. Alpha wording tidied.
13. April 2025 tariff window dated (2025-04-01 to 2025-04-30), noted market-wide.
14. Fig 2 "announcement events"; Fig 3 caption notes per-decile median (no winsorization).

## Author overrides (kept against the reviewer's suggestion)
- Heading "VolTree Model" kept (reviewer suggested "Framework"; author prefers "Model").
- No "optional" wording introduced; architecture figure untouched (author: VolTree uses both
  components; the leaf-graph simply contributes nothing when its validation weight α is zero).

## Tests
- `pytest baselines/2026-09-19_expected_schedule/test/test_earnings_dm.py` — 2 passed
  (contract + numerical regression). The regression asserts: artifact exists; horizons exactly
  h1/h5/h10/h22; 8 folds each; QLIKE(XGB+E) < QLIKE(no-earn); DM diff < 0; recomputed gain ==
  stored field and == paper headline (±0.05); DM p in (0, 1e-3).

## Compile / format check (v6)
- 14 pages total; body 12; `\begin{thebibliography}` on page 13 (SOICT: body ≤12 excl. refs).
- 0 Overfull \hbox, 0 undefined references/citations, 21 bibitems all cited.
- Residual scan: 0 "COVID" / "four external baselines" / "optional" / "VolTree Framework".
- Numeric table cells unchanged (0.3457, 1.5640, 2.7 to 3.2 all present).

## Code review
- Advisor review-5 (8.5/10) + AI-agent review-2 of v5 both processed; the one serious item
  (COVID) verified as a description bug and fixed. AI-agent confirmed the earlier HIGH
  earnings-DM audit issue is closed in v5; its suggested numerical regression test is now added.

## Data-quality gate
- N/A (no data change) — paper text + one test file only; no data/feature/manifest touched.

## Follow-ups (low priority, not blocking)
- Independent visual PDF QA before EasyChair upload (manual; render tooling not in this env).
- Prose-only numbers not tabulated (alpha per-horizon values, event-study n) — procedures
  described; optional to tabulate.
