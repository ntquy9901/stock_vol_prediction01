# Track-B Final Submission Assembly — Summary

Date: 2026-08-10. Scope: final assembly of the Track-B SOICT-2026 submission (paper §7 fill,
sweep report/ledger onto master, repro-bundle coverage gate, dashboard, consistency, push).

## What changed

| Path | Purpose |
|---|---|
| `docs/paper/track_b_paper_draft.md` | §7 placeholder replaced with the real beat-HAR sweep outcome; Abstract + Discussion thoroughness note; provenance footer cites the two new reports |
| `docs/paper/soict2026_trackb_v1.tex` | Same §7 fill + Abstract note (LNCS); source-trace comment for sweep numbers |
| `docs/reports/2026-08-10_0412_beat_har_sweep_results.md` | Copied to master (sweep C1–C6 results) |
| `docs/reports/2026-08-10_0130_gat_price_har_quick.md` | Copied to master (price-only GAT check) |
| `scripts/task_dashboard/task_ledger.json` | Appended sweep entry + final-assembly entry; fixed repro-bundle `diff_cover`; sweep `provenance_note` (docs only) |
| `submission_track_b/test/test_reproduce.py` | +8 offline run-entry-branch tests (main/menu/cmd_infer-guard) to raise reproduce.py coverage |
| `docs/reports/task_dashboard.html` | Regenerated |

## §7 text (no win claimed)

Tested a leaner price-only GAT + five research-backed levers vs the P0 pooled-HAR anchor (test
QLIKE 0.5676): C1 QLIKE-loss GAT 0.5730 (paired-t p=0.027; beats classical HAR 0.5793, ties P0
RMSE, worse on QLIKE); C2 HAR+graph-residual 0.5662 — statistically **ties** P0 (paired-t p=0.562)
and RMSE, the best but not a win; C3 spillover 0.5908; C5 spillover+omit-self k16 0.5748; C6
learned adjacency 0.5903 — all significantly worse. Price-only GAT beats HAR on 1/6 test metrics
(DM n.s. p=0.19–0.25). C4/C7 not run (reasons stated). Conclusion: neither the base graph nor any
of the five enhancements beats a well-specified HAR — strengthens the parsimony finding. Numbers
trace to `docs/reports/2026-08-10_0412_beat_har_sweep_results.md` and
`results/beat_har_sweep_2026-08-10_0130/analysis.json`.

## Quality gate

- **Repro-bundle diff-cover (measured, not asserted):** `reproduce.py` 83.5% on changed lines
  (pytest-cov --cov-branch, diff-cover vs 458b2ba; +8 offline tests; was 64.6%). Full-bundle 28%
  is vendored `trackb_code/` + data/GPU run-entry (`g1_final.py`, `cmd_train`) exercised end-to-end
  by the documented subset run. Card now green.
- **gat-quick card:** greened via a REAL clean push of `feature/gat-har-quick` (200 passed,
  diff-cover 87%, ruff pass, "quality gate passed"); origin HEAD `a5a792d`; fresh green
  `a5a792d.json` overwrote the 02:21 starvation fail.
- **Tests/lint:** 19 bundle tests pass; ruff clean on the changed test file.
- **Consistency:** G1/P3/P2/HAR/HARQ + sweep C1–C6 numbers match across md ↔ tex ↔ PAPER_MAP ↔
  `reproduce.py view`. No drift.

## Dashboard — 1 documented red

`beat-har-solution-sweep-2026-08-10` renders red only because the entry's `commits` list predates
the branch's final two commits; the sweep's real verified-green gate result is `52b9e75.json`
(tests_passed:true, diff-cover 90%, overall:pass). Completing the `commits` list with `52b9e75` is
an accurate provenance fix awaiting the user's explicit authorization. The gate status and commits
are left unchanged (no tampering). The two prior over-red cards (repro-bundle, gat-quick) are green
from real passing runs. No audit records were deleted or hand-passed; no QG_SKIP.

## Not done pending user authorization

The 30-second user "A" authorization to complete the sweep entry's `commits` with `52b9e75` →
0 red. Until then, 1 documented red is the honest state.
