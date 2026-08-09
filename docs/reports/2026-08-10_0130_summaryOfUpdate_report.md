# Track-B final submission package — consolidation, LNCS .tex, bundle re-sync

Date: 2026-08-10. Scope: consolidate the Track-B paper onto the ladder-consistent + multi-horizon +
classical-baseline canonical numbers, produce the conference LaTeX version, re-sync the
reproducibility bundle, and record the Model Confidence Set decision.

## What changed (files → purpose)

- `docs/paper/track_b_paper_draft.md` — consolidated Track-B paper (this pass owns this file).
  Removed all Track A content and every Track A↔B comparison (including the A1
  pooled-vs-common-date ablation). Collapsed G0: the ladder is now the single-basis nested
  `P0 → P1 → P2 → P3 → G1` where **P3 = G1 with the message-passing residual disabled** (graph-off
  readout determinism 0.0). Swapped the stale §6.1 numbers for the final h5 ladder, added a
  classical-baseline table and a multi-horizon (h1/5/10/22) table with per-horizon G1-vs-P3
  Diebold-Mariano verdicts. Re-grounded §5 training description (MSE-train / QLIKE-eval, RTX 4060 /
  PyTorch 2.6 / CUDA 12.4, single consistent basis, no frozen-backbone two-phase). Framing changed to
  parsimony: G1 = proposed full architecture; the cross-stock graph adds no statistically significant
  improvement at any horizon; the parsimonious news backbone (P2) attains the lowest test QLIKE;
  classical HAR/HARQ tie the deep models and GARCH is far worse.
- `docs/paper/soict2026_trackb_v1.tex` — new Springer LNCS conference version (`\documentclass[runningheads]{llncs}`),
  same package set and `\newcommand` canonical-number-macro pattern and author/institute block as
  `soict2026_draft_v3.tex`. Track-B only, clean P0→P1→P2→P3→G1 ladder, classical baselines,
  multi-horizon null, parsimony. Booktabs tables.
- `submission_track_b/reproduce.py` — `view` now reads the consistent-ladder JSON
  (`results/ladder_consistent_h5.json`), the classical baselines, and the four per-horizon files;
  ladder is P0→P1→P2→P3→G1 (no G0); prints val ladder, G1 test row, classical-baseline test table,
  and the multi-horizon graph verdict.
- `submission_track_b/PAPER_MAP.md`, `submission_track_b/README.md` — rewritten to the collapsed
  ladder, final numbers, and the MCS-not-computed note.
- `submission_track_b/results/` — added `ladder_consistent_h{1,5,10,22}.json`,
  `classical_baselines_h5.json`, `ladder_multihorizon.md`; removed stale
  `g0g1_graph_validation_comparison.json` and `pooled_*` JSONs (old G0/pooled numbers).
- `submission_track_b/test/test_reproduce.py` — updated config set to `{P0,P1,P2,P3,G1}`; added
  tests for no-G0, classical-baseline load, and full-precision G1-test match to the ladder JSON.
- Deleted stray `docs/paper/track_b_paper_draft - Copy.md`.

## Canonical numbers (single source of truth)

All values trace to `docs/reports/ladder_consistent_h5_2026-08-09_154402.json`,
`docs/reports/ladder_consistent_h{1,10,22}_2026-08-09_180326.json`, and
`docs/reports/classical_baselines_h5_2026-08-09_182129.json`.

Five-day held-out test, 3-seed mean (QLIKE): P0 0.567625, P1 0.564780, **P2 0.559854** (lowest in
study), P3 0.576488, G1 0.575926. Classical test QLIKE: HAR 0.579291, HARQ 0.573674, EWMA 0.600625,
log-HAR 0.779422, GARCH 1.76100, GJR-GARCH 1.82432, EGARCH 1.87379. Graph ablation G1 vs P3 (test):
paired-t p=0.7913 (n.s.), verdict B. Multi-horizon graph verdict B at h1/h5/h10/h22.

## Model Confidence Set (Step 1)

**Not computed — Diebold-Mariano used instead.** `arch.bootstrap.MCS` is installed, but the result
artifacts store per-configuration aggregate metrics, not per-observation prediction series, and clean
alignment onto one common observation set (GARCH covers a 32-of-33-ticker subset) was not feasible
within the time box without re-scoring the full multi-seed ladder. Recorded in the paper (§5,
Limitations), PAPER_MAP, and README; DM tests are the primary graph-significance evidence.

## Consistency check

Grep-verified the headline G1/P3/P2/HAR/HARQ numbers agree across `track_b_paper_draft.md`, the .tex
macros/tables, `PAPER_MAP.md`, and `reproduce.py view` output (differences are display precision only,
e.g. 0.575926 vs 0.5759, not numeric conflicts). No stale Track-A numbers (0.4430 / 0.5493 / 0.4603 /
0.8031) appear in any new file or the bundle. Track A / A1 / common-date fully removed from the paper.

## Verification / gates

- Bundle tests: `python -m pytest submission_track_b/test/test_reproduce.py -q` → **11 passed**.
- `python reproduce.py view` → runs clean, regenerated `output/results_table.md` + `output/summary.png`.
- Lint: `ruff check submission_track_b/reproduce.py submission_track_b/test/test_reproduce.py` → clean.
- LaTeX: no pdflatex/latexmk on this host; the .tex was **structurally linted** instead — balanced
  environments, all `\ref`/`\cite`/`\label` and custom `\newcommand` macros defined, braces balanced.
  Compilation was not run.
- Diff-coverage: `diff-cover --fail-under=100` — Not run (tool not installed; standing repo tooling gap).
- Data-quality (Pandera/Evidently): N/A — no change to data, features, manifest, or train pipeline.

## Follow-ups

- Compile `soict2026_trackb_v1.tex` with the official SOICT 2026 `llncs.cls` / `splncs04.bst` on a
  host with a LaTeX toolchain before submission; verify bibliography entries.
- Update the §7 "Attempts to strengthen the graph" placeholder paragraph (paper .md and .tex) with the
  beat-HAR sweep outcome once it returns.
- Optionally compute the MCS if per-observation prediction series are dumped from a future ladder run.
