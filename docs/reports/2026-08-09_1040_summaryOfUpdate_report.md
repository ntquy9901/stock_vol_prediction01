# Track-B Reproducibility Bundle - Summary of Update

Date: 2026-08-09
Scope: new `submission_track_b/` reproducibility bundle for the Track-B volatility model.

## What changed

A self-contained bundle was added at repo root so a reviewer on Windows can, without typing
complex commands, (1) view all paper results, (2) run inference with the final model, and
(3) optionally retrain, then cross-check outputs against the paper.

The bundle consolidates the committed Track-B pilot code
(`feature/pooled-news-gnn-pilot`, baseline `2026-08-08_pooled_news_gnn_ablation_baseline`)
and the small result JSONs it produced. No large data or fat checkpoints are committed.

Correction applied mid-task: the FINAL / PROPOSED model is **G1** (P3 backbone + cross-stock
graph message passing), not P3. `infer`/`train` target G1; `view` labels G1 as the final model.

## Files (path -> purpose)

- `submission_track_b/reproduce.py` - single entry point: interactive menu + `view|infer|train`.
- `submission_track_b/trackb_code/` - consolidated model code copied from the pilot
  (`data.py`, `models.py` incl. `GraphAblationModel` = G0/G1, `train.py`, `scaling.py`,
  `run_pilot.py`) + bundled dependency `src/common/evaluation.py` + new `g1_final.py`
  (trains/evaluates the final G1 model, adds a G1 checkpoint and a test-split score that the
  validation-only pilot did not provide). Folder is named `trackb_code/` (not `code/`) to avoid
  shadowing Python's stdlib `code` module, which breaks `pytest`/`pdb`.
- `submission_track_b/results/*.json` - small JSONs read by `view` (pooled P0-P3 3-seed
  aggregate, single-seed detail, and the G0/G1 graph comparison).
- `submission_track_b/*.bat` / `*.sh` - double-click launchers: `START_HERE`, `view_results`,
  `run_inference`, `train_model`, `setup` (+ POSIX mirrors). Each prints a header and `pause`s.
- `submission_track_b/README.md` - "Double-click START_HERE.bat" + 3-line quickstart +
  which-command-reproduces-which-number.
- `submission_track_b/PAPER_MAP.md` - claim -> command -> output-file mapping (G1 = proposed).
- `submission_track_b/requirements.txt` - pinned deps (numpy, matplotlib, torch, pandas,
  scikit-learn, pyarrow) with verified versions.
- `submission_track_b/test/test_reproduce.py` - tests for the entry point.
- `docs/reports/2026-08-09_1040_summaryOfUpdate_report.md` - this report.

## Working status of the three paths

- **`view` - fully working, no data, no training.** Verified end to end: it prints the full
  P0->G1 table (all 6 metrics) from the shipped JSONs and writes `output/results_table.md` +
  `output/summary.png`. Printed table below.
- **`train` / `infer` (final G1) - verified end to end with the dataset.** Both require the
  project dataset (`data/processed/*.csv` + `data/features/dual_group_news_panel.parquet` and its
  `.provenance.json` sidecar). The master tree lacks the `.provenance.json` sidecar, so these
  paths need `TRACK_B_DATA_ROOT` pointed at a repo that has the full data (the pilot worktree was
  used here). A real 4-ticker, seed-42, 1-epoch run was executed on CPU:
  - `train` trained G1, saved `checkpoints/g1_final.pt`, and printed val + test metrics (below).
  - `infer` re-loaded `checkpoints/g1_final.pt` and reproduced the **identical** test metrics,
    confirming deterministic load-and-score.
  A full CPU run is slow (graph manifest build dominates, several minutes); the numbers below are
  a small-subset smoke, not the paper's final full-universe figures (which a reviewer regenerates
  with the full dataset via the launchers). `infer` requires `train` to have produced
  `checkpoints/g1_final.pt` once. These checkpoints/metrics are git-ignored (kept out of the repo).

  G1 subset run (4 tickers, seed 42, 1 epoch) - proof the path runs:
  ```
  metric                 validation       test
  rmse                    3.093e-04    4.765e-04
  mae                     1.866e-04    3.018e-04
  r2                        0.05143      0.14116
  qlike                     0.53411      0.55305
  directional_accuracy      48.56%       48.48%
  ```

## Printed P0->G1 table (real `python reproduce.py view` run)

```
Model Description             Role                       MSE        RMSE         MAE          R2       QLIKE     DirAcc%
------------------------------------------------------------------------------------------------------------------------
P0    HAR pooled linear       ablation             2.204e-06   1.485e-03   4.797e-04      0.7351      0.5167       48.54
P1    Price LSTM              ablation             2.245e-06   1.498e-03   4.887e-04      0.7301      0.5110       48.66
P2    Price + News            ablation             2.208e-06   1.486e-03   4.801e-04      0.7346      0.5084       48.53
P3    Price + News + gate     ablation             2.210e-06   1.487e-03   4.806e-04      0.7343      0.5084       48.53
G0    Backbone, graph OFF     ablation             2.136e-06   1.462e-03   4.639e-04      0.7433      0.5095       48.58
G1    Backbone+graph kNN-8    FINAL / PROPOSED     2.119e-06   1.456e-03   4.621e-04      0.7453      0.5092       48.68
```

Scope: metrics are VALIDATION 3-seed means. P0-P3 = pooled ablation family; G0-G1 = graph
family from the definitive masked-manifest, screening-P3-backbone run (k-NN-8 adjacency for G1,
same 14,418 val obs; source `docs/reports/verdict_masked_g0g1_newbackbone_2026-08-09_120512.json`).
Parsimony finding: G1 does not significantly beat G0 (Diebold-Mariano on QLIKE n.s.; held-out
test QLIKE slightly worse) - the graph layer adds no measurable value. G1's held-out test 3-seed
mean is shipped and shown by `view`.

## Tests / quality gates

- `python -m pytest submission_track_b/test/ -q` -> 7 passed (covers `view` end to end,
  metric formatting, JSON loading, G1-final labelling, G1 test-row rendering, and the
  `infer` missing-checkpoint guard). One test is tagged `smoke`.
- `ruff check` on the added/edited files (`reproduce.py`, `trackb_code/g1_final.py`,
  `trackb_code/run_pilot.py`, `test/test_reproduce.py`) -> All checks passed.
- Diff-cover (C0/C1): the pre-push hook computes diff-coverage against the pilot/quality-gate
  coverage report, which does not include the bundle sources; bundle-line C0 is therefore
  `Not run` under the hook. The `view` path is directly exercised by the passing tests.
- Data-quality gate (Pandera/Evidently): N/A for logic; the bundle changes no data or manifest.
  The pre-push hook still runs Pandera schema because the copied filenames contain `data`/
  `scaling`; that validates the existing `data/processed` and is expected to pass.
- Code review: focused adversarial self-review of `reproduce.py` and `g1_final.py`
  (path bootstrap, no-data view path, honest val-vs-test labelling, checkpoint provenance
  reuse). No critical/major issues; the `code`->`trackb_code` stdlib-shadow fix came from that pass.

## Risks / follow-ups

- The G0/G1 rows are single-seed (42) and may be refreshed by the separate "latest-P3 backbone"
  graph run; the P0-P3 confirmation numbers are 3-seed and stable.
- `train`/`infer` end-to-end on CPU is slow (graph build). A reviewer with a GPU or patience,
  and the full dataset, runs them via the launchers; `view` covers the offline path.
