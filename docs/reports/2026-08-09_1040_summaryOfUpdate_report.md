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
- **`train` / `infer` (final G1) - implemented; require the project dataset** (`data/processed/*.csv`
  + `data/features/dual_group_news_panel.parquet` and its `.provenance.json` sidecar). The
  master tree lacks the `.provenance.json` sidecar, so these paths need `TRACK_B_DATA_ROOT`
  pointed at a repo that has the full data (e.g. the pilot worktree). They are import-verified
  and pass data-loading; a full CPU end-to-end run is slow (graph manifest build dominates) and
  is documented as reviewer-runs-with-data. `infer` requires `train` to have produced
  `checkpoints/g1_final.pt` once.

## Printed P0->G1 table (real `python reproduce.py view` run)

```
Model Description             Role                       MSE        RMSE         MAE          R2       QLIKE     DirAcc%
------------------------------------------------------------------------------------------------------------------------
P0    HAR pooled linear       ablation             2.204e-06   1.485e-03   4.797e-04      0.7351      0.5167       48.54
P1    Price LSTM              ablation             2.245e-06   1.498e-03   4.887e-04      0.7301      0.5110       48.66
P2    Price + News            ablation             2.208e-06   1.486e-03   4.801e-04      0.7346      0.5084       48.53
P3    Price + News + gate     ablation             2.210e-06   1.487e-03   4.806e-04      0.7343      0.5084       48.53
G0    Backbone, graph OFF     ablation             5.793e-06   2.407e-03   6.701e-04      0.7447      0.6876       48.61
G1    Backbone + graph ON     FINAL / PROPOSED     5.809e-06   2.410e-03   6.602e-04      0.7440      0.6963       48.52
```

Scope: metrics are VALIDATION (the pilot is validation-only screening). P0-P3 = pooled
ablation family; G0-G1 = graph common-date family (separate sample sets). G1 held-out test
metrics are produced by `train`/`infer`.

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
