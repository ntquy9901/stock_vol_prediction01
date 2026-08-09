# Track-B Volatility Model - Reproducibility Bundle

## Double-click `START_HERE.bat`

That opens a menu:

```
[1] View all results   (no training, no data)   <- start here
[2] Run inference       (final model G1 on test set)
[3] Train               (final model G1)
[4] Everything
[0] Quit
```

## 3-line quickstart (Windows, no typing)

1. (First time only) double-click **`setup.bat`** - creates a local Python environment and installs dependencies.
2. Double-click **`view_results.bat`** - prints the full results table and writes `output/results_table.md` + `output/summary.png`. Needs no data and no training.
3. (Optional) double-click **`train_model.bat`** then **`run_inference.bat`** to train and score the final model on the held-out test set (these need the project dataset, see below).

On macOS/Linux use the matching `.sh` scripts (`./view_results.sh`, etc.).

## The model ladder (what you are looking at)

| ID  | Model                                   | Role                    |
|-----|-----------------------------------------|-------------------------|
| P0  | HAR pooled linear                       | ablation                |
| P1  | Price LSTM                              | ablation                |
| P2  | Price + News                            | ablation (best test QLIKE) |
| P3  | Price + News + per-ticker gate (= G1 with the graph disabled) | ablation |
| **G1** | **P3 backbone + cross-stock graph message-passing (k-NN-8 adjacency)** | **FINAL / PROPOSED MODEL** |

`G1` is the proposed model. This is a **single-basis nested ladder**: `P3` is exactly `G1` read
out with the message-passing residual disabled (graph-off determinism 0.0), so there is **no
separate G0 row** and the graph ablation is **G1 vs P3**. Every rung and every classical baseline
is scored on the **same 14,418 validation / 14,464 test observations**.

**Headline (parsimony) finding:** the cross-stock graph (G1 vs P3) adds **no statistically
significant improvement** - the held-out **test** paired-t is not significant (p=0.7913) and the
Diebold-Mariano test on QLIKE is not significant across the three seeds, at **all four horizons**
(h1/h5/h10/h22, verdict B). The simplest news-augmented backbone (**P2**) attains the lowest test
QLIKE in the study (0.559854). Classical **HAR/HARQ tie** the deep models on the level metrics
(test QLIKE 0.579291 / 0.573674, R^2 ~0.767), while the **GARCH family is far worse** (test QLIKE
1.76-1.87, R^2 ~0). G1 is reported as the proposed full model but does not beat its own ablation.

## Which command reproduces which paper number

| You want...                                   | Command / launcher                          | Output |
|-----------------------------------------------|---------------------------------------------|--------|
| The full P0->G1 results table (all 6 metrics) | `view_results.bat` / `python reproduce.py view` | console + `output/results_table.md` |
| Summary chart of the ladder                   | `view_results.bat`                          | `output/summary.png` |
| Final model (G1) on the held-out **test** set | `train_model.bat` then `run_inference.bat`  | `checkpoints/g1_final_metrics.json`, `output/g1_final_metrics.json` |

See **`PAPER_MAP.md`** for the claim-by-claim mapping.

## What works offline vs. what needs data

- **`view` (option 1) works with NO dataset and NO training.** It reads the small result
  JSONs already shipped in `results/`. This is the primary path.
- **`train` and `infer` (options 2-3) need the project dataset** under a repo that contains:
  - `data/processed/*.csv` - per-ticker OHLCV/Parkinson-volatility files
  - `data/features/dual_group_news_panel.parquet` **and** its
    `dual_group_news_panel.provenance.json` sidecar (the news panel + its causal-fit metadata)

  The bundle locates that data automatically by walking up to the nearest folder that
  contains `data/processed`. To point it somewhere explicit, set the environment variable
  `TRACK_B_DATA_ROOT` to the repo root that holds `data/`. The dataset is **not** shipped
  in this bundle (it is large); reference it by path.

## Metrics reported (all six, on raw volatility scale)

MSE, RMSE, MAE, R-squared, QLIKE, and directional accuracy (%). Directional accuracy is
computed chronologically per ticker and macro-averaged (never across ticker boundaries).

## Notes / honesty

- The P0-G1 table shows **validation** 3-seed means; `view` also prints the held-out **test**
  3-seed means and the classical-baseline test table. All rows are on the **same** masked
  consistent basis (14,418 validation / 14,464 test observations), so `P0 -> P1 -> P2 -> P3 -> G1`
  is one controlled ladder and `G1 vs P3` is the exactly-nested graph ablation. A reviewer's own
  `train`/`infer` run prints its own test metrics to the console.
- The canonical numbers come from `docs/reports/ladder_consistent_h5_2026-08-09_154402.json`
  (shipped as `results/ladder_consistent_h5.json`), `docs/reports/classical_baselines_h5_2026-08-09_182129.json`
  (`results/classical_baselines_h5.json`), and the per-horizon files `results/ladder_consistent_h{1,10,22}.json`.
- **Model Confidence Set:** not computed (per-observation prediction series were not retained in the
  result artifacts; GARCH covers a 32-ticker subset). The Diebold-Mariano tests are the primary
  graph-significance evidence.
- The model code folder is named `trackb_code/` (not `code/`) to avoid shadowing Python's
  standard-library `code` module, which otherwise breaks `pytest`/`pdb`.

## Layout

```
submission_track_b/
  reproduce.py            single entry point (menu + view/infer/train)
  START_HERE.bat/.sh      interactive menu
  view_results.bat/.sh    option 1 - view all results (no data)
  run_inference.bat/.sh   option 2 - final model G1 on test set
  train_model.bat/.sh     option 3 - train final model G1
  setup.bat/.sh           one-time environment setup
  requirements.txt        pinned dependencies
  PAPER_MAP.md            paper claim -> command -> output mapping
  trackb_code/            consolidated Track-B model code (P0-P3, G0/G1, dataset, train, eval)
  results/                small result JSONs read by `view`
  output/                 generated by `view` (results_table.md, summary.png)
  checkpoints/            G1 checkpoint + metrics produced by `train`
  test/                   tests for the reproducibility entry point
```
