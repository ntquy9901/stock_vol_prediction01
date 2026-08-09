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
| P2  | Price + News                            | ablation                |
| P3  | Price + News + per-ticker gate          | ablation (graph backbone) |
| G0  | Backbone, graph message-passing OFF     | ablation                |
| **G1** | **Backbone + graph message-passing ON (k-NN-8 adjacency)** | **FINAL / PROPOSED MODEL** |

`G1` is the proposed model. `P0-P3` and `G0` are the ablations that motivate it.

**Headline (parsimony) finding:** the graph layer (G1 vs G0) adds **no statistically
significant improvement** - the Diebold-Mariano test on QLIKE is not significant across the
three seeds, and the held-out **test** QLIKE is actually slightly worse for G1. G0 and G1 are
otherwise near-identical on validation. The honest takeaway is that the simpler model is
preferred; G1 is reported as the proposed full model but does not beat its ablation.

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

- The P0-G1 table shows **validation** metrics (3-seed means). The **G0/G1 rows are the
  definitive masked-manifest, screening-P3-backbone run** (k-NN-8 adjacency for G1) over the same
  14,418 validation observations - these match the paper. The bundle also ships G1's held-out
  **test** 3-seed mean (shown in `view`); a reviewer's own `train`/`infer` run prints its own
  test metrics to the console.
- `P0-P3` (pooled family) and `G0-G1` (graph family) are two **separate** studies scored on
  different evaluation sets, so `P3 -> G1` is not a single controlled step.
- The G0/G1 numbers come from `docs/reports/verdict_masked_g0g1_newbackbone_2026-08-09_120512.json`.
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
