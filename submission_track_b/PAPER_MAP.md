# Paper Map - claim -> command -> output

This table lets a reviewer verify each paper number against the code that regenerates it.

> Naming: the paper's Track-B model ladder uses the IDs **P0, P1, P2, P3, G0, G1**.
> **G1 is the final / proposed model** (P3 backbone + cross-stock graph message passing);
> P0-P3 and G0 are ablations. If a co-authored `docs/paper/track_b_paper_draft.md` later
> fixes different Table/Figure numbers, keep these model IDs as the anchor.

## Primary result - the proposed model (G1)

| Paper claim | Command | Output file it regenerates |
|-------------|---------|----------------------------|
| Proposed model **G1** result (validation, all 6 metrics) | `view_results.bat` / `python reproduce.py view` | console table + `output/results_table.md` (G1 row) |
| Proposed model **G1** on the held-out **test** split | `train_model.bat` then `run_inference.bat` (or `python reproduce.py train` / `... infer`) | `checkpoints/g1_final_metrics.json` (`test_metrics`), mirrored to `output/g1_final_metrics.json` |
| Graph effect **G0 vs G1** (message passing on/off) | `python reproduce.py view` | `results/g0g1_graph_validation_comparison.json` (`results.G0` vs `results.G1`, `paired_delta`) |

## Ablation ladder (motivating the proposed model)

| Paper claim | Command | Output file it regenerates |
|-------------|---------|----------------------------|
| Full P0->G1 ladder table, 6 metrics per model | `python reproduce.py view` | `output/results_table.md` |
| **News effect** (P2 vs P1) with paired-t significance | `python reproduce.py view` | `results/pooled_20ep_aggregate.json` -> `paired_t.news_effect_P2_vs_P1` |
| **Gate effect** (P3 vs P2) | `python reproduce.py view` | `results/pooled_20ep_aggregate.json` -> `paired_t.gate_effect_P3_vs_P2` |
| **LSTM vs HAR** (P1 vs P0) | `python reproduce.py view` | `results/pooled_20ep_aggregate.json` -> `paired_t.P1_vs_HAR_P0` |
| Per-model P0-P3 means +/- std across 3 seeds (42/123/2026) | `python reproduce.py view` | `results/pooled_20ep_aggregate.json` -> `aggregated.*` |
| Single-seed (42) P0-P3 validation detail | (inspect JSON) | `results/pooled_seed42_validation_comparison.json` |
| Screening round (5-epoch) sanity vs confirmation (up-to-10-epoch) | (inspect JSON) | `results/pooled_5ep_aggregate.json` vs `results/pooled_20ep_aggregate.json` |

## Metrics definition (applies to every row above)

Six metrics on raw volatility scale: **MSE, RMSE, MAE, R-squared, QLIKE, directional accuracy (%)**.
Directional accuracy is computed chronologically per ticker (sign of successive target changes vs
sign of successive prediction changes) and macro-averaged across tickers; comparisons never cross
ticker boundaries.

## Scope caveats (state these when citing)

- The numbers in the shipped `results/*.json` are **validation** metrics: the pilot is a
  validation-only screening/confirmation experiment. The final model's **test** numbers come from
  the `train`/`infer` path (they require the dataset).
- **P0-P3** are a pooled ablation family (pooled validation set); **G0-G1** are a graph
  common-date family (common-date validation set). The two families use different sample sets, so
  `P3 -> G1` is not one controlled increment - report them as two separate ablation studies.
- The G0/G1 rows are single-seed (42). The G0/G1 graph variant may be refreshed by a separate
  "latest-P3 backbone" run; the P0-P3 confirmation numbers (3-seed) are stable.

## Exact reproduction command for the final model

```
# 1. one-time
setup.bat                       # or: pip install -r requirements.txt

# 2. train the final model G1 (seed 42), save checkpoint, score val+test
python reproduce.py train       # -> checkpoints/g1_final.pt, checkpoints/g1_final_metrics.json

# 3. re-score the saved final model on the test split
python reproduce.py infer       # loads checkpoints/g1_final.pt, prints 6 test metrics
```

`train`/`infer` auto-locate the dataset by walking up to the nearest folder containing
`data/processed`; override with `TRACK_B_DATA_ROOT=<repo-root-with-data>`.
