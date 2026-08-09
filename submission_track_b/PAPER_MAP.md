# Paper Map - claim -> command -> output

This table lets a reviewer verify each paper number against the code/data that regenerates it.

> Naming: the paper's Track-B model ladder is a single-basis nested ablation
> **P0 -> P1 -> P2 -> P3 -> G1**. **G1 is the final / proposed model** (P3 backbone + cross-stock
> graph message passing). **P3 is exactly G1 read out with the graph disabled** (graph-off
> determinism 0.0), so there is **no separate G0 row**; the graph ablation is **G1 vs P3**. Every
> rung and every classical baseline is scored on the **same 14,418 validation / 14,464 test
> observations**. Canonical source: `docs/reports/ladder_consistent_h5_2026-08-09_154402.json`
> (shipped here as `results/ladder_consistent_h5.json`).

## Primary result - the proposed model (G1)

| Paper claim | Command | Output file it regenerates |
|-------------|---------|----------------------------|
| Proposed model **G1** - k-NN-8, 3-seed VAL mean (mse 2.11845e-06, rmse 0.00145549, mae 0.00046206, r2 0.745403, qlike 0.509102, DirAcc 48.683) | `view_results.bat` / `python reproduce.py view` | console table + `output/results_table.md` (G1 VAL row); `results/ladder_consistent_h5.json` -> `rung_metrics.val.G1` |
| Proposed model **G1** - 3-seed TEST mean (mse 5.31428e-06, rmse 0.00230527, mae 0.000599607, r2 0.763520, qlike 0.575926, DirAcc 48.221) | `python reproduce.py view` | `results/ladder_consistent_h5.json` -> `rung_metrics.test.G1` |
| Graph ablation **G1 vs P3** and the **parsimony finding** (graph adds no significant gain: held-out test paired-t p=0.7913 n.s.; Diebold-Mariano on QLIKE not significant; verdict B) | `python reproduce.py view` | `results/ladder_consistent_h5.json` -> `graph_effect_dm`, `graph_effect_verdict.test` |
| Exact nesting: removing the graph from G1 lands on **P3** (graph-off readout determinism 0.0, all seeds) | (inspect JSON) | `results/ladder_consistent_h5.json` -> `nesting_check.*.graph_off_readout_determinism_max_abs_diff` |

## Ablation ladder (motivating the proposed model)

| Paper claim | Command | Output file it regenerates |
|-------------|---------|----------------------------|
| Full P0->G1 ladder table, 6 metrics per model, VAL + TEST | `python reproduce.py view` | `output/results_table.md`; `results/ladder_consistent_h5.json` -> `rung_metrics.{val,test}` |
| **News effect** (P2 vs P1): P2 has the lowest test QLIKE in the study (0.559854 vs P1 0.564780); paired-t on QLIKE t=-9.50 test / -7.69 val (df=2, |t|>4.30) | (derive from per-seed) | `results/ladder_consistent_h5.json` -> `rung_metrics.*.{P1,P2}.*.per_seed` |
| **Gate effect** (P3 vs P2): gate raises QLIKE (P3 test 0.576488 > P2 0.559854; paired-t t=+35.8 test) | (derive from per-seed) | `results/ladder_consistent_h5.json` -> `rung_metrics.*.{P2,P3}.*.per_seed` |
| **LSTM vs HAR** (P1 vs P0): temporal learning lowers test RMSE and QLIKE | (derive from per-seed) | `results/ladder_consistent_h5.json` -> `rung_metrics.*.{P0,P1}.*.per_seed` |

## Classical econometric baselines (grounding the deep ladder)

| Paper claim | Command | Output file it regenerates |
|-------------|---------|----------------------------|
| HAR / HARQ **tie** the deep models on level metrics (test QLIKE 0.579291 / 0.573674, r2 ~0.767); GARCH family far worse (test QLIKE 1.76-1.87, r2 ~0) | `python reproduce.py view` | `results/classical_baselines_h5.json` -> `rung_metrics.test.*` |
| GARCH family covers 32/33 tickers (LPB lacks raw OHLCV); all other baselines cover the full 14,464 test obs | (inspect JSON) | `results/classical_baselines_h5.json` -> `notes`, `garch_excluded_tickers` |

## Multi-horizon graph verdict (h1 / h5 / h10 / h22)

| Paper claim | Command | Output file it regenerates |
|-------------|---------|----------------------------|
| Graph verdict **B (null) at all four horizons** (G1 does not beat P3 on QLIKE under DM at h1/5/10/22) | `python reproduce.py view` | `results/ladder_consistent_h{1,5,10,22}.json` -> `graph_effect_verdict.test`; summary in `results/ladder_multihorizon.md` |
| h1 nuance: DM-MSE significant-negative on all 3 test seeds but QLIKE unstable (seed-2026 near-floor 1-day outlier) | (inspect) | `results/ladder_consistent_h1.json` -> `graph_effect_dm`; `results/ladder_multihorizon.md` |
| h22 nuance: graph improves VAL QLIKE (paired-t p=0.0002) but does not generalize to TEST (0/3 seeds) | (inspect) | `results/ladder_consistent_h22.json` -> `graph_effect_verdict`; `results/ladder_multihorizon.md` |

## Metrics definition (applies to every row above)

Six metrics on raw volatility scale: **MSE, RMSE, MAE, R-squared, QLIKE, directional accuracy (%)**.
Directional accuracy is computed chronologically per ticker (sign of successive target changes vs
sign of successive prediction changes) and macro-averaged across tickers; comparisons never cross
ticker boundaries.

## Scope caveats (state these when citing)

- The shipped ladder rows are **3-seed means** (seeds 42/123/2026) on the masked consistent basis:
  same 14,418 validation / 14,464 test observations across every rung and every classical baseline.
  A reviewer's own `train`/`infer` run (requires the dataset) prints its own numbers.
- **Parsimony finding (state this):** G1 does not significantly beat P3 - the held-out test paired-t
  is not significant (p=0.7913) and the Diebold-Mariano test on QLIKE is not significant across
  seeds, at any of the four horizons. The cross-stock graph layer adds no measurable value; the
  simpler news-augmented backbone (P2) attains the lowest test QLIKE in the study.
- **Model Confidence Set:** not computed. The result artifacts store per-configuration metrics but
  not per-observation prediction series, and clean alignment onto one common observation set (GARCH
  covers a 32-ticker subset) was not feasible in the submission window; the Diebold-Mariano tests are
  the primary graph-significance evidence.

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
