# Requirements — Graph WaveNet ablation on HNX daily volatility

Date started: 2026-08-29. Panel: HNX. Horizon: 1. Seeds: {42, 123, 2026}.

## Objective (question answered)
The delivered 4-edge sweep (statistical corr, sector, MTGNN-learned, DY-spillover) showed **no graph
edge beats a no-graph LSTM on HNX h1**. All prior graphs were bolted onto the SAME LSTM temporal branch.
This baseline asks a different, harder question:

> Does swapping the whole temporal + spatial stack for a **Graph WaveNet** — a dilated-causal-conv (TCN)
> temporal backbone + a **self-adaptive adjacency** learned graph — change the "graph does not help HNX
> daily volatility" picture?

Because Graph WaveNet replaces the temporal backbone (TCN, not LSTM) as well as the graph, this is framed
honestly as a **temporal-backbone experiment**, NOT a clean leave-one-out edge ablation. The clean,
in-family controlled comparison is the paper's own ablation: **GWN with the self-adaptive graph** vs
**GWN with the self-adaptive graph removed** (identity / no graph conv = pure TCN). That isolates the
adaptive-graph variable while holding the TCN backbone fixed.

## Paper fidelity (MANDATORY — CLAUDE.md "named-method must use published formula")
Reproduce Wu, Pan, Long, Jiang, Zhang, "Graph WaveNet for Deep Spatial-Temporal Graph Modeling",
IJCAI 2019 (arXiv:1906.00121); official code github.com/nnzhan/Graph-WaveNet (`model.py`
`nconv`/`linear`/`gcn`/`gwnet`). Components:
- **Self-adaptive adjacency** `A_adp = SoftMax(ReLU(E1 · E2ᵀ))` with learnable node embeddings E1, E2
  (official: `adp = softmax(relu(mm(nodevec1, nodevec2)), dim=1)`, nodevec1∈ℝ^{N×c}, nodevec2∈ℝ^{c×N}).
- **Dilated causal convolution** temporal with **gated activation** `tanh(·) ⊙ sigmoid(·)`,
  **residual + skip** connections, WaveNet-style stacked dilations (1,2,1,2,… reset each block).
- **Diffusion graph conv** (`gcn`/`nconv`): order-K propagation `x·A` concatenated across supports.
- Paper→code equation mapping DOCUMENTED in the module docstring + the report.
- An **independent test** recomputes `softmax(relu(E1 E2ᵀ))` from the raw parameters (NOT reusing the
  module's own forward) and matches the module (CLAUDE.md named-formula rule).

## Inputs (reused READ-ONLY from the delivered pipeline)
- Masked-union panel builder `masked_rich.build_masked_rich` — 5 node features
  `[parkinson_volatility, har_weekly, har_monthly, market_pk, volume_zscore_20]`, per-ticker StandardScaler,
  node/target masks, chronological 80/10/10 split, purge.
- HAR + HAR-X anchors (`baselines.har_fit/har_predict` + 5-feature OLS) — identical basis as siblings.
- QLIKE positivity floor `1e-2·t_mean` (shared across all compared models) + `cfg.qlike_floor` scoring.
- Date-clustered Diebold-Mariano (`run_masked_rich._dm_all` → `stats.date_clustered_dm`).
- Over/under-fit verdict `overfit_check.classify_fit`.
- Same seeds/folds for every variant so ONLY the model differs.

## Variants (same folds/seeds)
| key | model | role |
|-----|-------|------|
| `HAR` | HAR-RV OLS (3 feats) | context anchor (deterministic) |
| `HAR-X` | 5-feature linear OLS | context anchor (deterministic) |
| `LSTM` | no-graph LSTM (delivered `train_masked_rich`, `use_graph=False`) | DM reference + gate evidence |
| `LSTM_wGAT_vol2pk` | LSTM + weighted-GAT on directed vol→PK edge | prior-art null + gate evidence |
| `GWN_adaptive` | Graph WaveNet WITH self-adaptive adjacency | the graph model under test |
| `GWN_no_adaptive` | Graph WaveNet WITHOUT graph conv (pure TCN) | paper's "w/o adaptive" ablation |

## Outputs
- `results/graphwavenet_ablation/graphwavenet_ablation_hnx_h1.json` — metrics (MSE/RMSE/MAE/QLIKE/R²)
  ensemble + per-seed, `train_metrics`/`val_metrics`/`fit_diagnostics`/`learning_curves` per variant,
  date-clustered DM verdicts.
- `docs/reports/2026-08-29_graphwavenet.md` — metric table + DM + fit verdicts + paper→code mapping +
  honest conclusion.

## Success criteria (go/no-go)
1. GWN model reproduces the paper components (adaptive adj, dilated-causal gated TCN, diffusion gcn) with a
   documented equation→code mapping and an independent adaptive-adjacency test that PASSES.
2. `run_training` produces the result.json with ALL variants on the SAME folds/seeds, carrying full
   over/under-fit evidence (train/val/test + verdict + curves) per learned variant.
3. Date-clustered DM computed for: `GWN_adaptive vs GWN_no_adaptive`, `GWN_adaptive vs LSTM`,
   `GWN_no_adaptive vs LSTM`, `GWN_adaptive vs HAR`, `GWN_adaptive vs HAR-X`.
4. Pre-push gate passes: C0 line=100% + C1 branch≥95% on changed lines (diff-cover), `ruff --select F`
   clean, lessons-regression, overfit-evidence gate. UNIQUE test basenames (no `test_runner.py` /
   `test_smoke_forward.py` collision).
5. 3-lens adversarial code review; critical/major fixed.
6. Report the result STRAIGHT — a fifth null / no-lift is a valid strong robustness finding; a positive
   result is claimed only with DM p-values + seed-stability.

## Non-goals
- Not tuning GWN hyperparameters to win (fixed faithful architecture; channel widths reduced from the
  paper's traffic defaults for VRAM + small-data, documented).
- Not re-running the other panels/horizons (HNX h1 only, matching the 4-edge sweep's headline cell).
