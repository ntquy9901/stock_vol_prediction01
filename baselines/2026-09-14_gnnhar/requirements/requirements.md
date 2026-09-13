# GNNHAR baseline — requirements (Specify)

Date: 2026-09-14. Lifecycle: SDD (Specify → Clarify → Plan → Tasks → Implement → Validate).

## Goal
Add the published **GNNHAR** graph-neural-network HAR volatility model (Zhang, Pu, Cucuringu & Dong,
*Int. J. Forecasting* 2024, arXiv:2308.01419; official code https://github.com/chaozhang-ox/GNNHAR) as a
directly comparable baseline in the project's full-matrix protocol, on **HOSE** (run locally) and **SP500**
(Colab notebook, run later). This answers the reviewer objection that a *real* learned multi-hop
message-passing GNN could exploit cross-firm graph structure that the project's hand-built scalar
neighbour-mean (GBM+corr) cannot.

## Inputs
- Enriched per-stock frames via `scripts/eda/full_matrix.py::FM.load(market)` → `(frames, sect, edates)`.
- Modelling panel via `FM.panel(frames, {}, h)`: per-row `y = parkinson_variance.shift(-h)`, `date`,
  `ticker`, feature columns (HAR lags `har_daily/weekly/monthly`, `rq`, `mr_*`).
- Target: per-stock **Parkinson variance σ²** (a variance), column `parkinson_variance`. Horizons h∈{1,5,10,22}.
- Walk-forward folds `S1.FOLDS` from `2026-09-13` protocol, train start `S1.TRAIN_START`, embargo
  `int(h*1.6)+5` days, per-fold correlation graph `S1.build_graph` (train-only, top-k, positive weights).

## Output
- ONE JSON per market: `results/gamma_gbm/gnnhar_<market>.json`, written **incrementally after each
  horizon** (atomic tmp+replace, `out_path` param — mirrors `baselines/2026-09-13_garch/code/run_garch.py`
  `_checkpoint`) so a Colab disconnect keeps completed horizons.
- Top-level flat dicts keyed `<model>_h<h>` (so the pre-push overfit-evidence gate, which reads top-level
  `metrics`/`train_metrics`/`val_metrics`, genuinely validates every learned horizon):
  `metrics` (test), `train_metrics`, `val_metrics` (all 5: mse/rmse/mae/r2/qlike, all models),
  `fit_diagnostics` (per learned model, `overfit_check.classify_fit` verdict), `learning_curves`
  (per learned model, per seed, train+val QLIKE per epoch), `dm` (date-clustered Diebold-Mariano p-value +
  mean_diff + gain% per comparison), `qlike` convenience, `n`, `n_folds`. HOSE additionally carries
  `per_fold_qlike` (COVID-2020 / 2022 / Apr-2025 regime-spike folds dominate HOSE QLIKE and must be visible).

## Models (leave-one-out on the graph edge, per CLAUDE.md ablation rule)
- `HAR` — OLS on the 3 HAR lags, floored (paper baseline; `FM._har_ols`). Deterministic.
- `GBM` — own-8 gamma-GBM (`FM.gbm` on `[c for c in FM.OWN if c != 'rq']`), seed-ensemble. The project's
  own-history champion feature set (paper's `GBM(own-8)`).
- `GNNHAR` — **full** faithful GNNHAR2L: 3 HAR lags → Linear(F,1) own term + 2-layer GCN(nhid=9) → mlp,
  `res = relu(H1+H_gcn)`, corr graph, QLIKE loss (the paper model).
- `GNNHAR-nograph` — **Full − graph**: identical network with adjacency `A = I` (per-node nonlinear MLP,
  the paper's own key ablation). Isolates the marginal contribution of the cross-firm graph.

## Evaluation
- Pooled per-observation **QLIKE** (`metrics.per_obs_qlike`, floor `FM.FL` — the SAME floor every other
  model uses; non-negotiable for comparability), all 6 project metrics where meaningful (MSE/RMSE/MAE/R²/QLIKE;
  DirAcc N/A for pooled cross-section).
- **Date-clustered Diebold-Mariano** (`stats.date_clustered_dm`, HLN lag `h-1`) for: GNNHAR vs HAR,
  GNNHAR vs GBM(own-8), GNNHAR vs GNNHAR-nograph.

## Success criteria (go/no-go)
1. Faithful GNNHAR (GraphConvLayer `adj@(XW)+b`, `relu(H1+H_gcn)`, QLIKE loss, Adam 1e-3/wd 1e-5,
   best-val checkpoint, multi-seed ensemble) — matched to the official repo. GO iff the model code is the
   already-tested faithful core (`scripts/eda/gnnhar_sp500.py`), reused read-only.
2. HOSE run completes locally on GPU, produces `results/gamma_gbm/gnnhar_hose.json` with full over/under-fit
   evidence and honest QLIKE + DM vs HAR and GBM. **A faithful no-beat is a valid, publishable result** —
   report whatever is measured, no spin (prior project finding: graph/GNN has not beaten own-history GBM on
   QLIKE+DM).
3. SP500 notebook `notebooks/gnnhar_sp500_colab.ipynb` mirrors the resilient
   `notebooks/sp500_final_models_colab.ipynb` pattern (clone master, unpack Drive enriched bundle, install
   torch, background git-committer every 180s, run driver, incremental JSON) and is runnable with no edits.
4. Tests pass (pytest), diff-coverage C0=100%/C1≥95% on changed lines (entry `main()` `# pragma: no cover`).
5. 3-layer code review (Blind Hunter + Edge Case + Acceptance + performance lens) with critical/major fixed.

## Edge cases
- Missing tickers (IPO/delist) → masked loss + masked forward, absent cells excluded from QLIKE/DM (as
  `FM.panel` dropna does for the baselines).
- Dead-ReLU collapse under QLIKE → best-val checkpoint + restart-with-new-seed (faithful to the repo).
- Folds failing the pooled-rows gate (`< min_train`) → skipped, exactly like the sibling drivers.
- A learned horizon flagged overfit/underfit by `classify_fit` → reported honestly; not masked.

## Data-quality gate
N/A — no data change. The baseline only *reads* existing enriched frames (`data/processed_enriched/*`).
```
