# Requirements — MTGNN learned-graph ablation (HNX volatility)

## Goal
Test whether a **learned / adaptive** graph adjacency (MTGNN, Wu et al. 2020, arXiv:2005.11650) is a
better edge for the HNX volatility model than (a) no graph, (b) the shipped statistical vol→PK GAT edge,
and (c) the static sector edge — all under the SAME MaskedRichNet / HAR-X pipeline and folds, so only the
**edge mechanism** differs.

## Inputs
- HNX panel (screened universe, `estimator_forecast_ablation.screened_tickers("hnx")` → 162 tickers,
  N≈154 after the masked-panel min-valid filter), horizon 1, lookback 10.
- 5 node features `[parkinson_volatility, har_weekly, har_monthly, market_pk, volume_zscore_20]`
  (built read-only by the delivered pipeline).

## Outputs
- Learned adjacency built inside the model each forward (trainable), directed + top-k sparse + self-loop.
- Result JSON → `results/learned_graph_ablation/learned_graph_ablation_hnx_h1.json`:
  all 5 metrics (MSE, RMSE, MAE, QLIKE, R²) as ensemble + per-seed mean±std for 4 variants, over/under-fit
  evidence (train/val/test + fit verdict + learning curves), and date-clustered Diebold–Mariano tests.
- Short report → `docs/reports/2026-08-29_learned_graph_mtgnn.md`.

## Acceptance criteria
- Graph-learning layer implements MTGNN Eqs. (1)-(3) + top-k (Eqs. 5-6) faithfully, verified against the
  paper text and the official `nnzhan/MTGNN graph_constructor` code (cited in the module docstring), with a
  test that recomputes the equations independently.
- Adjacency is [N,N], directed/asymmetric, top-k sparse (≤k outgoing edges/node), self-loop, differentiable
  with finite gradients — all unit-tested.
- Everything except the edge source is identical to the fixed-edge variants (same LSTM branch, same 2-hop
  WeightedGATLayer, same masked panel, HAR-X anchor, per-ticker scalers, QLIKE floor).
- Run HNX h1, 10 epochs, ≥3 seeds; report mean±std + DM p-values (learned vs no-graph, learned vs stat-GAT,
  learned vs sector-GAT).
- Pass unit tests + smoke + pre-push quality gate (C0=100/C1≥95 on changed lines, ruff-F); capture
  over/under-fit evidence.

## Go / No-go verdict
Answer: **does the MTGNN-learned adjacency significantly improve QLIKE over no-graph AND over the
statistical edge on HNX** (date-clustered DM, α=0.05)? Report sign + p-value; a null/negative result is a
valid, reportable outcome.

## Hard constraints
- Do NOT edit live-training-path files (`baselines/2026-08-21_har_anchored_residual/code/*`,
  `scripts/eda/*`) — import read-only.
- GPU is shared; CPU-forced by default. Watch CUDA OOM; do not kill other processes.
- Fixed date strings in artifacts (no `datetime.now()`).
