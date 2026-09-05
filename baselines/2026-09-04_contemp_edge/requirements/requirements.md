# Contemporaneous-edge ablation — requirements

**Type:** probe / edge-construction ablation (reuses the delivered VolGA model unchanged; swaps only
the graph edge). Not a new architecture.

## Goal
Test whether replacing the delivered directed volume→Parkinson Top-5 edge with a symmetric
contemporaneous √PK correlation Top-5 edge changes the graph branch's marginal value (VolGA − no-graph
LSTM), holding the five node features and every other choice fixed.

## Input / output
- **Input:** enriched walk-forward panels (`enriched_glob(market)`), five node features, lookback 22.
- **Output:** `results/contemp_edge/contemp_contemp_{market}_h{H}.json` with `metrics` (HAR, HAR-X,
  LSTM, VolGA on MSE/RMSE/MAE/QLIKE) and `dm_date_clustered` (VolGA-vs-LSTM, VolGA-vs-HAR-X on
  QLIKE/SE/AE).

## Acceptance criteria
- Edge is **train-only** (no look-ahead): weights computed from rows ≤ last train anchor + horizon.
- Self-loop = 1; Top-K = `MR.EDGE_TOP_K`; NaN-safe on thin/degenerate node histories.
- Uniform protocol across horizons within a run (folds/seeds identical for h1/h5/h10/h22).
- Every reported number traces to a stored results JSON.

## Go / no-go
Reported honestly as a probe. **NOT** claimed as an improvement over the directed edge unless the
graph's marginal value (VolGA − LSTM) is significant and stable across horizons on a matched protocol.
Result: no-go as an improvement — contemp edge carries no QLIKE significance over the no-graph LSTM at
any horizon on VN100 or VN30 (7 folds, 3 seeds).
