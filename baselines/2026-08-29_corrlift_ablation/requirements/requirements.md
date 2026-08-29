# Requirements — Combined corr+lift edge probe (6th HNX graph-edge ablation)

**Date:** 2026-08-29
**Baseline id:** `2026-08-29_corrlift_ablation`

## Objective
Build and DM-test a 6th graph-edge construction on HNX daily volatility, replicating the edge of
Sonani, Badii & Moin (2025), "Stock Price Prediction Using a Hybrid LSTM-GNN Model", arXiv:2502.15813,
§3.2 — a COMBINED linear (Pearson-correlation) + non-linear (association-rule lift) undirected edge.
This extends a completed 5-probe sweep in which NO edge (statistical vol->PK, sector ICB, MTGNN-learned,
DY-spillover, Graph-WaveNet-adaptive) beat a no-graph LSTM on HNX h1 (no_graph LSTM QLIKE ~1.80-1.83).

## Inputs
- HNX screened Parkinson-variance panel (delivered pipeline, read-only): the masked union-of-dates panel
  with 5 node features `[parkinson_volatility, har_weekly, har_monthly, market_pk, volume_zscore_20]`,
  HAR-X residual anchor, per-ticker StandardScaler, QLIKE positivity floor, chronological train/val/test
  splits (0.80/0.10/0.10) and seeds.
- Raw HNX OHLCV (`data/raw/prices/hnx_vnstock/<TICKER>_ohlcv.csv`) for CLOSE prices -> returns.

## Edge construction (paper §3.2, published formulas)
ONE undirected weighted adjacency `A [N,N]`, self-loop = 1.0 (WeightedGATLayer convention), built from BOTH:
1. **Linear — Pearson correlation on daily returns.** `r_t = (P_t - P_{t-1})/P_{t-1}` (Eq.2); Pearson rho_ij
   between each pair's return series (Eq.3, standard). Edge fires if `|rho_ij| > 0.7`.
2. **Non-linear — association-rule lift (Apriori).** Per-stock "notable move" item: a trading day whose
   `|return|` exceeds that stock's TRAIN-median `|return|` (a per-stock notable-move indicator). A transaction
   = one trading day. `support(X) = P(X)` (fraction of transactions containing X); `lift(i,j) =
   support(i,j)/(support(i)*support(j))` (standard market-basket definitions). Edge fires if `lift_ij > 1.7`.

Combine: edge present if EITHER criterion fires. Weight = documented mean of the two normalised strengths
(`|rho|` in [0,1]; lift excess `lift-1` min-normalised over present lift edges), averaging only the criteria
that fired. Undirected (symmetric).

## Leakage rule (STRICT — our H1 lesson; the paper is silent on this)
The ENTIRE graph (returns, correlations, supports, lifts, per-stock item threshold) is computed from TRAIN
ROWS ONLY (close-price rows strictly before the train/val boundary date `D.d_va[0]`), then FROZEN for
val/test. Mirrors how `masked_rich` fits `adj_vol2pk` train-only.

## Controlled comparison (leave-one-out edge ablation)
Same folds/seeds/floor/scaler/2-hop WeightedGATLayer for three variants:
1. `no_graph_LSTM` — `use_graph=False`.
2. `stat_GAT_vol2pk` — shipped directed volume-shock->PK edge (context).
3. `corr_lift_GAT` — the new combined edge (drop-in adjacency replacement).

## Run configuration
- Panel HNX, horizon 1, 10 epochs (early stop), 3 seeds {42, 123, 2026}.
- GPU venv `.venv_gpu_encode/Scripts/python` (torch 2.6.0); single process; small batch (<= 32) under 8GB VRAM.
- Date-clustered Diebold-Mariano (QLIKE): corr_lift vs no_graph, corr_lift vs stat_vol2pk.

## Success criteria (go/no-go)
- Edge module: Pearson rho AND lift each match an INDEPENDENT recompute on a tiny fixture (named-method =
  published formula). Leakage: post-cutoff rows do not change the adjacency.
- Runner emits `results/corrlift_ablation/corrlift_ablation_hnx_h1.json` with per-variant metrics
  (MSE/RMSE/MAE/QLIKE/R2), per-seed stats, date-clustered DM, and over/under-fit evidence (train/val/test
  metrics + fit verdict + per-seed learning curves).
- Edge density reported: how many edges each criterion (corr, lift, either) contributes on the HNX nodes.
- Report `docs/reports/2026-08-29_corrlift.md` with metric table + DM verdicts + fit verdicts + edge density
  + paper->code formula map + honest conclusion.
- Pre-push gate green (C0=100% / C1>=95% on changed lines, ruff -F clean, lessons + overfit gates) WITHOUT
  QG_SKIP; 3-lens code review, critical/major fixed.

## Non-goals
- Not a final/production run. A 6th null is a valid robustness finding; report straight (objective wording).
  The paper predicts NORMALIZED PRICE on 10 US stocks with no val set / no DM / no seeds — an easier target
  and weaker methodology than this DM-tested HNX-volatility setup; frame accordingly.
