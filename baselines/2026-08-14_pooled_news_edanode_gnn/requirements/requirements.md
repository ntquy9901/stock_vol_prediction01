# Requirements — Pooled News + EDA-node-features + directed vol→PK GNN combo

## Objective
Test the one untested lever from the graph-EDA: combine the EDA-recommended node features
(MarketPK + volume_zscore_20, which alone DM-beat HAR on QLIKE) with PhoBERT news late-fusion
(the best QLIKE rung, P2=0.5599) and the directed volume→PK Top-5 graph, on the pooled backbone —
then test whether the full combo beats HAR (P0) and whether the graph/news add value over each
other. Prior work tested node-features price-only (eda_gnn E0–E3) and news without the extra node
features (ladder_consistent P0–G1); nobody has combined all three.

## Inputs
- `data/processed/*_processed.csv` (33 tickers, daily, `parkinson_volatility` = Parkinson VARIANCE σ²).
- `data/raw/prices/*_ohlcv.csv` (for volume_zscore_20).
- `data/features/dual_group_news_panel.parquet` (precomputed PhoBERT panel, provenance-gated).
- Horizon 5, seq 22, chronological 70/15/15 split, 3 seeds (42/123/2026).

## Rungs (one shared basis, nested)
- **P0** = pooled HAR linear regression (3 HAR features only) — the external baseline.
- **P1** = pooled price LSTM, 5 node features (HAR + MarketPK + volume_zscore_20), no news/graph.
- **P2** = P1 + PhoBERT news late-fusion (gate off), no graph.
- **P3** = P2 + per-ticker gate, graph off (the trained G1 read out with message-passing disabled).
- **G1 (the combo)** = P3 + message-passing over the directed volume→PK Top-5 adjacency.

## Output
- Per-seed `results/pooled_news_edanode_seed{seed}_<TS>/h5/ladder_metrics.json` with all 6 metrics
  (MSE/RMSE/MAE/R²/QLIKE/DirAcc) for val + test per rung.
- DM aggregate (3-seed ensemble, HLN-corrected, h=5) on the identical test observation set:
  **G1 vs P0** (combo vs HAR), **G1 vs P3** (graph effect), **P3 vs P0** (gate effect). These three
  are the rungs with per-observation test dumps. **P1/P2 DM vs HAR is NOT computed** here because
  the reused pooled rung (`run_pooled_rung`) does not emit a per-observation dump and modifying it
  would break §3.F hard isolation; P1/P2 are reported by 3-seed metric means, and the
  node-features-beat-HAR claim is already DM-established in the sibling `2026-08-11_eda_gnn_baseline`
  (E2 QLIKE DM p=0.012). Adding a combo-side P1/P2 prediction dump is a documented follow-up.

## Success criteria / go-no-go
- Leakage-safe basis (asserted): per-ticker train boundary for scalers + edge freeze, news causal
  cutoff, identical positivity floor across all compared rungs (H1/H2 from prior review), one-basis
  obs invariant (graph present-node obs == pooled obs) — **hard gate, must pass or abort**.
- Report the DM verdict **honestly per metric**; a partial win (e.g. QLIKE DM-significant) counts,
  a documented null is a valid outcome. Do NOT fabricate a win, cherry-pick a seed, or overfit test.
- Value of the run does not depend on beating HAR: either result (combo beats / ties / loses) is a
  reportable, paper-relevant finding closing the last open lever.

## Non-goals
- No new message-passing mechanism; reuse the existing news-capable GraphAblationModel (GAT) over
  the swapped vol→PK adjacency. No hyperparameter search. No horizons other than 5 (this run).
