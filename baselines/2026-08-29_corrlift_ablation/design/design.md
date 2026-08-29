# Design — Combined corr+lift edge probe

**Date:** 2026-08-29 · **Baseline id:** `2026-08-29_corrlift_ablation`

## Data flow
```
screened HNX tickers (EFA.screened_tickers)                 raw HNX OHLCV close
        |                                                          |
        v                                                          v
EFA._write_estimator_processed (parkinson)              corrlift_edge.load_close_wide
        |                                                          |
        v                                                          v
MR.build_masked_rich  -> D (X/nmask/tmask/y, adj_vol2pk, d_va, ...)  |
        |                              cutoff = D.d_va[0] (train/val boundary)
        |                                                          v
        |                         corrlift_edge.build_corrlift_adjacency(close_wide, cutoff)
        |                              (returns->Pearson |rho|>0.7  OR  lift>1.7, train-only, frozen)
        |                                                          |
        v                                                          v
        +--------> RMR.train_masked_rich(D, cfg, seed, use_graph, adj) x {variants} x {seeds}
                                              |
                                              v
                   metrics (MSE/RMSE/MAE/QLIKE/R2) + per-seed stats + date-clustered DM
                   + over/under-fit evidence (train/val/test + verdict + curves)
                                              |
                                              v
                   results/corrlift_ablation/corrlift_ablation_hnx_h1.json
```

## Files
- `code/corrlift_edge.py` — the NEW edge. Pure numpy/pandas, no torch, no training-path edits.
  - `load_close_wide(tickers, price_dir)`: union-date close panel aligned to `tickers` (NaN off own dates;
    all-NaN column for a ticker with no price file -> singleton node).
  - `daily_returns(close_wide)`: per-ticker `pct_change` on that ticker's OWN trading dates, reindexed to the
    union index (no cross-gap returns). `[T,N]`.
  - `pearson_corr(returns, min_overlap)`: pairwise Pearson rho on co-finite days; NaN when overlap < min.
  - `move_events(returns)`: per-stock TRAIN-median-|return| threshold; boolean notable-move `[T,N]` + finite mask.
  - `pairwise_lift(events, valid, min_pairs)`: market-basket support/lift on co-observed days; NaN when
    co-observed days < min_pairs or a support is 0.
  - `build_corrlift_adjacency(close_wide, cutoff_date, corr_thresh, lift_thresh, min_overlap, min_pairs)`:
    slices train rows (`date < cutoff_date`), computes rho + lift, combines to a symmetric `[N,N]` float32
    adjacency (self-loop=1), and returns `(adj, diag)` where `diag` has the edge-density breakdown.
- `code/run_corrlift_ablation.py` — thin orchestrator mirroring `2026-08-29_sector_gat_ablation`'s
  `run_sector_ablation.py`. Reuses the READ-ONLY pipeline modules (EFA, MR, RMR, VE, config) — same import
  list and helper shapes as the sector runner (`build_panel_masked`, `forward_pass_smoke`, `_train_variant`,
  `run_training`, `run_dry`, `main`). The corr+lift adjacency is a drop-in replacement for the sector adj.
- `test/test_corrlift_edge.py`, `test/test_corrlift_runner.py` — UNIQUE basenames (avoid the pytest
  prepend-import duplicate-basename shadowing that silently skips tests; do NOT reuse `test_runner.py` /
  `test_smoke_forward.py`).

## Key design decisions
- **Weight (edge magnitude, unsigned).** The paper's criteria are `|rho|>0.7` and `lift>1.7` — both are
  association STRENGTHS, not signed effects — so the combined weight is a non-negative magnitude in [0,1]:
  `|rho|` (corr side) and min-normalised `lift-1` (lift side), averaged over the criteria that fired. Kept in
  [0,1] to match the statistical adjacency's magnitude scale so `WeightedGATLayer` consumes it unchanged.
  Self-loop = 1.0 (same convention as `adj_vol2pk` / `adj_corr`).
- **Leakage cutoff = `D.d_va[0]`** (first val TARGET date). Every close row with `date < cutoff` is strictly
  BEFORE every val/test target date, so returns/corr/support/lift/threshold are computed with NO evaluation
  leakage and frozen — the strict interpretation the paper omits. Note this boundary is marginally LOOSER than
  the delivered `adj_corr` / `adj_vol2pk` cut, which uses the last-TRAIN-target row (`last_tr_row`) and thus
  excludes the ~horizon purge-gap rows between the last train target and the first val target; `d_va[0]`
  includes those few rows. The difference is a handful of rows out of thousands and touches no val/test data
  (confirmed nil-effect here: the graph is near-empty regardless). A test asserts post-cutoff rows do not
  change the adjacency.
- **Pairwise co-observed overlap** for both rho (min_overlap=100, matches `MR.EDGE_MIN_OVERLAP`) and lift
  (min_pairs=30, matches `MR._MIN_PAIRS`): HNX tickers list at different times, so pair statistics use only
  days both traded — the fair market-basket transaction set — mirroring the directed vol->PK edge's handling.
- **Per-stock item = notable move.** Threshold = the stock's TRAIN-median `|return|`; `event = |return| >
  threshold`. Documented as a per-stock notable-move indicator, NOT a universal cutoff (fair across HNX's
  heterogeneous liquidity).

## Gates (per §5 SDD)
- **Simplicity Gate:** one new pure-numpy edge module + a thin runner copied from the sector pattern; no new
  abstraction/config framework. PASS.
- **Anti-Abstraction Gate:** reuses `MaskedRichNet` / `train_masked_rich` / `build_masked_rich` directly;
  no wrapper around torch/pandas. PASS.
- **Performance/Batching Gate:** training reuses the delivered batched `train_masked_rich` (batched
  `[B,N,seq,5]` tensors + batched adjacency, GPU when available). The edge build is a one-off pairwise
  O(N^2 * T_train) numpy pass on the frozen train slice — not in any hot loop. Batch <= 32 keeps VRAM < 8GB
  on the RTX 4060 single process. PASS.

## Expected result (honesty bar)
5 prior edges are NULL on HNX h1 vs a no-graph LSTM (QLIKE ~1.80-1.83). With `|rho|>0.7` on thin HNX returns
the correlation criterion may fire on very few pairs (near-empty graph) — itself a reportable finding. A 6th
null is a valid, strong robustness result; report straight. Only claim a lift if DM p-values + seed-stability
support it.
