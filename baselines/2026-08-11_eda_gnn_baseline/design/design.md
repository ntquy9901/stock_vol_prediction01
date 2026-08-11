# EDA-recommended GNN — Plan (design.md)

## Reuse-first architecture
This baseline imports the pooled-news-GNN pilot pipeline read-only
(`baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code`) and adds four small modules. It
does not modify any pilot module (hard isolation).

Reused as-is: `data.load_and_split_price_data`, `data.build_pooled_manifest`,
`data.build_masked_graph_manifest`, `data.GraphSnapshot/GraphManifest/GraphNode`,
`scaling.PreprocessorStore`, `train.evaluate_records`, `run_pilot._stacked_snapshot_inputs`,
`run_pilot._mean_snapshot_mse`, `run_pilot._write_graph_predictions`, `models._ResidualMessagePassing`,
`diebold_mariano.diebold_mariano`, `dm_analysis.paired_losses`.

## Data flow
```
data/processed/*_processed.csv (date, parkinson_volatility)   data/raw/prices/*_ohlcv.csv (volume)
                 │                                                        │
     load_and_split_price_data (33 tickers, 70/15/15 per ticker)         │
                 │                                                        │
        features.build_extended_frames  ◄─── volume_zscore_20 (trailing rolling z, causal) ─────┘
                 │                       ◄─── market_pk (cross-sectional median sqrt(PK) at t) ──┐
                 │                                                                               │
     features.ExtendedTickerPreprocessor.fit (TRAIN rows only): clip(±3σ), HAR(1/5/22),          │
     append vol_z + market_pk, per-column StandardScaler (train), target scaler (train)          │
                 │                                                                               │
     build_pooled_manifest → 5-feature x_price_raw windows (seq=22, h=5), graph-bound train      │
                 │                                                                               │
   ┌─────────────┼───────────────────────────────┐                                              │
   │ E0 (feat[:3])│ E1 (feat[:4]) │ E2 (feat[:5]) │  E3 / E3off / G1corr (graph, 5 feat)         │
   │ HAR linreg   │ price-LSTM    │ price-LSTM     │  build_masked_graph_manifest → swap edge     │
   │              │               │                │  edges.build_vol2pk_adjacency (TRAIN-frozen) │
   └─────────────┴───────────────┴────────────────┴──────────────────────────────────────────────┘
                 │
     eda_model.PriceGraphModel (frozen-free, end-to-end): price-LSTM encoder + optional residual
     message-passing (masked, directed vol2pk) + shared head + per-ticker positivity floor
                 │
     eda_ladder: train 20 epochs (Adam wd=1e-5, grad-clip 1.0, MSE; best val-loss checkpoint),
     evaluate val+test present-node records via evaluate_records; dump per-obs predictions;
     Diebold–Mariano (QLIKE + SE) E1/E2/E3 vs E0 and E3 vs G1corr / E3off; aggregate 3 seeds.
```

## Key design decisions
1. **Extended features, obs-set preserving.** vol_z uses `rolling(20)` (19 leading NaN) and
   market_pk has 0 NaN; monthly HAR `rolling(22)` (21 leading NaN) already binds `valid_rows`, so
   the eligible-window set is unchanged. Per-column StandardScaler ⇒ the first 3 feature columns
   are bit-identical to the 3-feature pilot, so E0 (slice `[:, :3]`) reproduces the pilot HAR P0 and
   the val/test (id,date) set matches `ladder_consistent`. (Asserted by test.)
2. **volume_zscore_20 leakage-safety.** `z = (log1p(vol) − roll20.mean) / roll20.std` uses only a
   trailing window ⇒ causal. Re-standardised by the train StandardScaler (train-only params). LPB
   has no volume ⇒ vol_z ≡ 0 (neutral), keeping its windows eligible.
3. **MarketPK.** Cross-sectional median of `sqrt(parkinson_volatility)` across all present tickers
   at date t, computed from raw (pre-clip) PK; contemporaneous ⇒ leakage-safe. Aligned to each
   ticker by date (a ticker always contributes its own value ⇒ no new NaN). Standardised by the
   train scaler.
4. **Directed vol2pk edge, TRAIN-frozen.** `tv2p[i,j] = corr(vshock_i(t), sqrt(PK)_j(t+1))` over
   TRAIN dates only (rows = 32 volume tickers as sources, cols = 33 as targets). For each target j,
   Top-5 sources by |tv2p[·,j]|; adjacency `A[j,i] = tv2p[i,j]` (signed) + self-loop `A[j,j]=1`.
   One fixed matrix applied to every snapshot, then presence-masked (absent rows/cols zeroed;
   self-loop keeps every present node valid). The EDA's "dynamic" refers to the window-varying node
   features; the neighbour identity is train-frozen for leakage-safety (EDA plan §31/§55).
5. **PriceGraphModel end-to-end (Simplicity Gate).** The pilot's frozen-encoder + base-cache path is
   an optimisation for the news backbone; a plain end-to-end 2-layer price-LSTM + residual MP + head
   is simpler, standard, and fast enough (33 nodes × ~4.5k train snapshots × 20 epochs on GPU).
   Deviation from the pilot's cache reuse recorded here as a complexity-reduction. `use_gnn`/
   `apply_message_passing=False` gives the nested E3off readout (bit-identical to E3 minus the edge).
6. **Controlled "current-G1" comparison.** The literal current G1 is a news model with the
   correlation kNN-8 edge; its per-observation test dumps were not retained on disk. A clean
   edge-only DM requires identical features/backbone, so G1corr = PriceGraphModel with the
   correlation kNN-8 edge on the identical 5-feature backbone is the controlled "current-G1 edge"
   comparison (it directly tests the EDA's central claim: vol2pk edge vs correlation edge). The
   existing news-G1 aggregate test metrics are cited from `ladder_consistent` for context only.

## Simplicity & Anti-Abstraction gates
- Simplicity: no new config surface beyond the ladder; end-to-end training instead of the
  frozen-encoder cache (fewer moving parts). PASS (deviation noted, decision 5).
- Anti-Abstraction: reuses pilot manifest/eval/DM/message-passing directly; no wrappers. PASS.

## Files
- `code/features.py` — extended feature frames + `ExtendedTickerPreprocessor` + `build_extended_store`.
- `code/edges.py` — TRAIN-frozen directed vol2pk adjacency + snapshot adjacency swap.
- `code/eda_model.py` — `PriceGraphModel` (encoder + masked residual MP + head + positivity).
- `code/eda_ladder.py` — driver (basis, train E0..G1corr, evaluate, DM, aggregate, dump).
- `test/test_features.py`, `test/test_edges.py`, `test/test_eda_model.py`, `test/test_eda_ladder_smoke.py`.

## Tasks (verifiable)
1. features.py → tests: vol_z causal (future spike no effect), market_pk contemporaneous, LPB=0,
   obs-set unchanged vs pilot, first-3-cols bit-identical to pilot. 
2. edges.py → tests: tv2p uses no date>train_end; Top-5 per target; self-loop; presence mask zeroes
   absent; adjacency frozen across snapshots.
3. eda_model.py → tests: shape correctness; use_gnn=False == apply_message_passing=False; positivity
   floors nonpositive; absent nodes never influence present outputs.
4. eda_ladder.py → smoke test: tiny basis boots E0..E3 + DM without exception (tag `smoke`).
5. Real run 3 seeds → aggregate table + DM verdicts + report.
