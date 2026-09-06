# Design — MASTER for Parkinson-variance forecasting

## Data flow
```
enriched CSVs ──build_enriched_panel──▶ EnrichedPanel(pk[T,N], feats[T,N,5], anchors[A], masks, target_dates)
                                          │
              frozen_universe (train-row screen, once)
                                          │
   make_folds(n, test_start, K, val, h) ──▶ expanding-window folds (assert_no_leakage)
                                          │  per fold:
   pack_fold ──▶ D: X_*[A,N,T,5] (train-only per-node feature scaler), y_*[A,N] (raw pk[t+h]),
                    tmask/nmask, t_mean/t_std[N], d_va/d_te, har_*/har5_*  (all delivered, READ-ONLY)
   market_features.compute_market_raw(pk, vshock, window) ──▶ mkt_raw[T,4] (causal)
   market_features.fit_market_scaler(mkt_raw, last_train_anchor) ──▶ (mean,std)  (TRAIN-only)
   market_features.pack_market(mkt_std, anchors, lookback) ──▶ Xm_*[A,T,4]
                                          │
   models on identical folds/seeds/floor:
     HAR (3-feat OLS), HAR-X (5-feat OLS)          via run_walkforward._har_ols_preds  (READ-ONLY)
     LSTM (no graph), VolGA (2-hop wGAT, hmatched) via run_masked_rich.train_masked_rich (READ-ONLY)
     MASTER (this baseline)                        via master_net + train_master
                                          │
   pool per-(node,date) preds over OOS ──▶ metrics + non-lock QLIKE + winrate vs HAR-X
                                       ──▶ date-clustered DM: MASTER vs HAR-X / LSTM / VolGA
                                       ──▶ train/val/test fit evidence + learning curves
                                       ──▶ results/master_transformer/master_<market>_h<h>.json
```

## [N,T,D] mapping and gate design
- Each **anchor** (one trading date t) is one MASTER sample: the cross-section of N stocks, each with a
  T=lookback window of D features. The batched forward takes `[B, N, T, D]` (B anchors) — see below.
- D = 5 stock features (already per-node standardized by `pack_fold`) followed by M=4 market features.
  `gate_input_start_index = 5`, `gate_input_end_index = 5 + 4 = 9`.
- MASTER forward: `src = x[..., :5]`; `gate_input = x[:, :, -1, 5:9]` (market features at the **last**
  timestep — the current market state, causal); `src = src * Gate(gate_input)` (per-feature soft gate,
  softmax over d_feat scaled by beta); then Linear->PosEnc->TAttention (intra-stock temporal)->
  SAttention (**dense** inter-stock, softmax over ALL stocks at each timestep — the contrast to VolGA's
  sparse Top-5 edge)->TemporalAttention (aggregate over T)->Linear(d_model,1) -> `[B, N]`.

### Market features (M=4, all causal, per date t)
1. cross-sectional mean of Parkinson variance over stocks (market vol level)
2. cross-sectional dispersion (std) over stocks (market dispersion)
3. cross-sectional mean of volume z-shock (`volume_zscore`) (market volume shock)
4. causal rolling z of feature 1 over a 22-day window (how extreme today's market vol is)

Broadcast across stocks (same value for every stock at a given date), filled at every window timestep
(only the last is read by the gate). Standardized per fold with a scaler fit on TRAIN dates only
(rows `[0 : last_train_anchor]`), so val/test never enter the market scaler (no look-ahead).

## Normalization (proven pattern — CLAUDE.md LSTM-GNN lesson)
- Stock features: per-node train-only StandardScaler (from `pack_fold`).
- Target: per-node z-score `(y - t_mean)/t_std` (train-only), **linear** decoder output (no Softplus/ReLU),
  inverse-transform at eval `pred*t_std + t_mean`, then shared positivity floor
  `max(pred, POS_FLOOR_FRAC*t_mean + POS_FLOOR_EPS)`. Identical basis to `train_masked_rich` (zscore_floor)
  so MASTER is compared to LSTM/VolGA on the same floor; QLIKE uses the single shared `qlike_floor=1e-8`.

## Batched MASTER (vendored + adapted)
The reference modules process one `[N,T,D]` cross-section per forward (batch=1 over anchors, N as the
effective batch for the attention). To use the RTX 4060 well we add a leading anchor-batch dim `B` and
rewrite the three attention modules with explicit batched axes — **mathematically identical per anchor**
(no cross-anchor mixing): TAttention batches over `(B,N)` and attends over T; SAttention permutes to
`[B,T,N,dim]` and attends over N within each `(B,T)`; TemporalAttention aggregates over T per `(B,N)`.
A B=1 batched forward equals the per-anchor reference (pinned by test). Attribution header credits
arXiv:2312.15235 + the MIT-licensed SJTU-DMTai/MASTER repo.

## Config gates (SDD)
- **Simplicity Gate:** reuse ALL delivered walk-forward / fold / metric / DM / evidence machinery
  read-only; only the MASTER module, the market-feature builder, and the driver are new.
- **Anti-Abstraction Gate:** vendor the MASTER nn.Modules directly (adapted for batching); no wrapper
  framework; the reference qlib `SequenceModel` training loop is NOT vendored (we mirror the proven
  `train_masked_rich` loop instead).
- **Performance/Batching Gate:** anchors are batched `[B,N,T,D]` (no per-item batch=1), tensors stay on
  GPU inside the training loop, minibatch inference; VRAM peak is the SAttention `[B,T,N,N]` scores —
  bounded by modest `d_model=128` and `batch<=64` for N=102. Justified in code comments.
- **Config single-source:** all MASTER tunables (d_model, heads, dropout, beta, lr, epochs, patience,
  batch) live in `master_config.py`; floors/windows/seeds/lookback come from the canonical
  `pipeline_config`. No magic numbers in the driver / net.

## Files
- `code/master_net.py` — vendored+adapted batched MASTER nn.Module (attribution header).
- `code/master_config.py` — MasterConfig dataclass + tunables (single source for this baseline).
- `code/market_features.py` — causal market features, train-only scaler, per-anchor packing (pure, tested).
- `code/run_master.py` — walk-forward driver (train MASTER/LSTM/VolGA + HAR/HAR-X, pool, metrics, DM,
  evidence, JSON). Thin `run()`/`main()`; logic in tested helpers.
- `test/test_master.py` — shape, gate slicing, causal, scaler fit-on-train, positive floored output,
  reproducibility, per-anchor independence of the batched attention.

## Comparison plan (DM date-clustered, favours A if mean_diff<0)
- MASTER vs HAR-X (deep dense-attention vs linear champion) — PRIMARY.
- MASTER vs VolGA (dense learned attention vs sparse horizon-matched edge) — PRIMARY.
- MASTER vs LSTM (dense attention vs no cross-sectional structure).
All models trained IN-RUN on identical folds so DM is exactly paired; the delivered
`results/edge_hmatched/edgehm_<market>_h<h>.json` LSTM/VolGA numbers (lb10 canonical) are cited in the
report as an external consistency check.
