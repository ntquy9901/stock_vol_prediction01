# TrackAGatModel — Detailed Architecture (Quick Review)

Track-A-style parallel model: LSTM (temporal) + self-written multi-head GAT (cross-sectional,
`vol->PK` directed edge) + news branch with per-ticker gate, concatenated at the head.
Verified against `code/model.py`, `code/gat.py`, `code/run_ablation.py`.

Legend: `B`=batch, `N`=tickers per snapshot, `seq`=sequence length, `price_dim=5`, `news_dim=146`,
`hidden=64`, `heads=4`, `gnn_dim = hidden*heads = 256`.

---

## 1. Top-level block diagram (3 parallel branches)

```
INPUTS (one target date's cross-stock panel)
  price     [B, N, seq, 5]      (5 node features; see §4)
  news      [B, N, seq, 146]    (PhoBERT-derived news manifest)
  news_mask [B, N, seq]
  ticker_ids[B, N]
  adjacency [B, N, N]           (directed vol->PK Top-5, self-loops, frozen TRAIN-only)

 ┌───────────────────────── BRANCH 1: PRICE-LSTM (temporal, per node) ─────────────────────────┐
 │ price [B,N,seq,5]                                                                            │
 │   -> reshape [B*N, seq, 5]                                                                   │
 │   -> LSTM(in=5, hidden=64, layers=2, dropout=0.2)   -> take last step                       │
 │   -> h_lstm  [B, N, 64]                                                                      │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────── BRANCH 2: GAT graph (cross-sectional, RAW feats @ t) ───────────────────┐
 │ node_raw = price[:, :, -1, :]        # LAST timestep RAW features, NOT h_lstm                │
 │            [B, N, 5]                                                                          │
 │   -> GATLayer gat1 (5 -> 64, heads=4; concat)   [B, N, 256]                                  │
 │   -> GATLayer gat2 (256 -> 64, heads=4; concat) [B, N, 256] = h_gnn                          │
 │      attention masked by (adjacency>0), softmax over SOURCE nodes, LeakyReLU + ELU           │
 │      graph OFF  => adjacency := identity (self-loops only)                                   │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────── BRANCH 3: NEWS (per node) + per-ticker GATE ────────────────────────┐
 │ news * news_mask  [B,N,seq,146]                                                              │
 │   -> Linear(146 -> 64) + ReLU                                                                │
 │   -> LSTM(64 -> 64, layers=2, dropout=0.2) -> last step -> news_hidden [B,N,64]              │
 │   -> gate = sigmoid(gate_logits[ticker_ids])         # scalar per ticker, in (0,1)           │
 │   -> gated_news = gate * news_hidden   [B, N, 64]                                            │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘

                    h_lstm[64]        h_gnn[256]        gated_news[64]
                        └──────────────────┼──────────────────┘
                                           v
                       HEAD:  concat -> h  [B, N, 384]
                              Linear(384 -> 64) + ReLU + Dropout(0.2)
                              Linear(64 -> 1)              -> raw  [B, N]
                                           v
                    POSITIVITY FLOOR (denormalized space, eps=1e-6)
                       denorm  = raw*std + mean                 (per-ticker StandardScaler)
                       floored = eps*softplus(denorm/eps) + eps (>= 0; target is Parkinson VARIANCE)
                       out     = (floored - mean) / std         [B, N]  (normalized space)
```

Notes:
- Branch 2 consumes RAW node features at the last timestep, not the LSTM hidden state. This keeps
  the graph branch an independent cross-sectional view (fair leave-one-out graph ablation) and
  matches the `vol->PK` edge semantics (`volume_shock_i(t) -> sqrt(PK_j)(t+1)`).
- Without news (`use_news=False`), `gated_news` is zeros `[B,N,64]`; the gate is a no-op.

---

## 2. Directed `vol->PK` Top-5 edge

Adjacency built by lead-lag correlation, directed and asymmetric, frozen on TRAIN only.

```
For each TARGET ticker j, connect the Top-5 SOURCE tickers i ranked by:
    corr( volume_shock_i(t) ,  sqrt(PK_j)(t+1) )        # source volume LEADS target volatility

        volume_shock          sqrt(PK) next day
   [i=SRC] ───────────────────────────────────> [j=DST]
    (leads)        directed, asymmetric          (lags)

   adjacency[i, j] = 1  for i in Top5(j);   diagonal self-loops kept (adjacency[j, j] = 1)

Attention (per GAT layer): mask entries where adjacency<=0, softmax over SOURCE j, aggregate.
```

Chosen over undirected k-NN correlation-on-PK because:
- (a) lead-lag is predictive/causal, not contemporaneous
- (b) volume leads volatility
- (c) directed captures who-leads-whom
- (d) EDA-selected Top-5 (not an arbitrary k)

Leakage safety: edges are computed on TRAIN data only and frozen for val/test.

---

## 3. Leave-one-out ablation (`run_ablation.py`, h in {1, 5, 10, 22})

Build FULL, then RETRAIN each variant from scratch with exactly ONE component removed, so every
effect is measured on the same footing (each variant trains in the same graph on/off regime it is
evaluated in — no train/eval mismatch).

```
                         FULL (LSTM + vol->PK GAT graph + news + per-ticker gate)
                          |
        ┌─────────────────┼──────────────────┬──────────────────────┐
        v                 v                  v                        v
   minus_graph        minus_gate         minus_news              (external)
   DROP whole GAT     drop per-ticker    drop news branch          HAR
   branch (no         gate               (gate no-op w/o news)   pooled linear reg
   node/edge/GAT)     retrained          retrained                 fit
   retrained
```

| Variant       | LSTM | Graph (vol->PK) | News | Gate | Trained            |
|---------------|------|-----------------|------|------|--------------------|
| HAR           |  -   |       -         |  -   |  -   | pooled linear (ext)|
| FULL          | yes  |      yes        | yes  | yes  | retrained          |
| minus_graph   | yes  | NONE (no GAT)   | yes  | yes  | retrained          |
| minus_gate    | yes  |      yes        | yes  | OFF  | retrained          |
| minus_news    | yes  |      yes        | OFF  | n/a  | retrained          |

Contribution of component X (test QLIKE):

```
effect(X) = QLIKE(FULL) - QLIKE(minus_X)
    negative  => removing X hurt (QLIKE rose)  => X HELPED
    positive  => removing X improved QLIKE     => X did not help
```

Significance: Diebold-Mariano (HLN correction) on seed-ensembled test predictions, for
FULL vs each minus_X and FULL vs HAR.
```
