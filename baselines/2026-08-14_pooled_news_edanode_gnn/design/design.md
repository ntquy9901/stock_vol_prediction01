# Design — Pooled News + EDA-node-features + directed vol→PK GNN combo

## Approach: reuse two existing pipelines, swap only the basis
Grounded in a full read of both pipelines (see `docs/reports` merge map). The combo is a thin
orchestrator `code/combo_ladder.py` that reuses:
- **Pipeline B** `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/ladder_consistent.py`:
  the P1/P2 rungs (`run_pooled_rung`), the P3+G1 graph rungs (`run_graph_rungs`), the DM basis,
  news attach, and the one-basis invariant.
- **Pipeline A** `baselines/2026-08-11_eda_gnn_baseline/code/`: `features.augment_split_frames` +
  `build_extended_store` (5 node features, per-ticker train boundary), `edges.build_vol2pk_adjacency`
  + `swap_adjacency` (directed vol→PK Top-5), and `eda_ladder.run_e0` (pure-HAR P0).

### Data flow (build_basis)
```
load_and_split_price_data(data/processed)            # raw SplitFrames, 33 tickers, 70/15/15
  -> augment_split_frames(raw, data/raw/prices)      # + market_pk, volume_zscore_20 columns
  -> build_extended_store(augmented, EXTRA_COLS)     # per-ticker scalers, TRAIN-only (fixes H1)
  -> build_pooled_manifest(augmented, store, 22, 5)  # 5-feature price samples, order:
                                                     #  (pk_daily,har_weekly,har_monthly,market_pk,vol_z)
  -> attach_news(panel, cutoffs)                     # PhoBERT late-fusion, per-ticker causal cutoff
  -> build_masked_graph_manifest(knn-8)              # corr graph (scaffold)
  -> swap_adjacency(build_vol2pk_adjacency(top_k=5)) # replace edges with directed vol→PK, train-only
  -> allowed = train samples with target_date <= graph.train_end_date
  -> assert one-basis invariant (graph obs == pooled obs, val & test)
```

### Rung wiring (run_seed)
| Rung | Source fn | Features | News | Gate | Graph |
|---|---|---|---|---|---|
| P0 | A `run_e0` (first-3-col HAR slice) | HAR(3) | – | – | – |
| P1 | B `run_pooled_rung("P1")` | 5 | – | – | – |
| P2 | B `run_pooled_rung("P2")` | 5 | ✓ | off | – |
| P3 | B `run_graph_rungs` (graph-off readout) | 5 | ✓ | ✓ | off |
| G1 | B `run_graph_rungs` (graph-on) | 5 | ✓ | ✓ | vol→PK Top-5 |

**Why A's `run_e0` for P0, not B's `run_har_rung`:** B's `run_har_reference` slices
`x_price_raw[-1, -3:]` (LAST 3 cols). With the 5-feature order that is
`(har_monthly, market_pk, vol_z)` — not HAR. A's `run_e0` slices the FIRST 3 cols = true HAR, so
P0 stays the pure-HAR external baseline identical to the canonical ladder.

## Key design decisions
1. **Extend B, not A** (Anti-Abstraction gate): B already implements price+news+gate+GAT
   message-passing+positivity over an arbitrary adjacency, so no model code is written — only the
   basis is swapped. A's two feature/edge functions are already B-compatible (they import B's data
   structures).
2. **Message passing = B's GraphAblationModel GAT over the vol→PK adjacency** (not A's
   `_ResidualMessagePassing`). Deliberate: reuse the news-capable model. The vol→PK adjacency is
   directed with self-loops; it is fed to the GAT as a weighted matrix. This is a different
   mechanism than eda_gnn E3, documented as such — the combo is a new model, not E3+news.
3. **Training budget = 5 (backbone) + 5 (pooled P1/P2) + 15 (graph head), 3 seeds** — identical to
   canonical `ladder_consistent_h5` so DM vs the existing P0/P2/G1 numbers is valid (user-approved).

## Leakage-safety (must hold; asserted)
- Per-ticker train boundary for scalers (A's `build_extended_store`) — fixes prior H1.
- vol→PK edge estimated on each ticker's OWN train rows and frozen (`build_vol2pk_adjacency`).
- News causal per-ticker cutoff (`_train_news_cutoffs` + provenance-gated panel).
- Identical positivity floor on ALL compared rungs: graph rungs floor in-model
  (`POSITIVITY_EPSILON=1e-6`); P0 floored via A's `_floor_norm_records` (fixes prior H2). QLIKE DM
  floor `1e-8` identical across compared models.
- One-basis obs invariant asserted for val and test; abort if violated.

## Simplicity / Anti-Abstraction gates
- Simplicity: one new orchestrator file + tests; no new model, no config system, no new abstractions.
- Anti-Abstraction: direct reuse of both baselines' functions via `sys.path` bootstrap (folder name
  has dashes → not `python -m`-importable; scripts self-bootstrap per §3.F rule 4).

## Files
- `code/combo_ladder.py` — orchestrator (build_basis + run_seed + main).
- `test/test_combo_ladder.py` — leakage/basis unit tests + tiny-slice smoke.
- Results → `results/pooled_news_edanode_seed{seed}_<TS>/` (per §3.D, not in baseline folder).
- Aggregation reuses B's `docs/reports/ladder_consistent_dump.py` DM machinery (or a small
  combo aggregate) → DM table in the summary report.
