# Beat-HAR Solution Sweep — Requirements (spec)

Date started: 2026-08-10. Source plan: `docs/reports/2026-08-10_0033_beat_har_solution_plan.md`.

## Objective
Try to beat the classical HAR baseline for VN30 daily 5-day-ahead Parkinson-variance forecasting on
the consistent Track-B fair basis, by training pooled-LSTM + graph configurations C1..C7 (20 epochs)
whose common lever is a differentiable QLIKE training loss plus per-config graph/feature variations.

## Fair basis (frozen — identical to the consistent ladder)
- Masked kNN-8 mutual-correlation graph manifest, leakage-safe graph-bound train window
  (`target_date <= graph.train_end_date`), train-only per-ticker scalers (`graph_store`),
  temporal 70/15/15, Parkinson-**variance** target `shift(-5)`, positivity floor, present-node masking.
- Identical held-out val/test observations as `ladder_consistent_h5_2026-08-09_154402.json`
  (n_val = 14418, n_test = 14464, 33 tickers). Backbone (frozen P3 encoder + base cache) is the SAME
  MSE-trained backbone as the fair ladder; each config retrains only the graph-stage head +
  message-passing (which sets the prediction level) under its loss/adjacency/feature variation. This
  isolates the graph-stage lever on an identical basis and reuses the base cache. Design decision
  recorded in `design/design.md`.

## The bar to beat (TEST set, from the plan / cited JSONs)
- P0 pooled-HAR anchor QLIKE = 0.5676 (effective QLIKE bar).
- HARQ RMSE = 0.0022891, R² = 0.76682 (RMSE/R² wall).
- Classical per-ticker HAR QLIKE = 0.5793.

## Success criteria (per config)
- Partial win (primary): test QLIKE < P0 0.5676, Diebold–Mariano p<0.05 vs P0, consistent sign across
  3 seeds (42/123/2026), paired-t on seed means. QLIKE-only DM-significant partial win COUNTS.
- Full win (stretch): also beat HARQ on RMSE and R², all DM-significant.
- Honest null is a valid outcome: report documented null if no config clears the bar.

## Configs
- C1 QLIKE-loss GAT+news (knn-8, monolithic). Delivers shared differentiable-QLIKE loss.
- C2 HAR + graph-residual additive (ŷ = ŷ_HAR + g(graph), floors at HAR).
- C3 directed Diebold–Yilmaz spillover edges (train-only VAR, frozen) + news + QLIKE.
- C4 HAR-RV-X range/overnight node features (GK/RS/overnight, variance σ² units) + news + QLIKE.
- C5 spillover + omit self-loops + k-sweep {4,8,12,16} + QLIKE (directed top-k, isolated-node fallback).
- C6 learned/dynamic adjacency (MTGNN-style, input-independent embeddings) + news + QLIKE.
- C7 news-as-EDGE co-mention — FEASIBILITY-GATED (per-ticker panel has no multi-ticker article
  structure; if raw corpus lacks per-article ticker tags → mark INFEASIBLE and skip).

## Leakage invariants (mandatory, every config)
Graph structure (spillover VAR, learned A) estimated on TRAIN window only and frozen; train-only
scalers; temporal split; positivity floor on denormalized predictions; present-node masking. A
leakage-driven win is disqualified.

## Go / No-go
- Go: ≥5 configs trained + evaluated on the fair basis with all 6 metrics (val+test) and DM/paired-t
  vs HAR recorded; report states per-config whether it beats/ties/loses HAR per metric with
  significance; honest verdict (win or documented null).
- No-go: any config's "win" that is not DM/paired-t supported on held-out data, or that relies on a
  broken leakage invariant.
