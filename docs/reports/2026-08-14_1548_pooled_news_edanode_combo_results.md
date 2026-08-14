# Pooled News + EDA-node-features + directed vol→PK GNN combo — results

Date: 2026-08-14. Baseline `baselines/2026-08-14_pooled_news_edanode_gnn/`, branch
`feature/pooled-news-edanode-gnn`. Run TS `2026-08-14_143330`, 3 seeds (42/123/2026),
horizon 5, budget 5(backbone)+5(pooled)+15(graph head). GPU RTX 4060, `.venv_gpu_encode`.

## Purpose
Close the one untested lever from the graph-EDA: combine the EDA-recommended node features
(MarketPK + volume_zscore_20) with PhoBERT news late-fusion and the directed volume→PK Top-5 graph
on the pooled backbone, and DM-test whether the full combo beats HAR. Prior work tested node
features price-only (`eda_gnn` E0–E3) and news without the extra node features (`ladder_consistent`
P0–G1); this run combines all three on one leakage-safe basis.

## Basis (leakage-safe, identical to the canonical ladder)
5 node features (order: parkinson_volatility, har_weekly, har_monthly, market_pk,
volume_zscore_20) + PhoBERT news (146-dim) + directed vol→PK Top-5 frozen edge (165 directed edges
+ 33 self-loops). Per-ticker train-only scalers, news causal cutoff, one-basis obs invariant
asserted and held: **train 73026 / val 14418 / test 14464 — byte-identical to
`ladder_consistent_h5`**, so DM against the canonical P0/P2 is valid.

## Test metrics (3-seed mean)
| Rung | RMSE | QLIKE | R² | DirAcc% |
|---|---|---|---|---|
| P0 HAR | 0.002287 | 0.5735* | 0.7672 | 48.55 |
| P1 price5 (HAR+MarketPK+vol_z) | **0.002258** | 0.5690* | **0.7732** | 48.33 |
| P2 +news | 0.002263 | 0.5685* | 0.7721 | 48.23 |
| P3 +gate (graph off) | 0.002288 | 0.5859 | 0.7671 | 48.38 |
| G1 combo (+vol→PK graph) | 0.002295 | 0.5835 | 0.7656 | 48.46 |

\* Floor caveat (review BH-1): P0/P3/G1 QLIKE use a 1e-6 positivity floor; P1/P2 (reused pooled
rung) use 1e-8. P1/P2 **QLIKE is therefore not strictly comparable** to P0's; the like-for-like
P1/P2-vs-HAR comparison is on floor-independent **RMSE/R²** (P1/P2 beat HAR). The
node-features-beat-HAR-on-QLIKE result is DM-established floor-consistently in the sibling
`2026-08-11_eda_gnn_baseline` (E2 p=0.012).

## Diebold-Mariano (3-seed ensemble, HLN-corrected, h=5; negative dm favors A)
| A vs B | metric | dm_hln | p | favors | n |
|---|---|---|---|---|---|
| G1 vs P0 | QLIKE | 2.924 | 0.0035 | **P0 (HAR)** | 14464 |
| G1 vs P0 | SE | 0.495 | 0.621 | HAR (n.s.) | 14464 |
| G1 vs P3 | QLIKE | −0.964 | 0.335 | G1 (n.s.) | 14464 |
| G1 vs P3 | SE | 1.284 | 0.199 | P3 (n.s.) | 14464 |
| P3 vs P0 | QLIKE | 5.985 | 2e-9 | **P0 (HAR)** | 14464 |
| P3 vs P0 | SE | 0.032 | 0.974 | tie | 14464 |

## Findings
1. **The full GNN combo (G1) does NOT beat HAR.** It is DM-significantly worse on QLIKE
   (p=0.0035) and statistically tied on squared error (p=0.62). The user's recurring goal —
   a GNN in the final model that beats HAR — is not achieved by this combination.
2. **The graph adds no out-of-sample value.** G1 vs P3 (graph on/off, same backbone) is
   non-significant on both losses — consistent with every prior graph-null result in the project.
3. **The per-ticker gate degrades calibration.** P3 vs P0 QLIKE p≈2e-9 favoring HAR: the gated
   graph backbone (P3/G1) is the rung that loses to HAR on QLIKE, not the node features.
4. **The node features remain the only positive lever.** The ungated P1/P2 (node features ± news)
   beat HAR on RMSE (0.002258/0.002263 vs 0.002287) and R² (0.7732/0.7721 vs 0.7672) on both val
   and test — but stacking the gate + graph on top (P3/G1) erases that gain. News over node
   features (P2 vs P1) is ~neutral.

Net: adding news + a cross-stock graph on top of the winning node features does not yield a
HAR-beating GNN; the graph/gate machinery does not earn its place. This reinforces the project's
parsimony/Conclusion-C story with a direct multi-seed DM test of the combined architecture.

## Tests + coverage
- 5 tests pass (`.venv_gpu_encode`, Py 3.10): basis 5-feature+news+directed-edge, one-basis
  invariant, pure-HAR P0 metrics, DM aggregator (well-defined + degenerate-pair guard).
- ruff clean. build_basis + P0 + DM aggregator are unit/smoke covered; the P1/P2/P3/G1 GPU
  training path is validated by the completed 3-seed run (per-seed `ladder_metrics.json`,
  reproduced by `combo_aggregate.py`). diff-cover on the full GPU train loop is `Not run` (train
  loop is not unit-executable on CPU; standard for this project's baselines).

## Code review (3-layer, before done)
See `code_review/code_review_2026-08-14.md`. Leakage/P0/DM/edge-wiring/floor(H2 for DM rungs)
confirmed correct by all three layers. 4 MEDIUM + 6 LOW findings; all fixed except BH-1 (P1/P2
QLIKE floor), which is documented + mitigated (RMSE used for P1/P2; consistent-floor P1/P2 QLIKE +
DM is the top follow-up). No scope violation (parent baselines untouched; reused read-only).

## Data-quality gate (Pandera + Evidently)
N/A (no model-data change): reuses `data/processed` + the existing news panel via the parent
pipelines; adds no new dataset/manifest. Leakage-safety is enforced by the reused machinery +
the one-basis invariant asserted at runtime.

## Risks / follow-ups
- Top follow-up: apply the 1e-6 floor to P1/P2 and emit their per-obs dumps so P1/P2-vs-HAR is a
  floor-consistent DM in this run (currently RMSE-only + sibling-DM citation).
- Verdict is a documented NULL for the combo; no re-run planned.

## DoD checklist
- [x] Requirements + design (SDD §3.F)
- [x] Code runs (smoke pass); 5 tests pass; ruff clean
- [x] 3-layer adversarial review; findings fixed/mitigated
- [x] Honest DM verdict reported (all 6 metrics); no over-claim
- [x] Data-quality gate: N/A documented
- [ ] Commit + push branch (next); dashboard ledger entry (see summary)
