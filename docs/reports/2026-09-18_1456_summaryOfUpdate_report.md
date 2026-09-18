# Summary of update — Literal GBME + leaf-graph (Option A, HGBR leaves) — NO-GO on HOSE

Date: 2026-09-18. New baseline `baselines/2026-09-18_gbme_leafgraph_literal`. Tests the LITERAL "GBME + leaf-graph":
build the leaf-cooccurrence graph from the champion GBME's OWN trees (sklearn HistGradientBoostingRegressor) and
smooth GBME's OWN predictions — no XGBoost proxy. Answers whether the XGB+leafgraph HOSE gain (a committed GO)
transfers to the actual champion.

## HGBR leaf extraction (the new, verified capability)
HGBR has no public `.apply()`. Leaf indices are extracted from `model._predictors[i][0].nodes` by traversing
`num_threshold` directly on X (raw path — avoids `_bin_mapper`). A **reconstruction guard** asserts
`baseline + Σ leaf_value ≈ model._raw_predict(X)` each fold (max|Δ| ~1e-15) so the private-API extraction is
provably correct and fails loud on any sklearn drift. Module `code/hgbr_leaf.py`, unit-tested vs `_raw_predict`.

## Result (HOSE, 8 folds × 3 seeds, val-fit α, date-clustered DM)
| h | GBME | GBME+leafgraph | gain | DM p | α_mean | verdict |
|---|---|---|---|---|---|---|
| h1 | 1.5719 | 1.5723 | −0.028% | 0.033 | 0.20 | worse (DM-sig) |
| h5 | 1.6479 | 1.6483 | −0.024% | 0.025 | 0.31 | worse (DM-sig) |
| h10 | 1.6856 | 1.6853 | +0.016% | 0.353 | 0.43 | ns |
| h22 | 1.7270 | 1.7266 | +0.028% | 0.214 | 0.60 | ns |

**Verdict: NO-GO.** Smoothing GBME's own predictions over GBME's own leaf-graph does NOT help — it is
slightly worse at h1/h5 (DM-sig) and null at h10/h22.

## Interpretation (important for the paper)
The leaf-graph benefit is **XGB-specific, not a property of the champion**. On HOSE the committed
`XGB+leafgraph` beats its XGB base by +0.12–0.26% (DM-sig, spike-robust), yet the SAME graph idea applied to
GBME's own predictions is inert/slightly-negative. The plausible reason: XGBoost gamma's base per-stock
predictions are slightly noisier than HGBR's, so the cross-sectional leaf-neighbour denoising has something to
remove for XGB but not for the already-smoother HGBR/GBME. Consistent with the thin-market-denoising reading and
with the SP500 result (α→0 when the base is low-noise). So the deployable positive lever is **XGB+leafgraph**,
not GBME+leafgraph.

## Decision
Per the finding, GBME+leafgraph is de-prioritized (does not help HOSE; SP500 predictably worse and not completed).
The literal-integration capability (HGBR leaf extraction + reconstruction guard) is preserved for reference. Focus
stays on XGB+leafgraph (the working lever) + its no-earn ablation.

## Tests / gate
24 tests pass; diff-coverage C0=100% / C1=100% on all 5 code modules. Result JSONs pass the over/under-fit gate.
Adversarial code review recorded (`code_review/code_review_2026-09-18.md`). SP500 run was stopped intentionally
(GBME de-prioritized) — only HOSE (4 horizons, 8-fold) is committed.
