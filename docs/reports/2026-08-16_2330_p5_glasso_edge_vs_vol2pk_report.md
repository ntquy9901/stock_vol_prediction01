# P5 — Graphical-LASSO edge vs vol→PK edge (DM)

Motivation (EDA plan §41; Zhang et al. 2308.01419): a raw correlation edge mostly re-encodes the
market factor already in the `market_pk` node feature, so the principled alternative is a
graphical-LASSO **partial-correlation** edge (conditional dependence, common factor removed). This
tests whether that market-factor-robust edge beats the directed volume→volatility (vol→PK) edge.

## Setup
- Edge: train-only graphical-LASSO precision → partial correlation → Top-5 per node, frozen
  (`edges_glasso.build_glasso_adjacency`). 165 off-diagonal edges (= 33×5, same density as vol→PK).
- Model/basis identical to the main QLIKE run except the adjacency: 5 node features, QLIKE estimation,
  2-hop GAT, FULL config (news+gate+graph). 3 seeds {42,123,2026}, 15-epoch cap.
- Comparators reuse the main run's dumps (`2026-08-16_141447_gnnhar_qlike`): vol→PK FULL, HAR (P0),
  minus_graph (graph branch removed). DM = HLN, HAC lag h−1, seed-ensembled, QLIKE loss.
- Code: `run_glasso_edge.py`, `edges_glasso.py`. Runner: `run_glasso_edge.py <GTS> <VOL2PK_TS> 42,123,2026 15 1 5 10 22`.

## Result — DM of the graphical-LASSO edge vs each comparator (negative dm favours glasso)

| h | vs vol→PK edge | vs HAR | vs no-graph (minus_graph) |
|---|---|---|---|
| 1 | +4.98, p<0.001 (vol→PK) | +0.24, p=0.81 (tie) | +3.76, p<0.001 (no-graph) |
| 5 | −2.33, p=0.020 (glasso) | −1.98, p=0.048 (glasso) | −2.01, p=0.044 (glasso) |
| 10 | +3.04, p=0.002 (vol→PK) | +3.45, p=0.001 (HAR) | +3.63, p<0.001 (no-graph) |
| 22 | −1.37, p=0.171 (tie) | +2.92, p=0.003 (HAR) | +0.01, p=0.99 (tie) |

## Reading

- **The two edges are horizon-complementary, not one-dominates-the-other.** The directed vol→PK edge
  is better at h1 and h10 (p<0.01); the graphical-LASSO edge is better at h5 (p=0.020); they are
  indistinguishable at h22.
- **h5 is the standout: the graphical-LASSO edge lowers QLIKE relative to vol→PK, HAR, AND the
  graph-removed model, all at p<0.05.** This is the first configuration in the project where a graph
  edge shows a significant out-of-sample gain over both HAR and no-graph — i.e. the conditional-
  dependence edge adds value the node features alone do not, at the one-week horizon.
- Everywhere else a graph edge does not beat HAR: at h1 the glasso edge is worse than no-graph (it
  hurts), at h10 HAR wins, at h22 HAR wins and the glasso edge ties no-graph.
- Consistent with the project theme and the EDA "graph adds little OOS value" conclusion, except for
  the h5 graphical-LASSO result.

## Caveats
- The h5 win over HAR is marginal (p=0.048) and on 3 seeds with a 15-epoch cap; it should be
  confirmed with more seeds / full convergence before a strong claim.
- The glasso edge here uses a fixed regularisation path (alpha auto-raised until convergence) and
  Top-5 selection to match vol→PK density; alpha/Top-K were not tuned by cross-validation.

## Conclusion
The graphical-LASSO partial-correlation edge is a legitimate, market-factor-robust alternative and is
the best edge at h5 (where it beats HAR and no-graph), but it does not uniformly beat the vol→PK edge
(vol→PK wins h1/h10). The practical takeaway: any single fixed edge is horizon-specific; the only
significant graph-adds-value result is graphical-LASSO at the one-week horizon.
