# Consolidated metrics — Track A + Track B (as of 2026-08-08)

All numbers are validation-split metrics re-read from the actual `results/*/results.json` files (Track B)
and `docs/paper/architecture_diagrams_review.md` §6 (Track A). Corrected per-ticker DirAcc convention.

## Eval-basis caveat (read first)
The three clusters below were evaluated on DIFFERENT held-out sets and therefore are NOT comparable in
absolute terms across clusters — only WITHIN a cluster:
- Track A: common-date panel eval, 20-epoch, n=3 seeds.
- Track B pooled: per-ticker pooled val/test eval, 5-epoch screening, n=3 seeds.
- Track B graph (G0/G1): common-date graph eval, batched, seed 42 only.
The absolute RMSE gap between clusters (~0.0027 vs ~0.0015 vs ~0.0024) reflects the eval set, not model
quality. Bold marks the best value within each cluster/metric.

## Cluster 1 — Track A (paper v3): common-date panel, 20-epoch, n=3
| Model | RMSE ↓ | R² ↑ | QLIKE ↓ | DirAcc % |
|---|---|---|---|---|
| HAR (classical) | **0.002182** | 0.7419 | 0.5493 | 48.65 |
| Price-only backbone | 0.002923 | 0.7749 | 0.4603 | 48.47 |
| FULL (news+graph+gate) | 0.002734 | 0.8031 | 0.4430 | 47.77 |
| No-Graph ablation | 0.002788 | 0.7953 | 0.4657 | 48.29 |
| No-Gate ablation | 0.002723 | **0.8047** | **0.4366** | 48.22 |

News improves QLIKE/R² (significant vs backbone); HAR wins RMSE/MAE; graph and gate null.

## Cluster 2 — Track B pooled: 5-epoch screening, n=3
| Config | RMSE ↓ | R² ↑ | QLIKE ↓ | DirAcc % |
|---|---|---|---|---|
| pooled P0 (HAR) | **0.001485 ± 0.000000** | 0.7351 | 0.5167 | 48.54 |
| pooled P1 (price) | 0.001502 ± 0.000003 | 0.7287 | 0.5118 | 48.59 |
| pooled P2 (+news) | 0.001487 ± 0.000003 | 0.7344 | **0.5084** | 48.47 |
| pooled P3 (+gate) | 0.001489 ± 0.000003 | 0.7336 | 0.5086 | 48.58 |
| common-date P0 | 0.001491 ± 0.000000 | 0.7329 | 0.5147 | 48.48 |
| common-date P1 | 0.001493 ± 0.000003 | 0.7320 | 0.5128 | 48.79 |
| common-date P2 | 0.001503 ± 0.000001 | 0.7283 | 0.5178 | 48.36 |
| common-date P3 | 0.001507 ± 0.000006 | 0.7270 | 0.5163 | 48.62 |

pooled ≈ common-date (A1 negative); P2 (news) best QLIKE; P0 (HAR) best RMSE; P3 (gate) adds nothing.
Source: `results/a1_{pooled,commondate}_seed{42,123,2026}/h5/{P0,P1,P2,P3}/results.json`.

## Cluster 3 — Track B graph (batched, seed 42, common-date graph eval)
| Model | RMSE ↓ | R² ↑ | QLIKE ↓ | DirAcc % | nonpos |
|---|---|---|---|---|---|
| G0 (message-passing off) | **0.002407** | **0.7447** | **0.6876** | 48.61 | 0.0% |
| G1 (message-passing on) | 0.002410 | 0.7440 | 0.6963 | 48.52 | 0.0% |

G1 ≈ G0 — the cross-stock graph does not help; positivity safety gate holds (0% nonpositive).
Source: `results/pooled_news_gnn_g0g1_batched_2026-08-08_171457_seed42/h5/{G0,G1}/results.json`.

## Statistical verdicts (paired-t, df=2, t_crit = 4.303)
| Effect | Test | Result |
|---|---|---|
| News | P2 vs P1 (pooled) | RMSE t=−4.81, QLIKE t=−6.94 — 3/3 seeds, SIGNIFICANT |
| Gate | P3 vs P2 (pooled) | 0/3 improve — inert (null) |
| Graph | G1 vs G0 (batched) | Δval-loss +0.00198 — null |
| Pooling | pooled vs common-date (A1) | <1%, mixed sign — null |
| Direction | all models | DirAcc ~48.5%, anti-persistence ceiling, not discriminating |
| P1 vs HAR | pooled | RMSE t=+7.5 (P1 worse); QLIKE t=−7.7 (P1 better) |
| P2 vs HAR | pooled | RMSE t=+1.1 (tie); QLIKE t=−59.5 (news wins decisively) |

## Summary
Across both architectures, only the NEWS features carry a statistically robust contribution (improve
QLIKE; recover the RMSE the price-only deep model loses vs HAR to a tie). Graph, gate, and the pooled
data regime are each null. Classical HAR remains hard to beat on RMSE/R². Direction is near-random
everywhere. Caveats: Track B is 5-epoch/3-seed/horizon-5 screening (df=2 low power); G0/G1 single-seed.
