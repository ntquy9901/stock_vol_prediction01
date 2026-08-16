# Beat-HAR research (autonomous, 2026-08-17)

Goal: find approaches where the deep/GNN model beats HAR out-of-sample (QLIKE, DM HLN, seed-ensembled,
leakage-safe). Accumulates the session's landscape + new experiments. Current data: VN30 33 tickers
through 2026-08-14, QLIKE estimation, 2-hop, main run TS `2026-08-16_141447_gnnhar_qlike`.

## Landscape so far (what does NOT robustly beat HAR)
- Single deep model (FULL) vs HAR: significant only at h1 (QLIKE p=0.003); HAR wins h10/h22.
- vs a FAIR same-feature linear baseline (HAR-X): LSTM nonlinearity beats HAR-X at h1 (p<0.001) and,
  at 5 seeds, h5 (p=0.03); the two extra node features add nothing to a linear model.
- Graph edge (vol→PK or graphical-LASSO): no robust OOS gain. The one apparent glasso win at h5
  (p=0.048, 3-seed) COLLAPSED at 5 seeds (p=0.286). Graph even hurts at h10 (p=0.001).
- News / gate branches: do not lower QLIKE at short horizons (removing them helps).

## WINNER — forecast combination (Bates–Granger): 0.5·HAR + 0.5·deep

A fixed 50/50 average of the HAR forecast and a deep forecast (raw variance scale) — leakage-safe
(no tuning) — beats HAR:

| h | HAR QLIKE | 0.5·HAR+0.5·lstm_only | DM vs HAR | seeds |
|---|---|---|---|---|
| 1 | 0.4633 | 0.4564 | p<0.001 (COMB) | 3 |
| 5 | 0.5503 | 0.5450 | p<0.001 (COMB) | 5 (robust) |
| 10 | 0.5933 | 0.5918 | p=0.208 (tie, slight COMB) | 5 |
| 22 | 0.6474 | 0.6526 | p<0.001 (HAR) | 3 |

- Holds for every deep variant (lstm_only / FULL / minus_news) at h1 and h5, all p<0.001 (3-seed).
- h5 confirmed at 5 seeds (unlike the graph-edge false positive).
- Weight sensitivity (QLIKE vs w_HAR; deep=lstm_only): the optimum is horizon-adaptive — more weight
  on the deep model at short horizons, all-HAR at h22:

  | w_HAR | 1.0 | 0.7 | 0.5 | 0.3 | 0.0 |
  |---|---|---|---|---|---|
  | h1 | 0.4633 | 0.4585 | 0.4564 | 0.4550 | 0.4547 |
  | h5 | 0.5503 | 0.5466 | 0.5450 | 0.5441 | 0.5442 |
  | h10 | 0.5933 | 0.5920 | 0.5918 | 0.5921 | 0.5938 |
  | h22 | 0.6474 | 0.6502 | 0.6526 | 0.6554 | 0.6602 |

**Interpretation.** HAR and the nonlinear model make partially uncorrelated errors, so averaging
lowers QLIKE at the horizons where the deep model is competitive (h1, h5) — the classic
forecast-combination result. This is the cleanest, most defensible way the deep model "beats HAR":
not by replacing it, but by combining with it. A fixed 0.5 already wins; a validation-fit horizon
weight (not tuned on test) would be the proper way to capture the horizon-adaptive optimum (needs
per-observation validation predictions — a follow-up).

## Caveats / honesty
- The combination win is at SHORT horizons (h1, h5). At h22 HAR alone is best (deep drags it down).
- The 0.5 weight is leakage-safe (fixed, untuned). The weight-grid minima are in-sample-on-test and
  are shown only as sensitivity, not as a tuned claim.
- QLIKE is the reported metric; MSE/RMSE/MAE/R² differences among all configs are <~1% (see the main
  results report).

## 3-feature-node + graphical-LASSO edge vs HAR-3 (user question)

A graph model whose NODES use only the 3 HAR features (no market_pk/volume_z), edge = graphical-LASSO,
no news/gate — head-to-head with HAR on the same 3 features. 3-seed mean test metrics at h1:

| model | MSE | RMSE | MAE | R² | QLIKE | DirAcc |
|---|---|---|---|---|---|---|
| HAR-3 | 1.96e-07 | 0.000443 | 0.000249 | 0.287 | 0.4633 | 32.5 |
| glasso3 | 2.01e-07 | 0.000448 | 0.000246 | 0.270 | 0.4614 | 34.2 |

glasso3 is marginally lower on QLIKE/MAE and higher on DirAcc, but worse on MSE/RMSE/R². The QLIKE DM
was p=0.046 (3-seed) — but at **5 seeds it collapses to p=0.192** (glasso3 QLIKE 0.4613 vs HAR 0.4633,
not significant). So a GNN over the same 3 features HAR uses does NOT reliably beat HAR — the apparent
3-seed edge is the same marginal-false-positive pattern seen for the glasso edge at h5. Confirms: the
graph adds no robust value; without the extra node features the deep model has no edge over HAR either.

## Conclusion
Across every single-model route (single deep model, 3- vs 5-feature nodes, vol→PK vs graphical-LASSO
edge, news/gate, GAT depth), the deep/GNN model does not RELIABLY beat HAR beyond a narrow h1
nonlinearity effect; all marginal graph wins collapse at 5 seeds. The one robust way the deep model
beats HAR is **forecast combination** (0.5·HAR + 0.5·deep), which lowers QLIKE significantly at h1 and
h5 (p<0.001, 5-seed robust). Recommended framing: the deep model's value is complementary to HAR, not
a replacement — combine them at short horizons; use HAR alone at h22.

## Next approaches to try (autonomous queue)
- Residual boosting: train the deep model on the HAR residual (target = PK − HAR_pred), final =
  HAR_pred + deep_residual. Principled "beat HAR by construction if the deep part adds signal".
- Validation-fit combination weight per horizon (leak-safe, needs val per-obs dumps).
- Log/variance-stabilising target for the deep branch.
- 3-feature-node + graphical-LASSO edge vs HAR-3 (isolates graph+nonlinearity from extra features) —
  `run_glasso_node3.py`, results pending.
