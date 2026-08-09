# Consistent-basis Track-B ladder P0 -> P1 -> P2 -> P3 -> G1 (h10)

One basis for every rung: masked manifest, leakage-safe graph-bound train, shared per-ticker scalers, positivity floor, identical val/test observations. Masked k-NN-8 adjacency, seeds [42, 123, 2026], horizon 10. P3 = the trained G1 model read out with the GAT/message-passing residual disabled (replaces the old separate G0 row).

## Nesting verification (G1 minus the GAT = P3)

The P3 row is the identical trained G1 model with the message-passing residual removed, so the equality holds by construction; the runtime check confirms the graph-off readout is deterministic (max abs pred diff on a re-evaluation) and the residual is non-trivial (mean abs raw pred diff G1 vs P3).

| seed | graph-off determinism (max abs diff) | graph effect (mean abs raw diff) | n_val_obs |
|---|---|---|---|
| 42 | 0.00e+00 | 3.480e-05 | 14253 |
| 123 | 0.00e+00 | 2.950e-05 | 14253 |
| 2026 | 0.00e+00 | 3.613e-05 | 14253 |

## VAL metrics (3-seed mean +/- std)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 2.33764e-06 +/- 0 | 0.00152893 +/- 2.65574e-19 | 0.00049489 +/- 0 | 0.720288 +/- 0 | 0.547378 +/- 0 | 50.00 +/- 0.00 |
| P1 | 2.4164e-06 +/- 6.25476e-09 | 0.00155448 +/- 2.01181e-06 | 0.000509788 +/- 1.6985e-06 | 0.710864 +/- 0.000748418 | 0.554232 +/- 0.000629788 | 50.45 +/- 0.03 |
| P2 | 2.43387e-06 +/- 1.16172e-08 | 0.00156008 +/- 3.72371e-06 | 0.000512833 +/- 1.46691e-06 | 0.708773 +/- 0.00139006 | 0.556383 +/- 0.00191748 | 50.54 +/- 0.11 |
| P3 | 2.31654e-06 +/- 1.37423e-08 | 0.00152201 +/- 4.51329e-06 | 0.00048964 +/- 2.99033e-06 | 0.722812 +/- 0.00164435 | 0.554665 +/- 0.00136865 | 50.56 +/- 0.16 |
| G1 | 2.27004e-06 +/- 2.04788e-08 | 0.00150666 +/- 6.8007e-06 | 0.00048627 +/- 4.30499e-06 | 0.728376 +/- 0.0024504 | 0.548133 +/- 0.003751 | 50.00 +/- 0.09 |

## TEST metrics (3-seed mean +/- std)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 5.58056e-06 +/- 0 | 0.00236232 +/- 0 | 0.000626701 +/- 0 | 0.752343 +/- 0 | 0.606054 +/- 0 | 48.75 +/- 0.00 |
| P1 | 5.51546e-06 +/- 2.23897e-08 | 0.0023485 +/- 4.76955e-06 | 0.000631609 +/- 7.56135e-07 | 0.755232 +/- 0.000993619 | 0.612317 +/- 0.00103993 | 49.10 +/- 0.08 |
| P2 | 5.50704e-06 +/- 9.69296e-09 | 0.00234671 +/- 2.06561e-06 | 0.000633176 +/- 8.64559e-07 | 0.755606 +/- 0.000430159 | 0.614645 +/- 0.00216881 | 48.95 +/- 0.18 |
| P3 | 5.72058e-06 +/- 1.02179e-08 | 0.00239177 +/- 2.13552e-06 | 0.000626076 +/- 1.43966e-06 | 0.746129 +/- 0.000453454 | 0.616804 +/- 0.00110055 | 49.03 +/- 0.06 |
| G1 | 5.67168e-06 +/- 2.14235e-08 | 0.00238153 +/- 4.49558e-06 | 0.000626558 +/- 2.40375e-06 | 0.748299 +/- 0.000950743 | 0.615622 +/- 0.00141133 | 49.10 +/- 0.20 |

## Graph effect (G1 vs P3), VAL

**Verdict: B** (G1 QLIKE < P3 in 3/3 seeds; QLIKE delta mean=-6.532e-03; paired-t p=0.0447; DM-QLIKE all sig-neg=False)

| seed | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |
|---|---|---|---|---|---|
| 42 | -1.922 | 0.05466 | -2.208 | 0.02726 | 14253 |
| 123 | -3.660 | 0.0002534 | -2.593 | 0.009511 | 14253 |
| 2026 | -4.606 | 4.146e-06 | -2.810 | 0.004955 | 14253 |

## Graph effect (G1 vs P3), TEST

**Verdict: B** (G1 QLIKE < P3 in 3/3 seeds; QLIKE delta mean=-1.182e-03; paired-t p=0.0669; DM-QLIKE all sig-neg=False)

| seed | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |
|---|---|---|---|---|---|
| 42 | -0.444 | 0.6572 | -1.522 | 0.1279 | 14299 |
| 123 | -1.082 | 0.2795 | -1.836 | 0.06641 | 14299 |
| 2026 | -0.533 | 0.5942 | -1.523 | 0.1277 | 14299 |
