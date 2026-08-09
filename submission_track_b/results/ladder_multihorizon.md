# Multi-horizon Track-B ladder P0 -> P1 -> P2 -> P3 -> G1 (h1 / h5 / h10 / h22)

Consistent-basis nested ladder replicated at four forecast horizons (target = `volatility.shift(-h)`). One basis per horizon: masked k-NN-8 manifest, leakage-safe graph-bound train, shared per-ticker scalers, positivity floor, identical val/test observations across rungs, seeds 42/123/2026. P3 = the trained G1 read out with the GAT/message-passing residual disabled (exact nesting; graph-off determinism 0.0 all seeds, all horizons). Canonical per-horizon files: `ladder_consistent_h{1,10,22}_2026-08-09_180326.{json,md}` and `ladder_consistent_h5_2026-08-09_154402.{json,md}`.

## Basis size per horizon

| horizon | snapshots | val obs | test obs |
|---|---|---|---|
| 1 | 6482 | 14550 | 14596 |
| 5 | 6470 | 14418 | 14464 |
| 10 | 6452 | 14253 | 14299 |
| 22 | 6415 | 13857 | 13903 |

## Graph effect (G1 vs P3) across horizons

Diebold-Mariano at horizon h (HAC truncation lag h-1), HLN-corrected; QLIKE delta = mean(G1 - P3) over seeds (negative = graph helps). Verdict A = graph helps (G1 QLIKE < P3 all seeds AND DM-QLIKE significant-negative all seeds); B = null.

| horizon | split | G1 QLIKE<P3 seeds | QLIKE delta mean | paired-t p | DM-QLIKE all sig-neg | verdict |
|---|---|---|---|---|---|---|
| 1 | val | 2/3 | +4.343e-01 | 0.4317 | False | B |
| 1 | test | 2/3 | +3.088e-02 | 0.5305 | False | B |
| 5 | val | 3/3 | -3.899e-03 | 0.0096 | False | B |
| 5 | test | 2/3 | -5.620e-04 | 0.7913 | False | B |
| 10 | val | 3/3 | -6.532e-03 | 0.0447 | False | B |
| 10 | test | 3/3 | -1.182e-03 | 0.0669 | False | B |
| 22 | val | 3/3 | -6.170e-03 | 0.0002 | False | B |
| 22 | test | 0/3 | +4.871e-03 | 0.1425 | False | B |

## Per-seed Diebold-Mariano (TEST) across horizons

| horizon | seed | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |
|---|---|---|---|---|---|---|
| 1 | 42 | -5.168 | 2.399e-07 | -3.273 | 0.001067 | 14596 |
| 1 | 123 | -0.877 | 0.3804 | -3.417 | 0.0006348 | 14596 |
| 1 | 2026 | +1.600 | 0.1097 | -2.727 | 0.00639 | 14596 |
| 5 | 42 | +0.838 | 0.4021 | -1.856 | 0.06343 | 14464 |
| 5 | 123 | -1.732 | 0.08331 | -0.581 | 0.5613 | 14464 |
| 5 | 2026 | -1.856 | 0.06344 | -3.403 | 0.0006674 | 14464 |
| 10 | 42 | -0.444 | 0.6572 | -1.522 | 0.1279 | 14299 |
| 10 | 123 | -1.082 | 0.2795 | -1.836 | 0.06641 | 14299 |
| 10 | 2026 | -0.533 | 0.5942 | -1.523 | 0.1277 | 14299 |
| 22 | 42 | +0.414 | 0.6786 | +1.831 | 0.06716 | 13903 |
| 22 | 123 | +1.154 | 0.2486 | +1.284 | 0.1993 | 13903 |
| 22 | 2026 | +1.492 | 0.1358 | +1.096 | 0.2729 | 13903 |

## h1 VAL metrics (3-seed mean)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 1.84753e-06 | 0.00135924 | 0.000430766 | 0.777746 | 0.438032 | 31.89 |
| P1 | 1.83634e-06 | 0.00135512 | 0.000421054 | 0.779092 | 0.437249 | 30.94 |
| P2 | 1.83472e-06 | 0.00135452 | 0.000419399 | 0.779287 | 0.440128 | 30.98 |
| P3 | 1.87216e-06 | 0.00136824 | 0.000436988 | 0.774783 | 0.443238 | 30.92 |
| G1 | 1.82046e-06 | 0.00134924 | 0.000424782 | 0.781002 | 0.877562 | 33.69 |

## h1 TEST metrics (3-seed mean)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 4.05812e-06 | 0.00201448 | 0.000538739 | 0.819036 | 0.477328 | 32.01 |
| P1 | 4.05968e-06 | 0.00201487 | 0.000533751 | 0.818967 | 0.472876 | 31.17 |
| P2 | 4.06074e-06 | 0.00201513 | 0.000533219 | 0.81892 | 0.474336 | 31.27 |
| P3 | 4.21548e-06 | 0.00205314 | 0.00054902 | 0.812019 | 0.485719 | 31.20 |
| G1 | 3.96136e-06 | 0.00199031 | 0.0005331 | 0.823351 | 0.516594 | 34.39 |

## h5 VAL metrics (3-seed mean)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 2.1681e-06 | 0.00147245 | 0.000473666 | 0.739435 | 0.509637 | 48.52 |
| P1 | 2.22654e-06 | 0.00149216 | 0.000485894 | 0.732412 | 0.506196 | 48.54 |
| P2 | 2.18733e-06 | 0.00147896 | 0.000476297 | 0.737125 | 0.503117 | 48.52 |
| P3 | 2.15542e-06 | 0.00146813 | 0.000466772 | 0.740959 | 0.513001 | 48.44 |
| G1 | 2.11845e-06 | 0.00145549 | 0.000461872 | 0.745403 | 0.509102 | 48.68 |

## h5 TEST metrics (3-seed mean)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 5.24084e-06 | 0.00228929 | 0.000602656 | 0.766788 | 0.567625 | 48.53 |
| P1 | 5.12862e-06 | 0.00226464 | 0.000606516 | 0.771782 | 0.56478 | 47.98 |
| P2 | 5.15428e-06 | 0.0022703 | 0.000601621 | 0.77064 | 0.559854 | 48.04 |
| P3 | 5.34955e-06 | 0.00231291 | 0.000601409 | 0.761951 | 0.576488 | 47.88 |
| G1 | 5.31428e-06 | 0.00230527 | 0.000599607 | 0.76352 | 0.575926 | 48.22 |

## h10 VAL metrics (3-seed mean)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 2.33764e-06 | 0.00152893 | 0.00049489 | 0.720288 | 0.547378 | 50.00 |
| P1 | 2.4164e-06 | 0.00155448 | 0.000509788 | 0.710864 | 0.554232 | 50.45 |
| P2 | 2.43387e-06 | 0.00156008 | 0.000512833 | 0.708773 | 0.556383 | 50.54 |
| P3 | 2.31654e-06 | 0.00152201 | 0.00048964 | 0.722812 | 0.554665 | 50.56 |
| G1 | 2.27004e-06 | 0.00150666 | 0.00048627 | 0.728376 | 0.548133 | 50.00 |

## h10 TEST metrics (3-seed mean)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 5.58056e-06 | 0.00236232 | 0.000626701 | 0.752343 | 0.606054 | 48.75 |
| P1 | 5.51546e-06 | 0.0023485 | 0.000631609 | 0.755232 | 0.612317 | 49.10 |
| P2 | 5.50704e-06 | 0.00234671 | 0.000633176 | 0.755606 | 0.614645 | 48.95 |
| P3 | 5.72058e-06 | 0.00239177 | 0.000626076 | 0.746129 | 0.616804 | 49.03 |
| G1 | 5.67168e-06 | 0.00238153 | 0.000626558 | 0.748299 | 0.615622 | 49.10 |

## h22 VAL metrics (3-seed mean)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 2.50095e-06 | 0.00158144 | 0.000519785 | 0.698291 | 0.604413 | 49.06 |
| P1 | 2.56838e-06 | 0.00160261 | 0.00052906 | 0.690156 | 0.611546 | 49.29 |
| P2 | 2.62038e-06 | 0.00161874 | 0.000536949 | 0.683883 | 0.613358 | 49.36 |
| P3 | 2.48755e-06 | 0.00157719 | 0.000515883 | 0.699907 | 0.605964 | 49.53 |
| G1 | 2.40422e-06 | 0.00155054 | 0.000504323 | 0.709961 | 0.599794 | 49.45 |

## h22 TEST metrics (3-seed mean)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 6.05317e-06 | 0.00246032 | 0.0006521 | 0.728776 | 0.661699 | 48.97 |
| P1 | 6.11514e-06 | 0.00247288 | 0.000657827 | 0.726 | 0.676668 | 48.73 |
| P2 | 6.08262e-06 | 0.0024663 | 0.000662111 | 0.727456 | 0.678493 | 48.85 |
| P3 | 6.1328e-06 | 0.00247645 | 0.000651921 | 0.725208 | 0.673834 | 48.89 |
| G1 | 6.17323e-06 | 0.0024846 | 0.00064783 | 0.723397 | 0.678706 | 48.49 |

## Reading

- Nesting holds at every horizon: graph-off readout determinism 0.0 for all seeds (`nesting_check` in each canonical JSON); 'remove the GAT from G1' lands exactly on P3.
- Graph verdict is B (null) at all four horizons on the consistent basis: the graph never clears the A bar (G1 QLIKE < P3 in all seeds AND per-seed DM-QLIKE significant-negative in all seeds).
- h1 caveat: G1 improves MSE/RMSE/R2/DirAcc over P3 and DM-MSE is significant-negative in 3/3 test seeds, but QLIKE is unstable (seed 2026 inflates G1 QLIKE), so the QLIKE-based verdict stays B; the QLIKE instability at the 1-day horizon is driven by near-floor predictions.
- h10/h22: small VAL QLIKE improvement for G1 (h22 VAL paired-t p=0.0002) that does not carry to held-out TEST (h22 TEST 0/3 seeds), consistent with the h5 null.
