# Consistent-basis Track-B ladder P0 -> P1 -> P2 -> P3 -> G1 (h22)

One basis for every rung: masked manifest, leakage-safe graph-bound train, shared per-ticker scalers, positivity floor, identical val/test observations. Masked k-NN-8 adjacency, seeds [42, 123, 2026], horizon 22. P3 = the trained G1 model read out with the GAT/message-passing residual disabled (replaces the old separate G0 row).

## Nesting verification (G1 minus the GAT = P3)

The P3 row is the identical trained G1 model with the message-passing residual removed, so the equality holds by construction; the runtime check confirms the graph-off readout is deterministic (max abs pred diff on a re-evaluation) and the residual is non-trivial (mean abs raw pred diff G1 vs P3).

| seed | graph-off determinism (max abs diff) | graph effect (mean abs raw diff) | n_val_obs |
|---|---|---|---|
| 42 | 0.00e+00 | 3.256e-05 | 13857 |
| 123 | 0.00e+00 | 3.807e-05 | 13857 |
| 2026 | 0.00e+00 | 4.129e-05 | 13857 |

## VAL metrics (3-seed mean +/- std)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 2.50095e-06 +/- 0 | 0.00158144 +/- 2.65574e-19 | 0.000519785 +/- 0 | 0.698291 +/- 0 | 0.604413 +/- 0 | 49.06 +/- 0.00 |
| P1 | 2.56838e-06 +/- 2.2141e-08 | 0.00160261 +/- 6.90959e-06 | 0.00052906 +/- 3.16981e-06 | 0.690156 +/- 0.00267104 | 0.611546 +/- 0.00101832 | 49.29 +/- 0.07 |
| P2 | 2.62038e-06 +/- 3.04436e-08 | 0.00161874 +/- 9.41912e-06 | 0.000536949 +/- 4.01654e-06 | 0.683883 +/- 0.00367266 | 0.613358 +/- 0.00175702 | 49.36 +/- 0.08 |
| P3 | 2.48755e-06 +/- 1.11719e-08 | 0.00157719 +/- 3.54221e-06 | 0.000515883 +/- 1.8872e-06 | 0.699907 +/- 0.00134775 | 0.605964 +/- 0.000659643 | 49.53 +/- 0.26 |
| G1 | 2.40422e-06 +/- 2.79636e-08 | 0.00155054 +/- 9.02026e-06 | 0.000504323 +/- 3.44859e-06 | 0.709961 +/- 0.00337347 | 0.599794 +/- 0.000788617 | 49.45 +/- 0.04 |

## TEST metrics (3-seed mean +/- std)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 6.05317e-06 +/- 0 | 0.00246032 +/- 0 | 0.0006521 +/- 0 | 0.728776 +/- 0 | 0.661699 +/- 0 | 48.97 +/- 0.00 |
| P1 | 6.11514e-06 +/- 1.88811e-08 | 0.00247288 +/- 3.81837e-06 | 0.000657827 +/- 1.69237e-06 | 0.726 +/- 0.000846004 | 0.676668 +/- 0.00145667 | 48.73 +/- 0.02 |
| P2 | 6.08262e-06 +/- 5.67759e-09 | 0.0024663 +/- 1.15108e-06 | 0.000662111 +/- 1.90544e-06 | 0.727456 +/- 0.000254395 | 0.678493 +/- 0.00296348 | 48.85 +/- 0.04 |
| P3 | 6.1328e-06 +/- 6.8172e-09 | 0.00247645 +/- 1.37619e-06 | 0.000651921 +/- 1.3601e-06 | 0.725208 +/- 0.000305458 | 0.673834 +/- 0.0006966 | 48.89 +/- 0.14 |
| G1 | 6.17323e-06 +/- 7.83592e-09 | 0.0024846 +/- 1.57679e-06 | 0.00064783 +/- 1.4748e-06 | 0.723397 +/- 0.000351103 | 0.678706 +/- 0.00418931 | 48.49 +/- 0.24 |

## Graph effect (G1 vs P3), VAL

**Verdict: B** (G1 QLIKE < P3 in 3/3 seeds; QLIKE delta mean=-6.170e-03; paired-t p=0.0002; DM-QLIKE all sig-neg=False)

| seed | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |
|---|---|---|---|---|---|
| 42 | -3.179 | 0.001481 | -1.388 | 0.1652 | 13857 |
| 123 | -2.056 | 0.03976 | -1.917 | 0.05532 | 13857 |
| 2026 | -1.891 | 0.05866 | -1.968 | 0.04908 | 13857 |

## Graph effect (G1 vs P3), TEST

**Verdict: B** (G1 QLIKE < P3 in 0/3 seeds; QLIKE delta mean=+4.871e-03; paired-t p=0.1425; DM-QLIKE all sig-neg=False)

| seed | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |
|---|---|---|---|---|---|
| 42 | +0.414 | 0.6786 | +1.831 | 0.06716 | 13903 |
| 123 | +1.154 | 0.2486 | +1.284 | 0.1993 | 13903 |
| 2026 | +1.492 | 0.1358 | +1.096 | 0.2729 | 13903 |
