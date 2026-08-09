# Consistent-basis Track-B ladder P0 -> P1 -> P2 -> P3 -> G1 (h5)

One basis for every rung: masked manifest, leakage-safe graph-bound train, shared per-ticker scalers, positivity floor, identical val/test observations. Masked k-NN-8 adjacency, seeds [42, 123, 2026], horizon 5. P3 = the trained G1 model read out with the GAT/message-passing residual disabled (replaces the old separate G0 row).

## Nesting verification (G1 minus the GAT = P3)

The P3 row is the identical trained G1 model with the message-passing residual removed, so the equality holds by construction; the runtime check confirms the graph-off readout is deterministic (max abs pred diff on a re-evaluation) and the residual is non-trivial (mean abs raw pred diff G1 vs P3).

| seed | graph-off determinism (max abs diff) | graph effect (mean abs raw diff) | n_val_obs |
|---|---|---|---|
| 42 | 0.00e+00 | 4.377e-05 | 14418 |
| 123 | 0.00e+00 | 3.251e-05 | 14418 |
| 2026 | 0.00e+00 | 3.194e-05 | 14418 |

## VAL metrics (3-seed mean +/- std)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 2.1681e-06 +/- 0 | 0.00147245 +/- 0 | 0.000473666 +/- 0 | 0.739435 +/- 1.35974e-16 | 0.509637 +/- 0 | 48.52 +/- 0.00 |
| P1 | 2.22654e-06 +/- 1.10337e-08 | 0.00149216 +/- 3.69475e-06 | 0.000485894 +/- 1.97956e-06 | 0.732412 +/- 0.00132605 | 0.506196 +/- 0.000978811 | 48.54 +/- 0.02 |
| P2 | 2.18733e-06 +/- 1.34737e-08 | 0.00147896 +/- 4.55849e-06 | 0.000476297 +/- 3.83736e-06 | 0.737125 +/- 0.00161928 | 0.503117 +/- 0.000334143 | 48.52 +/- 0.08 |
| P3 | 2.15542e-06 +/- 9.60693e-09 | 0.00146813 +/- 3.2727e-06 | 0.000466772 +/- 2.97187e-06 | 0.740959 +/- 0.00115457 | 0.513001 +/- 0.000449907 | 48.44 +/- 0.09 |
| G1 | 2.11845e-06 +/- 6.69297e-09 | 0.00145549 +/- 2.30026e-06 | 0.000461872 +/- 2.97317e-06 | 0.745403 +/- 0.000804369 | 0.509102 +/- 0.00111176 | 48.68 +/- 0.07 |

## TEST metrics (3-seed mean +/- std)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 5.24084e-06 +/- 1.0374e-21 | 0.00228929 +/- 0 | 0.000602656 +/- 0 | 0.766788 +/- 0 | 0.567625 +/- 0 | 48.53 +/- 0.00 |
| P1 | 5.12862e-06 +/- 7.5581e-09 | 0.00226464 +/- 1.66841e-06 | 0.000606516 +/- 1.15375e-06 | 0.771782 +/- 0.000336327 | 0.56478 +/- 0.00128522 | 47.98 +/- 0.11 |
| P2 | 5.15428e-06 +/- 1.14114e-08 | 0.0022703 +/- 2.5124e-06 | 0.000601621 +/- 2.47682e-06 | 0.77064 +/- 0.000507797 | 0.559854 +/- 0.000904283 | 48.04 +/- 0.09 |
| P3 | 5.34955e-06 +/- 2.36663e-08 | 0.00231291 +/- 5.11288e-06 | 0.000601409 +/- 1.93572e-06 | 0.761951 +/- 0.00105312 | 0.576488 +/- 0.00047625 | 47.88 +/- 0.07 |
| G1 | 5.31428e-06 +/- 1.14142e-08 | 0.00230527 +/- 2.47576e-06 | 0.000599607 +/- 2.78639e-06 | 0.76352 +/- 0.000507917 | 0.575926 +/- 0.00295859 | 48.22 +/- 0.06 |

## Graph effect (G1 vs P3), VAL

**Verdict: B** (G1 QLIKE < P3 in 3/3 seeds; QLIKE delta mean=-3.899e-03; paired-t p=0.0096; DM-QLIKE all sig-neg=False)

| seed | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |
|---|---|---|---|---|---|
| 42 | -1.258 | 0.2083 | -2.230 | 0.02573 | 14418 |
| 123 | -2.400 | 0.01642 | -2.432 | 0.01503 | 14418 |
| 2026 | -2.738 | 0.006194 | -1.837 | 0.06623 | 14418 |

## Graph effect (G1 vs P3), TEST

**Verdict: B** (G1 QLIKE < P3 in 2/3 seeds; QLIKE delta mean=-5.620e-04; paired-t p=0.7913; DM-QLIKE all sig-neg=False)

| seed | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |
|---|---|---|---|---|---|
| 42 | +0.838 | 0.4021 | -1.856 | 0.06343 | 14464 |
| 123 | -1.732 | 0.08331 | -0.581 | 0.5613 | 14464 |
| 2026 | -1.856 | 0.06344 | -3.403 | 0.0006674 | 14464 |
