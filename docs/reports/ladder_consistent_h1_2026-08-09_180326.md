# Consistent-basis Track-B ladder P0 -> P1 -> P2 -> P3 -> G1 (h1)

One basis for every rung: masked manifest, leakage-safe graph-bound train, shared per-ticker scalers, positivity floor, identical val/test observations. Masked k-NN-8 adjacency, seeds [42, 123, 2026], horizon 1. P3 = the trained G1 model read out with the GAT/message-passing residual disabled (replaces the old separate G0 row).

## Nesting verification (G1 minus the GAT = P3)

The P3 row is the identical trained G1 model with the message-passing residual removed, so the equality holds by construction; the runtime check confirms the graph-off readout is deterministic (max abs pred diff on a re-evaluation) and the residual is non-trivial (mean abs raw pred diff G1 vs P3).

| seed | graph-off determinism (max abs diff) | graph effect (mean abs raw diff) | n_val_obs |
|---|---|---|---|
| 42 | 0.00e+00 | 9.751e-05 | 14550 |
| 123 | 0.00e+00 | 7.835e-05 | 14550 |
| 2026 | 0.00e+00 | 7.941e-05 | 14550 |

## VAL metrics (3-seed mean +/- std)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 1.84753e-06 +/- 0 | 0.00135924 +/- 0 | 0.000430766 +/- 6.63936e-20 | 0.777746 +/- 0 | 0.438032 +/- 6.7987e-17 | 31.89 +/- 0.00 |
| P1 | 1.83634e-06 +/- 2.52444e-09 | 0.00135512 +/- 9.31526e-07 | 0.000421054 +/- 1.49015e-06 | 0.779092 +/- 0.000303685 | 0.437249 +/- 0.00159903 | 30.94 +/- 0.03 |
| P2 | 1.83472e-06 +/- 4.24429e-10 | 0.00135452 +/- 1.56676e-07 | 0.000419399 +/- 6.05474e-07 | 0.779287 +/- 5.1058e-05 | 0.440128 +/- 0.0008975 | 30.98 +/- 0.01 |
| P3 | 1.87216e-06 +/- 2.76188e-08 | 0.00136824 +/- 1.008e-05 | 0.000436988 +/- 7.60766e-06 | 0.774783 +/- 0.00332249 | 0.443238 +/- 0.00190283 | 30.92 +/- 0.07 |
| G1 | 1.82046e-06 +/- 6.04277e-09 | 0.00134924 +/- 2.23879e-06 | 0.000424782 +/- 1.83374e-06 | 0.781002 +/- 0.000726933 | 0.877562 +/- 0.7689 | 33.69 +/- 0.25 |

## TEST metrics (3-seed mean +/- std)

| rung | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| P0 | 4.05812e-06 +/- 0 | 0.00201448 +/- 0 | 0.000538739 +/- 0 | 0.819036 +/- 1.35974e-16 | 0.477328 +/- 0 | 32.01 +/- 0.00 |
| P1 | 4.05968e-06 +/- 3.48281e-09 | 0.00201487 +/- 8.64183e-07 | 0.000533751 +/- 1.23208e-06 | 0.818967 +/- 0.000155309 | 0.472876 +/- 0.000853056 | 31.17 +/- 0.02 |
| P2 | 4.06074e-06 +/- 2.49969e-09 | 0.00201513 +/- 6.20256e-07 | 0.000533219 +/- 5.1683e-07 | 0.81892 +/- 0.000111468 | 0.474336 +/- 0.000642323 | 31.27 +/- 0.09 |
| P3 | 4.21548e-06 +/- 5.20623e-08 | 0.00205314 +/- 1.26758e-05 | 0.00054902 +/- 3.50865e-06 | 0.812019 +/- 0.00232161 | 0.485719 +/- 0.00414704 | 31.20 +/- 0.06 |
| G1 | 3.96136e-06 +/- 2.8701e-08 | 0.00199031 +/- 7.20403e-06 | 0.0005331 +/- 1.59228e-06 | 0.823351 +/- 0.00127986 | 0.516594 +/- 0.0681133 | 34.39 +/- 0.19 |

## Graph effect (G1 vs P3), VAL

**Verdict: B** (G1 QLIKE < P3 in 2/3 seeds; QLIKE delta mean=+4.343e-01; paired-t p=0.4317; DM-QLIKE all sig-neg=False)

| seed | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |
|---|---|---|---|---|---|
| 42 | -4.562 | 5.113e-06 | -2.258 | 0.02398 | 14550 |
| 123 | -4.694 | 2.7e-06 | -1.561 | 0.1185 | 14550 |
| 2026 | +1.420 | 0.1556 | -1.044 | 0.2967 | 14550 |

## Graph effect (G1 vs P3), TEST

**Verdict: B** (G1 QLIKE < P3 in 2/3 seeds; QLIKE delta mean=+3.088e-02; paired-t p=0.5305; DM-QLIKE all sig-neg=False)

| seed | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |
|---|---|---|---|---|---|
| 42 | -5.168 | 2.399e-07 | -3.273 | 0.001067 | 14596 |
| 123 | -0.877 | 0.3804 | -3.417 | 0.0006348 | 14596 |
| 2026 | +1.600 | 0.1097 | -2.727 | 0.00639 | 14596 |
