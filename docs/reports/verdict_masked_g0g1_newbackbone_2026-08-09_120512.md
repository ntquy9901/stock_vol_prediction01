# Masked G0/G1 graph verdict (screening-config P3 backbone)

Backbone: screening P3 (5 epochs, dropout 0.2), leakage-safe graph-bound train set. Masked manifest + positivity floor. Horizon 5, 15 epochs, seeds [42, 123, 2026].

## Adjacency: dense

**Verdict: B** (G1<G0 in 2/3 seeds; delta_valloss mean=-7.861e-04; paired-t p=0.2818; DM-QLIKE all sig-neg=False)

| seed | G0 valloss | G1 valloss | delta(G1-G0) | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.837168 | 0.837278 | +1.104e-04 | -0.525 | 0.5999 | -1.020 | 0.3078 | 14418 |
| 123 | 0.837673 | 0.836955 | -7.175e-04 | +0.003 | 0.9977 | -1.472 | 0.1411 | 14418 |
| 2026 | 0.836740 | 0.834989 | -1.751e-03 | -0.331 | 0.7403 | -1.128 | 0.2593 | 14418 |

### VAL metrics (mean over seeds)
| metric | G0 | G1 | G1-G0 |
|---|---|---|---|
| mse | 2.13622e-06 | 2.11949e-06 | -1.67e-08 |
| rmse | 0.00146158 | 0.00145585 | -5.73e-06 |
| mae | 0.000463869 | 0.000464359 | +4.9e-07 |
| r2 | 0.743267 | 0.745277 | +0.00201 |
| qlike | 0.509512 | 0.508961 | -0.000551 |
| directional_accuracy | 48.577 | 48.8811 | +0.304 |

### TEST metrics (mean over seeds)
| metric | G0 | G1 | G1-G0 |
|---|---|---|---|
| mse | 5.31786e-06 | 5.2854e-06 | -3.25e-08 |
| rmse | 0.00230605 | 0.002299 | -7.05e-06 |
| mae | 0.000599182 | 0.000604267 | +5.08e-06 |
| r2 | 0.763361 | 0.764806 | +0.00144 |
| qlike | 0.573077 | 0.576181 | +0.0031 |
| directional_accuracy | 47.774 | 47.9592 | +0.185 |

## Adjacency: knn

**Verdict: B** (G1<G0 in 3/3 seeds; delta_valloss mean=-6.348e-04; paired-t p=0.0185; DM-QLIKE all sig-neg=False)

| seed | G0 valloss | G1 valloss | delta(G1-G0) | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.837168 | 0.836594 | -5.741e-04 | +0.337 | 0.7358 | -1.386 | 0.1658 | 14418 |
| 123 | 0.837673 | 0.837150 | -5.231e-04 | -0.521 | 0.6022 | -2.010 | 0.04449 | 14418 |
| 2026 | 0.836740 | 0.835933 | -8.073e-04 | -0.790 | 0.4294 | -1.496 | 0.1348 | 14418 |

### VAL metrics (mean over seeds)
| metric | G0 | G1 | G1-G0 |
|---|---|---|---|
| mse | 2.13622e-06 | 2.11902e-06 | -1.72e-08 |
| rmse | 0.00146158 | 0.00145569 | -5.89e-06 |
| mae | 0.000463869 | 0.00046206 | -1.81e-06 |
| r2 | 0.743267 | 0.745333 | +0.00207 |
| qlike | 0.509512 | 0.509197 | -0.000315 |
| directional_accuracy | 48.577 | 48.6768 | +0.0998 |

### TEST metrics (mean over seeds)
| metric | G0 | G1 | G1-G0 |
|---|---|---|---|
| mse | 5.31786e-06 | 5.31318e-06 | -4.68e-09 |
| rmse | 0.00230605 | 0.00230503 | -1.01e-06 |
| mae | 0.000599182 | 0.000599781 | +5.99e-07 |
| r2 | 0.763361 | 0.763569 | +0.000208 |
| qlike | 0.573077 | 0.575919 | +0.00284 |
| directional_accuracy | 47.774 | 48.2589 | +0.485 |
