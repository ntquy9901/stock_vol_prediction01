# TrackA volatility models -- robustness report (held-out test dumps)

Source dumps: `results/trackA_ablation_h{h}_seed{s}_2026-08-15_085544_loo/<rung>/predictions_test.json` (no training; existing dumps only).
Seed ensemble: [42, 123, 2026, 7, 2024] (per-observation mean over seeds on shared keys). Horizons: [1, 5, 10, 22].
Rung -> dump dir: HAR->P0, FULL->FULL, LSTM_only->lstm_only, minus_graph->minus_graph.
Bootstrap: moving-block by date, 2000 resamples, fixed numpy seed 12345; block length max(20, 2*horizon) trading days (capped at #dates).
Metrics: MSE/RMSE/MAE/R2/QLIKE only (no directional accuracy). QLIKE uses the shared positivity floor 1e-8 (dm_report._qlike) for every rung.

## (A) Per-year metrics by rung

### Horizon h=1

| Year | Rung | n | MSE | RMSE | MAE | R2 | QLIKE |
|---|---|---|---|---|---|---|---|
| 2023 | HAR | 529 | 0.000000 | 0.000332 | 0.000206 | 0.2278 | 0.389174 |
| 2023 | FULL | 529 | 0.000000 | 0.000330 | 0.000208 | 0.2372 | 0.391233 |
| 2023 | LSTM_only | 529 | 0.000000 | 0.000327 | 0.000204 | 0.2484 | 0.381192 |
| 2023 | minus_graph | 529 | 0.000000 | 0.000327 | 0.000204 | 0.2479 | 0.384656 |
| 2024 | HAR | 3656 | 0.000000 | 0.000247 | 0.000150 | 0.1290 | 0.464027 |
| 2024 | FULL | 3656 | 0.000000 | 0.000251 | 0.000154 | 0.1002 | 0.474842 |
| 2024 | LSTM_only | 3656 | 0.000000 | 0.000248 | 0.000148 | 0.1206 | 0.463772 |
| 2024 | minus_graph | 3656 | 0.000000 | 0.000249 | 0.000150 | 0.1143 | 0.467464 |
| 2025 | HAR | 6925 | 0.000005 | 0.002235 | 0.000634 | 0.8333 | 0.506859 |
| 2025 | FULL | 6925 | 0.000005 | 0.002169 | 0.000627 | 0.8430 | 0.499647 |
| 2025 | LSTM_only | 6925 | 0.000005 | 0.002174 | 0.000622 | 0.8422 | 0.494827 |
| 2025 | minus_graph | 6925 | 0.000005 | 0.002181 | 0.000624 | 0.8413 | 0.497482 |
| 2026 | HAR | 3486 | 0.000007 | 0.002641 | 0.000815 | 0.7858 | 0.462825 |
| 2026 | FULL | 3486 | 0.000007 | 0.002670 | 0.000834 | 0.7810 | 0.459045 |
| 2026 | LSTM_only | 3486 | 0.000007 | 0.002712 | 0.000839 | 0.7740 | 0.456016 |
| 2026 | minus_graph | 3486 | 0.000007 | 0.002685 | 0.000830 | 0.7785 | 0.458382 |

FULL-minus-HAR mean QLIKE per year (h=1); negative => FULL lower QLIKE that year:

| Year | FULL-HAR mean QLIKE diff |
|---|---|
| 2023 | +0.002059 |
| 2024 | +0.010815 |
| 2025 | -0.007212 |
| 2026 | -0.003780 |

### Horizon h=5

| Year | Rung | n | MSE | RMSE | MAE | R2 | QLIKE |
|---|---|---|---|---|---|---|---|
| 2023 | HAR | 501 | 0.000000 | 0.000331 | 0.000217 | 0.2156 | 0.423456 |
| 2023 | FULL | 501 | 0.000000 | 0.000333 | 0.000214 | 0.2074 | 0.416060 |
| 2023 | LSTM_only | 501 | 0.000000 | 0.000332 | 0.000215 | 0.2099 | 0.414084 |
| 2023 | minus_graph | 501 | 0.000000 | 0.000333 | 0.000216 | 0.2072 | 0.416804 |
| 2024 | HAR | 3612 | 0.000000 | 0.000259 | 0.000173 | 0.0373 | 0.536153 |
| 2024 | FULL | 3612 | 0.000000 | 0.000259 | 0.000164 | 0.0434 | 0.523936 |
| 2024 | LSTM_only | 3612 | 0.000000 | 0.000259 | 0.000163 | 0.0437 | 0.522903 |
| 2024 | minus_graph | 3612 | 0.000000 | 0.000259 | 0.000163 | 0.0443 | 0.522586 |
| 2025 | HAR | 6865 | 0.000007 | 0.002576 | 0.000714 | 0.7782 | 0.615310 |
| 2025 | FULL | 6865 | 0.000006 | 0.002541 | 0.000699 | 0.7840 | 0.615300 |
| 2025 | LSTM_only | 6865 | 0.000006 | 0.002540 | 0.000698 | 0.7842 | 0.617421 |
| 2025 | minus_graph | 6865 | 0.000006 | 0.002543 | 0.000699 | 0.7837 | 0.617233 |
| 2026 | HAR | 3486 | 0.000009 | 0.002924 | 0.000895 | 0.7373 | 0.551309 |
| 2026 | FULL | 3486 | 0.000009 | 0.002944 | 0.000897 | 0.7337 | 0.553348 |
| 2026 | LSTM_only | 3486 | 0.000009 | 0.002948 | 0.000899 | 0.7330 | 0.554877 |
| 2026 | minus_graph | 3486 | 0.000009 | 0.002944 | 0.000898 | 0.7337 | 0.551939 |

FULL-minus-HAR mean QLIKE per year (h=5); negative => FULL lower QLIKE that year:

| Year | FULL-HAR mean QLIKE diff |
|---|---|
| 2023 | -0.007397 |
| 2024 | -0.012217 |
| 2025 | -0.000010 |
| 2026 | +0.002040 |

### Horizon h=10

| Year | Rung | n | MSE | RMSE | MAE | R2 | QLIKE |
|---|---|---|---|---|---|---|---|
| 2023 | HAR | 466 | 0.000000 | 0.000351 | 0.000230 | 0.1548 | 0.469138 |
| 2023 | FULL | 466 | 0.000000 | 0.000349 | 0.000227 | 0.1619 | 0.457629 |
| 2023 | LSTM_only | 466 | 0.000000 | 0.000351 | 0.000230 | 0.1536 | 0.467355 |
| 2023 | minus_graph | 466 | 0.000000 | 0.000351 | 0.000231 | 0.1526 | 0.468445 |
| 2024 | HAR | 3557 | 0.000000 | 0.000269 | 0.000185 | -0.0253 | 0.569092 |
| 2024 | FULL | 3557 | 0.000000 | 0.000263 | 0.000169 | 0.0180 | 0.541486 |
| 2024 | LSTM_only | 3557 | 0.000000 | 0.000268 | 0.000180 | -0.0199 | 0.560140 |
| 2024 | minus_graph | 3557 | 0.000000 | 0.000269 | 0.000182 | -0.0284 | 0.563193 |
| 2025 | HAR | 6790 | 0.000007 | 0.002671 | 0.000741 | 0.7610 | 0.653421 |
| 2025 | FULL | 6790 | 0.000007 | 0.002663 | 0.000733 | 0.7624 | 0.671890 |
| 2025 | LSTM_only | 6790 | 0.000007 | 0.002647 | 0.000735 | 0.7651 | 0.658677 |
| 2025 | minus_graph | 6790 | 0.000007 | 0.002648 | 0.000736 | 0.7650 | 0.657083 |
| 2026 | HAR | 3486 | 0.000009 | 0.002971 | 0.000921 | 0.7288 | 0.602116 |
| 2026 | FULL | 3486 | 0.000009 | 0.003003 | 0.000932 | 0.7231 | 0.624290 |
| 2026 | LSTM_only | 3486 | 0.000009 | 0.002993 | 0.000931 | 0.7249 | 0.614226 |
| 2026 | minus_graph | 3486 | 0.000009 | 0.002992 | 0.000932 | 0.7250 | 0.613853 |

FULL-minus-HAR mean QLIKE per year (h=10); negative => FULL lower QLIKE that year:

| Year | FULL-HAR mean QLIKE diff |
|---|---|
| 2023 | -0.011509 |
| 2024 | -0.027606 |
| 2025 | +0.018470 |
| 2026 | +0.022174 |

### Horizon h=22

| Year | Rung | n | MSE | RMSE | MAE | R2 | QLIKE |
|---|---|---|---|---|---|---|---|
| 2023 | HAR | 382 | 0.000000 | 0.000324 | 0.000232 | 0.0734 | 0.515885 |
| 2023 | FULL | 382 | 0.000000 | 0.000325 | 0.000231 | 0.0664 | 0.518351 |
| 2023 | LSTM_only | 382 | 0.000000 | 0.000329 | 0.000235 | 0.0459 | 0.527537 |
| 2023 | minus_graph | 382 | 0.000000 | 0.000323 | 0.000226 | 0.0769 | 0.509453 |
| 2024 | HAR | 3425 | 0.000000 | 0.000279 | 0.000202 | -0.1767 | 0.614683 |
| 2024 | FULL | 3425 | 0.000000 | 0.000267 | 0.000183 | -0.0829 | 0.575704 |
| 2024 | LSTM_only | 3425 | 0.000000 | 0.000270 | 0.000187 | -0.1004 | 0.584074 |
| 2024 | minus_graph | 3425 | 0.000000 | 0.000268 | 0.000183 | -0.0853 | 0.579197 |
| 2025 | HAR | 6610 | 0.000008 | 0.002787 | 0.000759 | 0.7313 | 0.714289 |
| 2025 | FULL | 6610 | 0.000008 | 0.002818 | 0.000755 | 0.7253 | 0.755412 |
| 2025 | LSTM_only | 6610 | 0.000008 | 0.002812 | 0.000756 | 0.7265 | 0.752090 |
| 2025 | minus_graph | 6610 | 0.000008 | 0.002825 | 0.000752 | 0.7238 | 0.748741 |
| 2026 | HAR | 3486 | 0.000009 | 0.003031 | 0.000954 | 0.7178 | 0.674105 |
| 2026 | FULL | 3486 | 0.000010 | 0.003082 | 0.000975 | 0.7082 | 0.721017 |
| 2026 | LSTM_only | 3486 | 0.000009 | 0.003034 | 0.000957 | 0.7173 | 0.708047 |
| 2026 | minus_graph | 3486 | 0.000009 | 0.003043 | 0.000958 | 0.7156 | 0.720421 |

FULL-minus-HAR mean QLIKE per year (h=22); negative => FULL lower QLIKE that year:

| Year | FULL-HAR mean QLIKE diff |
|---|---|
| 2023 | +0.002466 |
| 2024 | -0.038979 |
| 2025 | +0.041123 |
| 2026 | +0.046912 |

## (B) Per-ticker QLIKE (FULL vs HAR)

### Horizon h=1

Tickers with FULL lower QLIKE: 20/33; FULL higher: 13/33; tie: 0.

5 tickers where FULL is most below HAR (most negative FULL-HAR):

| ticker_id | n | QLIKE HAR | QLIKE FULL | FULL-HAR |
|---|---|---|---|---|
| 5 | 686 | 0.443178 | 0.413876 | -0.029302 |
| 14 | 311 | 0.405455 | 0.377472 | -0.027983 |
| 16 | 300 | 0.529820 | 0.505653 | -0.024167 |
| 27 | 309 | 0.565013 | 0.540883 | -0.024129 |
| 13 | 404 | 0.467728 | 0.450277 | -0.017452 |

5 tickers where FULL is most above HAR (most positive FULL-HAR):

| ticker_id | n | QLIKE HAR | QLIKE FULL | FULL-HAR |
|---|---|---|---|---|
| 19 | 599 | 0.511424 | 0.578880 | +0.067456 |
| 25 | 592 | 0.509601 | 0.527792 | +0.018192 |
| 6 | 484 | 0.491298 | 0.507250 | +0.015952 |
| 29 | 306 | 0.571497 | 0.586012 | +0.014515 |
| 18 | 314 | 0.584124 | 0.598583 | +0.014459 |

### Horizon h=5

Tickers with FULL lower QLIKE: 20/33; FULL higher: 13/33; tie: 0.

5 tickers where FULL is most below HAR (most negative FULL-HAR):

| ticker_id | n | QLIKE HAR | QLIKE FULL | FULL-HAR |
|---|---|---|---|---|
| 0 | 684 | 0.714714 | 0.651616 | -0.063098 |
| 20 | 149 | 0.405314 | 0.366394 | -0.038920 |
| 2 | 416 | 0.692512 | 0.662871 | -0.029641 |
| 24 | 259 | 0.434121 | 0.410830 | -0.023291 |
| 17 | 262 | 0.393496 | 0.375290 | -0.018206 |

5 tickers where FULL is most above HAR (most positive FULL-HAR):

| ticker_id | n | QLIKE HAR | QLIKE FULL | FULL-HAR |
|---|---|---|---|---|
| 29 | 302 | 0.759123 | 0.827090 | +0.067967 |
| 26 | 468 | 0.735991 | 0.780462 | +0.044471 |
| 18 | 310 | 0.731789 | 0.752775 | +0.020986 |
| 22 | 687 | 0.519747 | 0.535399 | +0.015652 |
| 19 | 595 | 0.628546 | 0.643273 | +0.014727 |

### Horizon h=10

Tickers with FULL lower QLIKE: 13/33; FULL higher: 20/33; tie: 0.

5 tickers where FULL is most below HAR (most negative FULL-HAR):

| ticker_id | n | QLIKE HAR | QLIKE FULL | FULL-HAR |
|---|---|---|---|---|
| 0 | 679 | 0.778908 | 0.700356 | -0.078552 |
| 19 | 590 | 0.701617 | 0.676715 | -0.024901 |
| 2 | 411 | 0.746767 | 0.722301 | -0.024466 |
| 20 | 144 | 0.445570 | 0.425782 | -0.019788 |
| 9 | 643 | 0.470535 | 0.452213 | -0.018322 |

5 tickers where FULL is most above HAR (most positive FULL-HAR):

| ticker_id | n | QLIKE HAR | QLIKE FULL | FULL-HAR |
|---|---|---|---|---|
| 26 | 463 | 0.912046 | 1.061209 | +0.149164 |
| 29 | 297 | 0.870908 | 1.008144 | +0.137236 |
| 18 | 305 | 0.776779 | 0.821063 | +0.044284 |
| 16 | 291 | 0.821794 | 0.853798 | +0.032005 |
| 15 | 542 | 0.461348 | 0.490805 | +0.029457 |

### Horizon h=22

Tickers with FULL lower QLIKE: 8/33; FULL higher: 25/33; tie: 0.

5 tickers where FULL is most below HAR (most negative FULL-HAR):

| ticker_id | n | QLIKE HAR | QLIKE FULL | FULL-HAR |
|---|---|---|---|---|
| 0 | 667 | 0.870758 | 0.801617 | -0.069141 |
| 19 | 578 | 0.808275 | 0.785909 | -0.022366 |
| 2 | 399 | 0.824542 | 0.806536 | -0.018006 |
| 9 | 631 | 0.516762 | 0.505597 | -0.011165 |
| 25 | 571 | 0.823194 | 0.813923 | -0.009271 |

5 tickers where FULL is most above HAR (most positive FULL-HAR):

| ticker_id | n | QLIKE HAR | QLIKE FULL | FULL-HAR |
|---|---|---|---|---|
| 26 | 451 | 1.254247 | 1.491142 | +0.236896 |
| 29 | 285 | 0.910529 | 1.074554 | +0.164025 |
| 16 | 279 | 0.936703 | 1.029677 | +0.092974 |
| 28 | 637 | 0.755419 | 0.829713 | +0.074294 |
| 8 | 251 | 0.542891 | 0.594455 | +0.051564 |

## (C) Block-bootstrap 95% CI on mean QLIKE loss difference

d_t = QLIKE_FULL - QLIKE_comparator, aligned per observation. Negative point estimate => FULL lower QLIKE. CI excluding 0 => difference significant at the 5% level under the moving-block resampling.

| Horizon | Comparison | n_obs | n_dates | block_len | point (FULL-comp) | CI lo (2.5%) | CI hi (97.5%) | excludes 0 |
|---|---|---|---|---|---|---|---|---|
| 1 | FULL vs HAR | 14596 | 714 | 20 | -0.001541 | -0.008895 | +0.005101 | no |
| 1 | FULL vs LSTM_only | 14596 | 714 | 20 | +0.006147 | +0.002824 | +0.009567 | yes |
| 5 | FULL vs HAR | 14464 | 710 | 20 | -0.002820 | -0.014490 | +0.014601 | no |
| 5 | FULL vs LSTM_only | 14464 | 710 | 20 | -0.001049 | -0.004356 | +0.001116 | no |
| 10 | FULL vs HAR | 14299 | 702 | 20 | +0.006934 | -0.012517 | +0.032397 | no |
| 10 | FULL vs LSTM_only | 14299 | 702 | 20 | +0.003771 | -0.008229 | +0.018883 | no |
| 22 | FULL vs HAR | 13903 | 690 | 44 | +0.021779 | -0.012158 | +0.049861 | no |
| 22 | FULL vs LSTM_only | 13903 | 690 | 44 | +0.002517 | -0.004791 | +0.008312 | no |

### Objective reading

- h=1 FULL vs HAR: mean QLIKE difference -0.001541 (FULL lower QLIKE), 95% CI [-0.008895, +0.005101], includes 0.
- h=1 FULL vs LSTM_only: mean QLIKE difference +0.006147 (FULL higher QLIKE), 95% CI [+0.002824, +0.009567], excludes 0.
- h=5 FULL vs HAR: mean QLIKE difference -0.002820 (FULL lower QLIKE), 95% CI [-0.014490, +0.014601], includes 0.
- h=5 FULL vs LSTM_only: mean QLIKE difference -0.001049 (FULL lower QLIKE), 95% CI [-0.004356, +0.001116], includes 0.
- h=10 FULL vs HAR: mean QLIKE difference +0.006934 (FULL higher QLIKE), 95% CI [-0.012517, +0.032397], includes 0.
- h=10 FULL vs LSTM_only: mean QLIKE difference +0.003771 (FULL higher QLIKE), 95% CI [-0.008229, +0.018883], includes 0.
- h=22 FULL vs HAR: mean QLIKE difference +0.021779 (FULL higher QLIKE), 95% CI [-0.012158, +0.049861], includes 0.
- h=22 FULL vs LSTM_only: mean QLIKE difference +0.002517 (FULL higher QLIKE), 95% CI [-0.004791, +0.008312], includes 0.

