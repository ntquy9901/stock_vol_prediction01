# DM (date-clustered) QLIKE p-values — volatility-proxy robustness

p-value against the HAR-X baseline; winner in parentheses. '<0.001' denotes p below 0.001.

## yang_zhang

| estimator | panel | h | QLIKE HAR-X | QLIKE LSTM | QLIKE LSTM+GAT | LSTM vs HAR-X | LSTM+GAT vs HAR-X | LSTM+GAT vs LSTM |
|---|---|---|---|---|---|---|---|---|
| yang_zhang | vn30 | 1 | 0.0057 | 0.0069 | 0.0129 | <0.001 (HAR-X) | <0.001 (HAR-X) | <0.001 (LSTM) |
| yang_zhang | vn100 | 1 | 0.0062 | 0.0070 | 0.0080 | 0.455 (LSTM) | <0.001 (HAR-X) | <0.001 (LSTM) |
| yang_zhang | hnx | 1 | 0.0298 | 0.0312 | 0.0318 | 0.383 (LSTM) | 0.250 (HAR-X) | <0.001 (LSTM) |
| yang_zhang | hose | 1 | 0.0133 | 0.0171 | 0.0169 | <0.001 (HAR-X) | <0.001 (HAR-X) | 0.477 (LSTM+GAT) |
| yang_zhang | sp500 | 1 | 0.0318 | 0.0116 | 0.0625 | <0.001 (LSTM) | 0.004 (LSTM+GAT) | <0.001 (LSTM) |
| yang_zhang | vn30 | 5 | 0.0796 | 0.0871 | 0.0947 | 0.714 (HAR-X) | 0.320 (HAR-X) | 0.093 (LSTM) |
| yang_zhang | vn100 | 5 | 0.0651 | 0.0703 | 0.0774 | 0.957 (HAR-X) | 0.886 (HAR-X) | 0.577 (LSTM) |
| yang_zhang | hnx | 5 | 0.1283 | 0.1140 | 0.1105 | <0.001 (LSTM) | <0.001 (LSTM+GAT) | 0.989 (LSTM+GAT) |
| yang_zhang | hose | 5 | 0.0840 | 0.0841 | 0.0868 | 0.103 (LSTM) | 0.143 (LSTM+GAT) | 0.248 (LSTM) |
| yang_zhang | sp500 | 5 | 0.4440 | 0.0940 | 0.1294 | <0.001 (LSTM) | <0.001 (LSTM+GAT) | <0.001 (LSTM) |
| yang_zhang | vn30 | 10 | 0.2231 | 0.2439 | 0.2200 | 0.650 (HAR-X) | 0.618 (LSTM+GAT) | 0.271 (LSTM+GAT) |
| yang_zhang | vn100 | 10 | 0.1700 | 0.2034 | 0.2396 | 0.478 (HAR-X) | 0.572 (HAR-X) | 0.083 (LSTM+GAT) |
| yang_zhang | hnx | 10 | 0.2374 | 0.2328 | 0.1959 | 0.236 (LSTM) | <0.001 (LSTM+GAT) | 0.078 (LSTM+GAT) |
| yang_zhang | hose | 10 | 0.1790 | 0.2531 | 0.2105 | 0.344 (HAR-X) | 0.517 (HAR-X) | 0.159 (LSTM+GAT) |
| yang_zhang | sp500 | 10 | 0.9917 | 0.2077 | 0.2126 | <0.001 (LSTM) | <0.001 (LSTM+GAT) | 0.807 (LSTM) |
| yang_zhang | vn30 | 22 | 0.3281 | 0.3936 | 0.4079 | 0.260 (HAR-X) | 0.206 (HAR-X) | 0.059 (LSTM) |
| yang_zhang | vn100 | 22 | 0.2983 | 0.5826 | 0.4518 | 0.216 (HAR-X) | 0.161 (HAR-X) | 0.357 (LSTM+GAT) |
| yang_zhang | hnx | 22 | 0.3404 | 0.3355 | 0.3455 | 0.663 (LSTM) | 0.198 (HAR-X) | 0.411 (LSTM) |
| yang_zhang | hose | 22 | 0.3025 | 0.3320 | 0.3349 | 0.711 (HAR-X) | 0.687 (HAR-X) | 0.928 (LSTM+GAT) |
| yang_zhang | sp500 | 22 | 0.4532 | 0.3647 | 0.6130 | <0.001 (LSTM) | 0.421 (LSTM+GAT) | 0.230 (LSTM) |

## rogers_satchell

| estimator | panel | h | QLIKE HAR-X | QLIKE LSTM | QLIKE LSTM+GAT | LSTM vs HAR-X | LSTM+GAT vs HAR-X | LSTM+GAT vs LSTM |
|---|---|---|---|---|---|---|---|---|
| rogers_satchell | vn30 | 1 | 0.9691 | 1.0283 | 1.0032 | 0.255 (HAR-X) | 0.072 (LSTM+GAT) | <0.001 (LSTM+GAT) |
| rogers_satchell | vn100 | 1 | 1.0665 | 1.1861 | 1.0854 | 0.899 (LSTM) | 0.290 (LSTM+GAT) | 0.005 (LSTM+GAT) |
| rogers_satchell | hnx | 1 | 3.8521 | 3.6672 | 3.6630 | <0.001 (LSTM) | <0.001 (LSTM+GAT) | 0.817 (LSTM) |
| rogers_satchell | hose | 1 | 2.2861 | 2.3269 | 2.2552 | 0.639 (LSTM) | 0.005 (LSTM+GAT) | <0.001 (LSTM+GAT) |

## parkinson

| estimator | panel | h | QLIKE HAR-X | QLIKE LSTM | QLIKE LSTM+GAT | LSTM vs HAR-X | LSTM+GAT vs HAR-X | LSTM+GAT vs LSTM |
|---|---|---|---|---|---|---|---|---|
| parkinson | vn30 | 1 | 0.5159 | 0.6794 | 0.6186 | <0.001 (HAR-X) | 0.006 (HAR-X) | <0.001 (LSTM+GAT) |
| parkinson | vn100 | 1 | 0.5115 | 0.6212 | 0.5521 | 0.001 (HAR-X) | 0.214 (HAR-X) | <0.001 (LSTM+GAT) |
| parkinson | hnx | 1 | 1.8721 | 1.8207 | 1.8157 | <0.001 (LSTM) | <0.001 (LSTM+GAT) | 0.853 (LSTM) |
| parkinson | hose | 1 | 1.2439 | 1.3100 | 1.3548 | 0.664 (LSTM) | 0.472 (LSTM+GAT) | 0.003 (LSTM+GAT) |
| parkinson | sp500 | 1 | 0.3618 | 0.6377 | 0.5430 | <0.001 (HAR-X) | <0.001 (HAR-X) | <0.001 (LSTM) |

Total cells: 29