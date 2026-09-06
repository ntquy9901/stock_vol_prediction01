# Error attribution (HAR-X OOS) — vn30 h10

Test ticker-days: 10,757. QLIKE floor: 1.00e-08. HAR-X loss proxies data hardness (model-agnostic).

## Worst 15 tickers by mean QLIKE

| ticker | mean QLIKE | mean sqerr | n | dirty-day share |
|---|---|---|---|---|
| VCB | 0.870 | 2.21e-07 | 347 | 1/347 (0%) |
| BCM | 0.796 | 2.49e-07 | 347 | 4/347 (1%) |
| ACB | 0.750 | 1.14e-07 | 347 | 1/347 (0%) |
| BID | 0.747 | 1.58e-07 | 347 | 1/347 (0%) |
| SAB | 0.746 | 8.96e-08 | 347 | 0/347 (0%) |
| VJC | 0.744 | 3.13e-07 | 347 | 1/347 (0%) |
| VIB | 0.731 | 2.60e-07 | 347 | 1/347 (0%) |
| GAS | 0.718 | 2.20e-07 | 347 | 1/347 (0%) |
| MBB | 0.671 | 1.22e-07 | 347 | 2/347 (1%) |
| CTG | 0.669 | 1.49e-07 | 347 | 1/347 (0%) |
| GVR | 0.666 | 2.95e-07 | 347 | 4/347 (1%) |
| VNM | 0.648 | 8.58e-08 | 347 | 1/347 (0%) |
| STB | 0.641 | 2.71e-07 | 347 | 1/347 (0%) |
| HDB | 0.637 | 1.78e-07 | 347 | 1/347 (0%) |
| SHB | 0.630 | 2.59e-07 | 347 | 1/347 (0%) |

## Worst 30 ticker-days by QLIKE (with data annotation)

| ticker | date | target | QLIKE | kind | dirty flags |
|---|---|---|---|---|---|
| VCB | 2025-04-09 | 6.87e-03 | 35.07 | spike | - |
| TCB | 2025-04-09 | 4.38e-03 | 19.22 | spike | - |
| BID | 2025-04-09 | 3.64e-03 | 19.14 | spike | - |
| ACB | 2025-04-09 | 3.33e-03 | 18.83 | spike | - |
| STB | 2025-04-09 | 4.00e-03 | 18.35 | spike | - |
| GAS | 2025-04-09 | 3.42e-03 | 17.72 | spike | - |
| VIB | 2025-08-05 | 4.39e-03 | 17.71 | spike | - |
| SHB | 2025-08-05 | 5.07e-03 | 14.46 | spike | - |
| PDR | 2025-04-11 | 5.38e-03 | 14.45 | spike | - |
| STB | 2025-04-04 | 3.26e-03 | 14.29 | spike | - |
| CTG | 2025-04-09 | 2.97e-03 | 14.03 | spike | - |
| VIB | 2025-04-09 | 3.96e-03 | 13.81 | spike | - |
| MBB | 2025-04-09 | 3.28e-03 | 13.47 | spike | - |
| VNM | 2025-04-04 | 2.85e-03 | 13.38 | spike | - |
| VHM | 2025-12-25 | 6.93e-03 | 12.79 | spike | - |
| PDR | 2026-07-28 | 5.00e-03 | 12.61 | spike | - |
| GVR | 2025-04-11 | 4.91e-03 | 12.56 | spike | - |
| TPB | 2025-04-09 | 4.77e-03 | 11.97 | spike | - |
| VPB | 2025-04-09 | 2.63e-03 | 10.43 | spike | - |
| HDB | 2024-12-30 | 2.82e-03 | 10.36 | spike | - |
| VRE | 2025-05-12 | 1.00e-08 | 10.09 | near-floor | dirty,zero_range,zero_volume |
| VHM | 2026-06-18 | 1.00e-08 | 10.08 | near-floor | dirty,zero_range |
| FPT | 2025-04-09 | 3.55e-03 | 9.97 | spike | - |
| VRE | 2025-05-05 | 1.00e-08 | 9.97 | near-floor | dirty,zero_range,zero_volume |
| VHM | 2026-08-06 | 1.00e-08 | 9.94 | near-floor | dirty,zero_range,zero_volume |
| VIC | 2025-04-10 | 1.00e-08 | 9.85 | near-floor | dirty,zero_range |
| BID | 2025-04-04 | 1.79e-03 | 9.81 | spike | - |
| SHB | 2025-07-29 | 3.65e-03 | 9.65 | spike | - |
| VPB | 2025-05-12 | 1.00e-08 | 9.58 | near-floor | dirty,zero_range,zero_volume |
| SHB | 2025-04-10 | 1.00e-08 | 9.56 | near-floor | dirty,zero_range |

## Worst 15 forecast dates by mean QLIKE (market-wide)

| date | mean QLIKE | tickers |
|---|---|---|
| 2025-04-09 | 9.854 | 31 |
| 2025-04-10 | 8.892 | 31 |
| 2025-04-04 | 5.127 | 31 |
| 2025-08-05 | 3.039 | 31 |
| 2025-04-11 | 2.995 | 31 |
| 2025-10-20 | 2.668 | 31 |
| 2025-07-29 | 2.576 | 31 |
| 2025-04-08 | 2.424 | 31 |
| 2025-04-22 | 2.048 | 31 |
| 2026-07-23 | 1.782 | 31 |
| 2025-04-03 | 1.755 | 31 |
| 2026-07-22 | 1.385 | 31 |
| 2026-01-14 | 1.299 | 31 |
| 2025-05-12 | 1.270 | 31 |
| 2026-01-12 | 1.148 | 31 |
