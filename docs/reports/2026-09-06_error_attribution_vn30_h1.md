# Error attribution (HAR-X OOS) — vn30 h1

Test ticker-days: 10,881. QLIKE floor: 1.00e-08. HAR-X loss proxies data hardness (model-agnostic).

## Worst 15 tickers by mean QLIKE

| ticker | mean QLIKE | mean sqerr | n | dirty-day share |
|---|---|---|---|---|
| VCB | 0.667 | 1.76e-07 | 351 | 1/351 (0%) |
| SAB | 0.665 | 7.48e-08 | 351 | 0/351 (0%) |
| BCM | 0.625 | 1.77e-07 | 351 | 4/351 (1%) |
| GVR | 0.581 | 2.42e-07 | 351 | 4/351 (1%) |
| VJC | 0.561 | 2.64e-07 | 351 | 1/351 (0%) |
| CTG | 0.555 | 1.19e-07 | 351 | 1/351 (0%) |
| VIB | 0.537 | 2.30e-07 | 351 | 1/351 (0%) |
| BID | 0.535 | 1.13e-07 | 351 | 1/351 (0%) |
| VHM | 0.524 | 4.35e-07 | 351 | 4/351 (1%) |
| GAS | 0.520 | 1.53e-07 | 351 | 1/351 (0%) |
| MBB | 0.517 | 9.74e-08 | 351 | 2/351 (1%) |
| TCB | 0.501 | 1.36e-07 | 351 | 1/351 (0%) |
| ACB | 0.492 | 8.43e-08 | 351 | 1/351 (0%) |
| STB | 0.492 | 2.43e-07 | 351 | 1/351 (0%) |
| SHB | 0.490 | 2.24e-07 | 351 | 1/351 (0%) |

## Worst 30 ticker-days by QLIKE (with data annotation)

| ticker | date | target | QLIKE | kind | dirty flags |
|---|---|---|---|---|---|
| GAS | 2025-04-03 | 7.42e-04 | 26.56 | spike | - |
| GVR | 2025-04-11 | 4.91e-03 | 24.54 | spike | - |
| SAB | 2025-04-03 | 9.77e-04 | 22.67 | spike | - |
| VJC | 2025-04-03 | 1.82e-03 | 21.32 | spike | - |
| CTG | 2024-12-25 | 1.60e-03 | 20.86 | spike | - |
| PDR | 2025-04-11 | 5.38e-03 | 20.52 | spike | - |
| VCB | 2024-10-31 | 5.11e-04 | 16.01 | spike | - |
| SAB | 2025-12-01 | 1.45e-03 | 15.64 | spike | - |
| VCB | 2026-04-23 | 1.45e-03 | 15.46 | spike | - |
| SSI | 2024-12-05 | 2.49e-03 | 13.29 | spike | - |
| BCM | 2025-04-11 | 2.59e-03 | 13.03 | spike | - |
| SHB | 2025-07-07 | 1.60e-03 | 12.74 | spike | - |
| BID | 2026-04-23 | 1.63e-03 | 12.70 | spike | - |
| VCB | 2025-04-10 | 1.00e-08 | 11.22 | near-floor | dirty,zero_range |
| VIC | 2025-04-10 | 1.00e-08 | 11.19 | near-floor | dirty,zero_range |
| STB | 2025-04-10 | 1.00e-08 | 10.95 | near-floor | dirty,zero_range |
| TPB | 2025-04-10 | 1.00e-08 | 10.93 | near-floor | dirty,zero_range |
| VRE | 2025-04-10 | 1.00e-08 | 10.90 | near-floor | dirty,zero_range |
| TCB | 2025-04-10 | 1.00e-08 | 10.88 | near-floor | dirty,zero_range |
| POW | 2025-04-10 | 1.00e-08 | 10.85 | near-floor | dirty,zero_range |
| BID | 2025-04-10 | 1.00e-08 | 10.82 | near-floor | dirty,zero_range |
| VIB | 2025-04-10 | 1.00e-08 | 10.82 | near-floor | dirty,zero_range |
| FPT | 2025-04-10 | 1.00e-08 | 10.80 | near-floor | dirty,zero_range |
| SHB | 2025-04-10 | 1.00e-08 | 10.75 | near-floor | dirty,zero_range |
| HDB | 2025-04-10 | 1.00e-08 | 10.74 | near-floor | dirty,zero_range |
| MBB | 2025-04-10 | 1.00e-08 | 10.73 | near-floor | dirty,zero_range |
| ACB | 2025-04-10 | 1.00e-08 | 10.72 | near-floor | dirty,zero_range |
| CTG | 2025-04-10 | 1.00e-08 | 10.70 | near-floor | dirty,zero_range |
| GAS | 2025-04-10 | 1.00e-08 | 10.68 | near-floor | dirty,zero_range |
| VHM | 2025-04-10 | 1.00e-08 | 10.67 | near-floor | dirty,zero_range |

## Worst 15 forecast dates by mean QLIKE (market-wide)

| date | mean QLIKE | tickers |
|---|---|---|
| 2025-04-10 | 10.493 | 31 |
| 2025-04-03 | 4.981 | 31 |
| 2025-04-09 | 2.780 | 31 |
| 2025-10-20 | 2.602 | 31 |
| 2025-04-11 | 2.602 | 31 |
| 2025-04-22 | 2.143 | 31 |
| 2024-12-25 | 1.920 | 31 |
| 2025-07-29 | 1.847 | 31 |
| 2025-04-04 | 1.835 | 31 |
| 2026-07-20 | 1.651 | 31 |
| 2024-12-05 | 1.601 | 31 |
| 2025-12-16 | 1.588 | 31 |
| 2026-04-23 | 1.515 | 31 |
| 2025-12-12 | 1.356 | 31 |
| 2025-08-05 | 1.087 | 31 |
