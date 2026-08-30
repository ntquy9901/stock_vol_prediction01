# HNX per-ticker diagnostic — 2026-08-30 01:04

One row per HNX ticker (299 total). Diagnostic to find per-stock data/model issues. Interactive table: `docs/reports/2026-08-30_hnx_per_ticker_diagnostic.html`.

Model columns: pending — GPU busy / skipped (DATA-only).

## Aggregate

- Flagged: 250 red, 5 amber, 44 ok.
- Aggregate exact-zero Parkinson (row-weighted) = 0.453; mean per-ticker zero-fraction = 0.444.
- Total valid rows across tickers = 1065741.

Flag rules — red: exact-zero Parkinson fraction > 0.30, any OHLC violation, or n_valid < 252. Amber: test floor-activation > 0.20, or per-ticker model QLIKE > 2x the panel median.

## Worst tickers

### Highest zero-Parkinson fraction

- LCD: 0.982
- QST: 0.957
- PTD: 0.950
- TET: 0.946
- BED: 0.933
- THS: 0.931
- PTX: 0.926
- KST: 0.912
- SGH: 0.911
- DNC: 0.910

### Fewest valid rows

- TD6: 298
- CAR: 739
- PCH: 1015
- DVM: 1022
- PPT: 1031
- HMR: 1125
- GMA: 1160
- VTZ: 1179
- KSF: 1211
- CST: 1324

### Most OHLC violations

- PTX: 327
- NAP: 214
- DTK: 209
- SPC: 166
- L40: 147
- KSV: 144
- DTG: 117
- MAS: 116
- GDW: 109
- TOT: 96

### Highest floor-activation

- S55: 0.916
- STP: 0.785
- HJS: 0.719
- PGS: 0.694
- TTC: 0.648
- VIT: 0.607
- HMH: 0.593
- EBS: 0.557
- PTS: 0.550
- HKT: 0.548

### Highest per-ticker QLIKE (max of 3 models)

- (none)

## Scope note

- The delivered target ``parkinson_volatility`` is computed from the intraday range only (high/low), so an open/close-outside-[low,high] bar does NOT corrupt the Parkinson target directly; it corrupts the overnight-augmented estimators (Garman-Klass / Rogers-Satchell / Yang-Zhang) that read open/close. It is still flagged as a raw-data-geometry defect.
- The exact-zero Parkinson fraction (high==low days) is the defect that DOES drive the Parkinson target and its QLIKE floor.

## Recommendation

- Tickers flagged red on zero-Parkinson fraction are unreliable: their Parkinson targets are dominated by limit/illiquid days and their point/QLIKE metrics are floor-driven, not forecast-driven. Exclude them from headline tables or report separately.
- Tickers with high test floor-activation are floor-driven even inside the screened panel; note this in the paper's data section as a QLIKE caveat rather than a model result.
- The screen already drops the worst illiquid names; the residual red/amber rows inside the kept universe are the ones to disclose as data-quality limitations.
