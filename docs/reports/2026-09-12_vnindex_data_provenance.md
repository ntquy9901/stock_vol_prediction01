# VNINDEX daily series: two-source construction and cross-check

**Date:** 2026-09-12
**Artifact:** `data/raw/prices/_market_index/vnindex.csv` (6,361 trading days, 2000-07-31 to 2026-09-11)
**Builder:** `scripts/etl_vn_index/build_vnindex.py` (tests: `scripts/etl_vn_index/test_build_vnindex.py`)
**Provenance record:** `data/raw/prices/_market_index/vnindex_provenance.json`

This series is the market-index target for the complex-network volatility experiment (Complexity-2026 replication, Section 2.4). It is stitched from two independent sources and cross-checked on their overlap.

## 1. Sources

| Segment | Source | Coverage | License |
|---|---|---|---|
| Historical | Zenodo DOI [10.5281/zenodo.21873555](https://doi.org/10.5281/zenodo.21873555), "VN-Index Daily Dataset for Volatility Forecasting (2000-2024)" (file `Dữ liệu Lịch sử VN Index.csv`, Investing.com-style export) | 2000-07-31 to 2024-12-16 | CC BY 4.0 |
| Recent supplement | vnstock VCI feed (`Quote(symbol="VNINDEX", source="VCI")`) | 2024-12-17 to 2026-09-11 | vnstock community |

The stitch boundary is 2024-12-17: Zenodo supplies all dates before it, vnstock from it onward. The vnstock community feed caps 1-day history at about 8 years, so it can only fill the recent tail; Zenodo supplies the long history back to the index's inception week (base value 100 on 2000-07-28; the Zenodo file begins 2000-07-31).

## 2. Cross-check (discrepancy verification)

**Internal, source-vs-source.** The two feeds overlap on 1,566 trading days (2018-09-13 to 2024-12-16). Comparing the daily Close:

| Statistic | Value |
|---|---|
| Mean absolute Close difference | 0.00056% |
| Max absolute Close difference | 0.2874% (2018-11-13: Zenodo 905.38 vs vnstock 907.99) |
| Days with difference > 0.1% | 4 of 1,566 |
| Days with difference > 0.5% | 0 of 1,566 |

The two independent sources agree to within 0.29% on every overlapping day and to 0.0006% on average, so the stitch introduces no level break in the index-point series.

**External anchor.** The last stitched session (2026-09-11) has Close 1,795.21, matching the reported market close of 1,795.21 that day ([Nhân Dân](https://en.nhandan.vn/infographic-selling-pressure-spreads-vn-index-falls-below-1800-points-on-september-11-post166516.html)).

## 3. Parsing and validation

- The Zenodo export uses Vietnamese headers (`Ngày, Lần cuối, Mở, Cao, Thấp, KL, % Thay đổi`), `DD/MM/YYYY` dates, and volume in display units (`538.93K`, `1.2M`, or blank). The builder parses dates day-first, prices as floats, and volume via the `K/M/B` suffix into a share count.
- `validate()` fails loud on non-monotonic or duplicate dates, non-positive or missing Close, and `high < low` geometry. The stitched output passes all checks.

## 4. Known limitation: volume display unit

The Close/OHLC index points share one scale across both sources (verified in Section 2). The VOLUME display unit differs: the Zenodo "KL" column is in Investing.com display units while the vnstock feed reports raw share counts, a scale gap at the 2024-12-17 boundary. The `vol_source` column tags each row (`zenodo_K_units` before the boundary, `vnstock_raw_shares` after). The experiment's headline target is the future index return-volatility, which is computed from Close and is unaffected. The secondary average-log-volume target must not stitch across the boundary; any window whose feature or target period straddles 2024-12 is excluded for that target.

## 5. S&P 500 index (^GSPC) target

The US market-index target is the real `^GSPC` (S&P 500 price index), for the multi-market arm of the experiment.

| Field | Value |
|---|---|
| Source | Yahoo Finance `^GSPC` (yfinance), daily OHLC + Adj Close + Volume |
| Coverage | 2000-01-03 to 2026-09-11, 6,713 trading days (single source, no stitch) |
| Cross-check | FRED `SP500` (public domain), 2,514-day overlap 2016-09-12 to 2026-09-11 |
| Close agreement | mean 0.00007%, max 0.12% (2021-08-11), 0 days > 0.5% |

Artifact `data/raw/prices/_market_index/gspc.csv`; provenance `gspc_provenance.json`; builder `scripts/etl_vn_index/build_gspc.py`. Index volume is not comparable to a stock or ETF volume, so only OHLC is used (Close for the return-volatility target).

## 6. Text for the paper (data section)

The VNINDEX daily series (2000-07-31 to 2026-09-11) is stitched from two sources: the Zenodo VN-Index daily dataset (DOI 10.5281/zenodo.21873555, CC BY 4.0) for 2000-2024 and the vnstock VCI feed for the 2025-2026 tail. The two sources overlap on 1,566 trading days, on which the daily Close agrees to a mean absolute difference of 0.0006% and a maximum of 0.29% (no day above 0.5%), so the join introduces no level break; the final session matches the reported market close. The S&P 500 index target is the real `^GSPC` (Yahoo Finance, 2000-01-03 to 2026-09-11), cross-checked against the FRED SP500 series on a 2,514-day overlap (mean 0.00007%, max 0.12%). Volume display units differ across sources and are used only for the secondary volume target, which is not stitched across the join.
