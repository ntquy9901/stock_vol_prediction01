# Summary of update — overnight run 2026-09-12

## Scope
Autonomous overnight run: obtain real Vietnamese earnings-announcement dates, run the full model set for HOSE
(all metrics + Diebold-Mariano), and bring the S&P 500 + HOSE paper to a near-final state. The GNNHAR results
are produced separately on Colab A100 and are to be merged by the operator.

## Vietnamese earnings-date acquisition (real dates, partial coverage)
Two independent sources were crawled; both keep the actual disclosure date, never a legal filing deadline.
- **SSC disclosure portal** (`congbothongtin.ssc.gov.vn`, Playwright headed Chromium, 411 pages): 3,872 records /
  381 tickers, but the portal retains only **2025-01-15 to 2026-09-10**. Comprehensive for that window.
- **vnstock/VCI news feed** (paginated `/v1/news`): 1,873 records / 218 tickers, **2016-09 to 2026-08**, sparse
  (median 2 dates/ticker; VN30 20/30, VN100 66/100). Publish-date validated as real: period-end to publish lag
  median 26 days (matches the ~30-day quarterly norm), not deadline-clustered, no future dates.
- **Combined** (`results/gamma_gbm/hose_earnings_combined.parquet`): 5,723 dates / 402 tickers, 2016-2026.
  Coverage is dense for 2025-2026 (about 380 tickers/fold) and thin before (24-35 tickers/fold).
Fetchers: `scripts/etl_vn_earnings/crawl_hose_playwright.py`, `fetch_vn_earnings_api.py` (raw data gitignored).

## Results

### S&P 500 (all metrics + oracle) — `results/gamma_gbm/paper_metrics_sp500.json`
- Graph null holds on every metric: correlation graph never beats the market aggregate on QLIKE/RMSE/MAE/R2;
  corr-vs-market DM p = 0.84/0.47/0.53/0.05 (h1/h5/h10/h22).
- Earnings is the lever on every metric: GBM+earn cuts QLIKE 9-11% and gives lowest RMSE/MAE, highest R2 at every
  horizon; DM GBM+earn vs GBM p=0.000. R2 h1 0.187 to 0.211.
- Oracle (illegal contemporaneous neighbour) +19.2/25.4/27.2/27.1%, p<0.001.

### HOSE (all metrics + DM, real earnings) — `results/gamma_gbm/paper_metrics_hose.json`
- Own-history GBM is the best legal model at every horizon (QLIKE 1.568/1.648/1.684/1.726, R2 0.198/0.133/0.107/
  0.073); HAR/HARQ trail (QLIKE ~1.81-1.84).
- Graph null replicates: corr-vs-market DM p=0.138/0.055/0.465/0.533; sector/placebo ~ GBM. h1 GBM+market QLIKE
  2.5126 is a thin-market spike artifact (one fold QLIKE 8.87 in the per-fold record), not a real result.
- **Earnings does not transfer to Vietnam.** With real announcement dates, GBM+earn ties or slightly worsens GBM:
  QLIKE 1.5730/1.6485/1.6862/1.7262 vs GBM 1.5679/1.6478/1.6842/1.7263; DM p=0.000/0.214/0.122/0.919. No gain even
  in the comprehensively covered 2025-2026 folds. The S&P 500 earnings lever is market-specific.
- Oracle +1.3/2.6/3.2/2.4% (p=0.009/0.000/0.000/0.012), smaller than S&P 500, same direction (contemporaneous).

## Paper state — `docs/paper/2026-09-11_sp500_paper_v2.md` (on GitHub master)
- Sections 1-8; 3,860 words; em-dash count 0.
- Section 3 adds Table 1c (all-metrics graph null). Section 4 adds Table 2b (all-metrics earnings). Section 3
  oracle now uses S&P 500 numbers.
- Section 7 (new) reports HOSE: Table 3 (all metrics), graph null replication, earnings no-transfer, oracle.
  Abstract, Introduction, and Limitations updated to localize the earnings lever to the S&P 500 and to state the
  VN earnings-coverage caveat.
- Commits: `79cec811` (SP500 all-metrics + oracle), `b00fa5bf` (HOSE section).

## Pending (operator, morning)
- Run the Colab A100 notebook `notebooks/train_gnnhar_sp500_colab.ipynb` (SP500 + HOSE) with SMOKE=False; it
  writes per-horizon JSONs with all 5 metrics + train/val/test fit evidence + fit_diagnostics + DM, and commits
  them to GitHub. The prior local GNNHAR run direction: GNNHAR does not beat own-history GBM and the graph hurts
  vs a no-graph control on QLIKE. Merge these into Section 5 as the eighth reimplemented method.

## Coverage limits stated honestly
- VN earnings pre-2025 are sparse; the no-transfer result is strongest where coverage is comprehensive
  (2025-2026) and is stated as "no incremental value under available real dates", not a full-history null.
- No dates were imputed; period-ends/deadlines were never substituted for announcement dates.

## Checks
- Probe scripts pass `ruff --select F`. Paper commits passed the pre-push quality gate (docs-only).
- Data-quality/P1-P6 on the crawled earnings: dates in [2000-07-20, 2026-09-11], no future dates, per-ticker
  sorted+deduped, verified as real disclosure dates. Full raw-ingestion Pandera gate not run on the earnings CSV
  (not OHLCV; sanity checks applied and reported).
