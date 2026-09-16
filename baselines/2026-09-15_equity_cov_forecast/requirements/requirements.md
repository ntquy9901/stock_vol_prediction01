# Requirements — Equity conditional-covariance forecasting (cross-stock structure test)

> SDD Specify phase. Baseline `2026-09-15_equity_cov_forecast`. Constitution = project `CLAUDE.md`.
> Motivation: 5 prior graph attempts on **per-stock variance** were NO-GO because own-history + a single
> market factor dominate. This baseline moves to a target where cross-stock structure is **intrinsic** — the
> covariance matrix Σ_t of a basket — and asks whether cross-stock structure helps forecast it OOS.

## 1. Objective
Forecast the conditional covariance matrix Σ_t of a basket of equities from **daily returns only** (no
intraday), and test whether cross-stock structure (factor / graph estimators) beats strong own/shrinkage
baselines, measured by the **out-of-sample global-minimum-variance-portfolio (GMV) realized volatility** —
the canonical daily-data covariance-forecast metric (Engle, Ledoit & Wolf 2019, *Large Dynamic Covariance
Matrices*, JBES; Engle & Colacito 2006).

## 2. Scope
- Markets: **SP500** (primary — source is split/dividend-adjusted, deep history) and **HOSE** (secondary,
  with the split-adjust caveat below). Data already present: `data/processed_enriched/{sp500_clean,hose}/*.csv`
  with a precomputed `daily_return` column.
- IN scope: covariance estimation + walk-forward GMV-portfolio evaluation, model ladder up to a graph model,
  DM significance, regime robustness.
- OUT of scope: intraday realized covariance (no data — see §7); options-implied; return *level* (direction)
  forecasting; trading-cost optimization beyond a turnover report.

## 3. Data & target
- **Returns**: `r_{i,t}` = existing `daily_return` (close-to-close). Winsorize per stock at the exchange
  price band / robust bound before use (HOSE ±7% band; SP500 robust ±k·MAD) to kill rare unadjusted-split
  jumps (measured: HOSE `|r|>0.5` ≈ 0.001%, beyond the ±7% limit → corporate-action artifact).
- **Panel**: aligned `dates × N` matrix; point-in-time membership per fold (a stock enters the basket only on
  dates it has data) to limit survivorship bias.
- **Target** (daily-data honest framing): there is no per-day realized covariance without intraday data;
  the *estimation target* is the **conditional covariance**, and it is judged **only by the economic OOS
  metric** below — never by fitting to a smoothed sample covariance in-sample.

## 4. Success metric (primary) + secondary
- **Primary**: OOS realized volatility of the GMV portfolio `w_t = Σ̂_t⁻¹1 / (1ᵀΣ̂_t⁻¹1)` (long-short) and a
  long-only variant (`w≥0`), computed from returns strictly after the estimation date, aggregated across
  rebalances → annualized. **Lower = better covariance forecast.**
- **Significance**: Diebold-Mariano / HAC test on the squared out-of-sample portfolio-return series between
  two models (the ELW method).
- **Secondary**: Frobenius / Stein loss vs the ex-post sample covariance of the holdout window; portfolio
  turnover (realism); condition number of Σ̂ (stability).

## 5. Success / Go–No-Go (pre-registered)
- **The real bar = beat Ledoit-Wolf shrinkage**, not sample covariance. Beating sample-cov is trivial.
- **GO (cross-stock structure helps)**: a factor and/or graph estimator lowers OOS GMV vol vs Ledoit-Wolf
  with DM significance at **≥2 rebalance horizons on ≥1 market**.
- **Partial (informative either way)**: if the **factor** model beats LW but the **graph** adds nothing on
  top of the factor → "cross-stock structure helps via a common factor; the network adds no incremental
  OOS value" — consistent with the per-stock findings, and publishable.
- **NO-GO**: no factor/graph model beats LW at any horizon → structure does not help even where it is
  intrinsic (strong negative result).

## 6. Acceptance criteria (Definition of Done, per CLAUDE.md)
- [ ] Walk-forward, leakage-safe (Σ̂_t uses only data `< t`; holdout strictly after t; point-in-time basket).
- [ ] Full model ladder run (sample, EWMA, Ledoit-Wolf, DCC, factor-rank-k, graph) on SP500 + HOSE.
- [ ] Primary metric (GMV OOS vol, long-short + long-only) + DM + secondary metrics, all reported.
- [ ] COVID / regime-spike robustness (per HOSE rule): re-evaluate excluding shock windows; verdict sign
      must hold.
- [ ] Tests pass (pytest), C0=100% / C1≥95% on changed lines; ≥1 smoke; ≥1 real-data-slice test.
- [ ] `/code-review` 3-layer + performance lens; findings resolved.
- [ ] Overfit/robustness evidence for any learned model (graph GNN) per the gate.
- [ ] Config single-sourced (window W, N, rebalance horizons, shrinkage intensity grid, seeds) in one module.
- [ ] Summary report + push.

## 7. [NEEDS CLARIFICATION] — resolved / open
- **Intraday data**: RESOLVED = none (daily OHLC only) → use the daily GMV-portfolio metric (ELW 2019),
  which requires no intraday. True realized covariance is out of scope.
- **HOSE split adjustment** (OPEN, caveat): VN prices are not split-adjusted (project memory). Mitigation =
  winsorize at the ±7% band; flag remaining risk in the paper. `[NEEDS CLARIFICATION]` if a split-adjusted
  VN price source is available, prefer it — otherwise SP500 is the primary, HOSE secondary-with-caveat.
- **Basket size N** (OPEN, design-defaulted): default N≈50–100 most-liquid, full-history names per market
  (invertibility vs realism trade-off); final N fixed in config after the liquidity/history screen.
