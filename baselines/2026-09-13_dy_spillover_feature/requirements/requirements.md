# Requirements — Diebold–Yilmaz volatility-spillover feature for the per-stock gamma-GBM

**Date:** 2026-09-13
**Experiment #1** of the complex-network follow-up. Tests whether the Diebold–Yilmaz (2012)
volatility-spillover index, built **causally** (no look-ahead), adds incremental forecasting value to
the per-stock gamma-GBM volatility model.

## Goal
The reference complex-network paper estimated its spillover/topology graph using data from *within the
test period* (look-ahead). This experiment reproduces the spillover idea **without that flaw**: build a
rolling-window Diebold–Yilmaz spillover series that at each date `t` uses only volatility data dated
`<= t`, broadcast it onto the per-stock panel, and test it as an exogenous GBM feature under the
project's standard walk-forward + QLIKE + Diebold–Mariano protocol.

## Input
- Per-stock enriched frames from `FM.load(market)` (`data/processed_enriched/<market>/*.csv`): columns
  `date`, `parkinson_variance`, own-history block `FM.OWN` (9 features), `sector` (ICB code for VN via
  `S1.SECT`, GICS sector string for SP500 via `results/gamma_gbm/sp500_sectors.json`).
- No raw OHLCV and no future data are used.

## Output
- `results/gamma_gbm/dy_spillover_<market>.json` — per horizon `h in {1,5,10,22}`:
  `n`, `GBM` (own QLIKE), `GBM+spillover` QLIKE, `gain_pct`, `dm_p` (date-clustered DM of spillover vs
  own), `train_metrics`, `test_metrics`, `fit_diagnostics` (per-model overfit verdict), plus a
  `success` flag against the criterion below and a `diag` block (n sectors, n VAR windows, n failed
  windows).

## Feature definition (CAUSAL)
1. Sector aggregate series: for each ICB/GICS sector with `>= DY_SECTOR_MIN_STOCKS` constituents, the
   daily series = `log(max(mean_i parkinson_variance_i, floor))` over that sector's stocks present that
   day. `K` well-conditioned series (HOSE: ~12–15; SP500: ~11).
2. Rolling window (`DY_WINDOW` trading days, re-estimated every `DY_STEP` days): fit a `VAR(DY_VAR_LAG)`
   on the trailing window (data dated `<= t` only), compute the generalized FEVD (Pesaran–Shin 1998 /
   Diebold–Yilmaz 2012), derive the **total spillover index** `S_t` and the **net directional spillover
   TO each sector** `NET_{s,t}`. Values between anchors are forward-filled (last known). All causal.
3. Per-stock feature at `(sector s, date t)`: `net_spillover = NET_{s,t}`, `total_spillover = S_t`.
   Stocks whose sector is not a VAR series get `NaN` (handled natively by HistGradientBoosting).

## Models / evaluation
- `GBM(own) = FM.OWN` vs `GBM(own+spillover) = FM.OWN + [net_spillover, total_spillover]`.
- Expanding walk-forward over `S1.FOLDS`, `L`-day embargo `int(h*1.6)+5`, seed-averaged predictions
  (`FM.SEEDS`), pooled per-observation QLIKE, date-clustered Diebold–Mariano, train/test QLIKE for the
  fit-diagnostics verdict. Identical protocol to
  `baselines/2026-09-12_complex_network/code/verify_index_vol_feature.py`.

## Success criterion / go–no-go
`success = True` only if, on QLIKE, GBM+spillover beats GBM(own) with **DM p < 0.05 AND** an
economically non-trivial, sign-consistent `|gain_pct| >= 0.3%` across horizons **AND** no overfit
verdict. Explicit warning recorded: with `n ~ 400k` pooled observations a `|gain| ~ 0.04%`
"DM-significant" result is NOT a real win. Prior expectation is **NO-GO (~15–20% chance of a win)**;
the experiment reports whatever the DM says without spin.

## Acceptance
- [ ] Causal spillover feature: value at row-date `t` computable from vols dated `<= t` only
      (verified by a perturbation test).
- [ ] GFEVD sign correct on a known synthetic VAR (B driven by lagged A ⇒ net A→B positive).
- [ ] HOSE run produces a real result JSON with per-horizon QLIKE + DM.
- [ ] SP500 via committed Colab notebook (not run locally).
- [ ] Tests: C0 line = 100%, C1 branch >= 95% on changed lines. Adversarial code review, HIGH/MEDIUM
      fixed.
