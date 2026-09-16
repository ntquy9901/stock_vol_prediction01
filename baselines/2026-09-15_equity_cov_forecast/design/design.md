# Design + Plan — Equity conditional-covariance forecasting

> SDD Plan phase. Implements `requirements.md`. Status: DESIGN (no code yet). Passes the 3 SDD gates (§6).
> Core idea: test cross-stock structure where it is **intrinsic to the target** (the covariance matrix),
> judged by the economic OOS metric that needs no intraday data (GMV-portfolio out-of-sample variance).

## 1. Why this can win where per-stock variance could not
Per-stock variance forecasting is dominated by own-history AR + a single market factor, so an N×N graph is
redundant (proven, 5×). A **covariance matrix** has O(N²) off-diagonal entries that own-history of one stock
cannot produce — cross-stock structure is the target itself. The open question is only whether that structure
is **forecastable OOS** better than shrinkage. That is a genuine, unsettled question (ELW 2019 show nonlinear
shrinkage and DCC each win in different regimes) — so the result is informative whichever way it lands.

## 2. Data flow
```
processed_enriched/<market>/*.csv (daily_return)
   → winsorize per stock at band (HOSE ±7%, SP500 robust ±k·MAD)     [split-artifact guard]
   → aligned panel R [T × N], point-in-time membership per fold
   → walk-forward over rebalance dates t (expanding/rolling window W):
        for each model m:  Σ̂_t^m  from R[:t]  only
        w_t^m = GMV weights (long-short + long-only)
        hold h days → realized portfolio returns r_p[t+1 : t+h] = R[t+1:t+h] @ w_t^m
   → pool r_p across rebalances per model
   → metric: annualized OOS vol(r_p); DM on r_p²; Frobenius/Stein vs ex-post cov; turnover; cond(Σ̂)
```
No target `shift(-h)` leakage risk (covariance, not a shifted scalar); the only guard is **holdout strictly
after t** and **Σ̂_t from data < t**.

## 3. Model ladder (ordered — lesson-applied: NO learned latent embedding first)
| # | Model | Cross-stock? | Learned? | Role |
|---|---|---|---|---|
| 1 | Sample covariance (rolling W) | full, unregularized | no | floor (noisy, may be singular if N≳W) |
| 2 | EWMA / RiskMetrics cov (λ) | full | no | cheap dynamic baseline |
| 3 | **Ledoit-Wolf linear shrinkage** (+ nonlinear NLS if feasible) | shrunk | no | **the bar to beat** |
| 4 | DCC-GARCH (Engle 2002) | dynamic corr | econometric | medium-N SOTA |
| 5 | **Factor rank-k** (market+sector, or PCA/POET Fan-Liao-Mincheva 2013): `Σ=BFBᵀ+D` | low-rank | linear | "graph as a low-rank prior" |
| 6 | **Graph estimator** (falsification target): shrink correlation toward a graph-structured target, or a GNN that outputs correlation adjustments end-to-end | network | yes (GNN) | the incremental-network test |

Stop-early rule: if #5 (factor) does not beat #3 (LW), a heavier #6 graph is very unlikely to — report and
stop (don't chase). If #5 beats #3, #6 tests whether the *network* beats the *factor*.

## 4. Architecture decisions (avoid the basis-drift trap we just proved)
- **Do NOT** stitch embeddings from separately-trained GNNs into a downstream model (the OOF-neural-feature
  design produced a 9σ basis drift → 4 observations blew up QLIKE by 100% of the fold's error). If a GNN is
  used at #6, it must be **end-to-end** (one model produces Σ̂ directly) so there is no cross-model latent
  basis mismatch, and its output is a **correlation matrix** (bounded, PSD-projected) not an unbounded latent.
- **PSD + invertibility**: every Σ̂ is projected to PSD and ridged (`Σ̂ + εI`) so GMV weights are well-defined;
  condition number is reported. Shrinkage (#3) handles N≳W; sample cov (#1) may be singular and is expected to.
- **Graph target for #6**: correlation-of-returns kNN or sector block — same families already tried, reused.

## 5. SDD gates
- **Simplicity**: one panel builder + one walk-forward evaluator + a dict of estimator fns; reuse existing
  screen/DM/robustness utilities. No new framework.
- **Anti-Abstraction**: use `sklearn`/`scipy`/`statsmodels` (Ledoit-Wolf = `sklearn.covariance.LedoitWolf`;
  DCC via a thin `arch`/custom); reuse project `stats.date_clustered_dm`, metrics, spike-window utils.
- **Performance/Batching**: covariance ops are small matrix algebra — batch estimator evaluation across
  rebalance dates; vectorize GMV over folds; GPU only for the #6 GNN (batched over dates, no batch=1). N≈50–100
  → matrices tiny; the walk-forward loop is the cost, parallelize estimators per fold.

## 6. Files (planned)
- `code/cov_config.py` — single-source tunables: `N`, window `W`, `rebalance_horizons` (e.g. 5/10/22),
  shrinkage grid, EWMA λ, factor `k`, GNN epochs/lr/patience/seeds, spike windows, winsor band, ridge ε.
- `code/panel.py` — load `daily_return`, winsorize, point-in-time aligned panel, liquidity/history screen.
- `code/estimators.py` — `sample_cov`, `ewma_cov`, `ledoit_wolf`, `dcc`, `factor_rank_k`, `graph_cov` — each
  `R_train → Σ̂` (PSD-projected, ridged).
- `code/evaluate.py` — walk-forward GMV eval: weights (long-short + long-only), OOS portfolio returns, pooled
  annualized vol, DM on r_p², Frobenius/Stein, turnover, cond number, spike robustness. Atomic checkpoint.
- `code/run_cov.py` — driver: `python run_cov.py [sp500|hose]` → `results/cov_forecast/cov_<market>.json`.
- `test/` — panel alignment + winsor; PSD/invertibility of each Σ̂; **GMV-weight causality** (Σ̂_t invariant to
  future rows); a known-answer GMV vol on a synthetic 2-asset case; DM plumbing; real-data-slice smoke; #6
  overfit evidence.

## 7. Metric detail (leakage-safe, ELW 2019)
For rebalance dates `t_1<…<t_R` spaced by `h`:
`r_p^m(τ) = Rᵀ(τ)·w_{t_j}^m` for `τ ∈ (t_j, t_{j+1}]`, using weights fixed at `t_j` from `Σ̂_{t_j}` (data <t_j).
Pool all `r_p^m(τ)` → `AnnVol_m = √252 · std(r_p^m)`. Best model = lowest AnnVol. DM on `(r_p^A)² − (r_p^B)²`
with HAC/HLN correction. Long-only repeats with `w≥0, Σw=1` (QP). Turnover = `Σ|w_{t_{j+1}}−w_{t_j}|`.

## 8. Robustness (per HOSE spike rule)
Recompute AnnVol + DM **excluding** COVID (2020-02–04), 2022, and Apr-2025-tariff windows; the winning sign
must hold. Report per-regime GMV vol and a storm-decile breakdown. HOSE additionally: sensitivity to the
winsor band (±7% vs robust) since unadjusted splits could distort correlations.

## 9. Validity caveats (state in report/paper)
- Daily-data "conditional covariance", **not** intraday realized covariance — named honestly (per the
  named-estimator rule).
- HOSE not split-adjusted → residual correlation distortion after winsorization; SP500 is the clean primary.
- GMV long-short can take large offsetting positions; the long-only variant is the realistic economic read.
- Survivorship: point-in-time membership mitigates but VN delisting history may be incomplete.

## 10. Plan (numbered tasks, each verifiable)
1. `panel.py` + winsor + screen → verify: aligned panel shape, no NaN, band caps applied (test). 
2. `estimators.py` #1–#3 (sample/EWMA/LW) → verify: PSD, invertible, LW matches `sklearn` on a slice (test).
3. `evaluate.py` walk-forward GMV + DM → verify: causality test + synthetic known-answer GMV vol.
4. Run ladder #1–#3 on SP500 → verify: LW beats sample (sanity); record baseline AnnVol.
5. Add #4 DCC + #5 factor-rank-k → verify: factor vs LW = the first real question; DM.
6. If factor beats LW: add #6 graph (end-to-end, PSD, bounded) + overfit evidence → verify: DM vs factor.
7. HOSE repeat + spike robustness + winsor sensitivity.
8. `/code-review` + summary report + push.

## 12. Combining with the champion GBM (if a correlation model wins) — DCC decomposition
GBM and a covariance model forecast **different objects and do not compete — they compose**. GBM forecasts
**per-stock variance** σ̂²_i = the **diagonal** of Σ (own-history, the project's strongest lever); a covariance
model's only defensible value-add is the **correlation matrix R** = the **off-diagonals** (the cross-stock part
GBM structurally cannot produce). Combine via the DCC/DECO decomposition (Engle 2002):
```
Σ̂_t = D̂_t · R̂_t · D̂_t ,   D̂_t = diag(σ̂_{GBM,i})  (GBM marginals) ,   R̂_t = correlation model
```
This is the RIGHT "graph + GBM" combination — the opposite of the 5× failed "graph feature → GBM predicts a
scalar variance". Here GBM owns the marginals (its strength), and the factor/graph only has to win on
**correlations** (a narrower, better-posed task). Add these as extra ladder rungs, all sharing the GBM diagonal:
`Σ̂ = D_GBM · R · D_GBM` with `R ∈ {sample-corr, LW-corr, factor-corr, graph-corr}`. The metric is unchanged
(GMV OOS vol); this isolates *which correlation estimator* helps once the diagonal is fixed at the GBM forecast.
Implementation note: `estimators.py` already returns Σ̂; a thin `with_gbm_diagonal(Σ̂, σ̂_gbm)` rescales any Σ̂ to
`D_GBM R D_GBM` (extract R from Σ̂, re-scale by GBM vols), so the GBM-composed variants reuse every estimator
and the same eval harness. GBM diagonal loaded from the existing gamma-GBM per-stock h=1 variance forecast.

## 11. How this closes the thesis narrative (equity + the crypto companion)
This baseline supplies the equity half of the 3-regime contrast: (i) crypto (own-history vs BTC-spillover —
where cross-asset should help), (ii) **equity covariance (here) — where cross-stock structure is intrinsic**,
(iii) equity per-stock variance (done, 5× NO-GO — where own-history dominates). Together they answer *when*
inter-asset relationships carry OOS volatility-forecast value, rather than a single isolated negative.
