# Design — Diebold–Yilmaz spillover feature for the per-stock gamma-GBM

**Date:** 2026-09-13. Plan (SDD §5) for the requirements in `../requirements/requirements.md`.

## 1. Data flow
```
FM.load(market) ─► frames{ticker: df[date, parkinson_variance, FM.OWN..., sector]}
        │
        ▼  dy_spillover.sector_logvar_panel(frames, min_stocks)
   panel: DataFrame index=date, columns=qualifying sector codes
        │  value = log(max(mean_i parkinson_variance_i over sector's present stocks, FL))
        ▼  dy_spillover.rolling_spillover(panel, WINDOW, STEP, LAG, H, MIN_SECTORS, WIN_MIN_FRAC)
   net_df: DataFrame index=date, columns=sector codes  (NET_{s,t}, causal, ffilled)
   total : Series   index=date                          (S_t,       causal, ffilled)
        │
        ▼  dy_spillover.merge_spillover(frames, net_df, total)
   frames{ticker: df + [net_spillover, total_spillover]}   (broadcast by (sector,date) / date)
        │
        ▼  run_dy.run_dy(market)   (walk-forward, identical to verify_index_vol_feature.py)
   results/gamma_gbm/dy_spillover_<market>.json
```

## 2. VAR + generalized FEVD math (Diebold–Yilmaz 2012, JoE 182(1); Pesaran–Shin 1998, Econ. Lett. 58)
Reduced-form `VAR(p)`: `y_t = Σ_{l=1}^p Φ_l y_{t-l} + u_t`, `u_t ~ (0, Σ)`. Moving-average form
`y_t = Σ_{h>=0} A_h u_{t-h}` with `A_0 = I`, `A_h = Σ_l Φ_l A_{h-l}` (statsmodels `VARResults.ma_rep`).

**Generalized H-step FEVD** (order-invariant; does not use a Cholesky ordering):
```
              σ_jj^{-1} · Σ_{h=0}^{H-1} ( e_i' A_h Σ e_j )^2
θ_ij(H) = ───────────────────────────────────────────────────
                 Σ_{h=0}^{H-1} e_i' A_h Σ A_h' e_i
```
`σ_jj = Σ[j,j]`, `e_i` the i-th unit vector. Rows do not sum to 1 (shocks are not orthogonal), so
row-normalise: `θ̃_ij = θ_ij / Σ_k θ_ik`.

**Spillover scalars** (`K` = number of series):
- Total spillover index `S = 100 · (Σ_{i≠j} θ̃_ij) / K = 100 · (1 − trace(θ̃)/K)`.
- Directional TO others, from series j: `TO_j = 100 · (Σ_i θ̃_ij − θ̃_jj) / K` (column j, off-diagonal).
- Directional FROM others, to series i: `FROM_i = 100 · (Σ_j θ̃_ij − θ̃_ii)/K = 100·(1 − θ̃_ii)/K`.
- **Net** `NET_j = TO_j − FROM_j` (positive ⇒ net transmitter of volatility).

`gfevd(ma, sigma)` returns `θ̃`; `spillover_scalars(θ̃)` returns `(S, net_vector)`. Both pure/testable.

## 3. Causal rolling construction (the crux — no look-ahead)
- Anchors = every `DY_STEP`-th date of the panel. Each anchor `t` uses the trailing window
  `panel.iloc[pos-WINDOW+1 : pos+1]` — rows dated `<= t` **only**. This is the sole source of
  look-ahead risk in the reference paper, and is eliminated by the inclusive-trailing window.
- A window is scored only if it has `>= WINDOW·WIN_MIN_FRAC` rows after dropping rows with any NaN and
  `>= DY_MIN_SECTORS` columns with non-zero variance; otherwise the anchor yields `NaN`.
- Anchor values are `reindex`-ed onto the full date axis and **forward-filled**: date `d` receives the
  spillover from the most recent anchor `<= d` (still causal: anchor `<= d`, anchor's window `<= anchor`).
- `merge_spillover` assigns `NET_{s,t}` by `(sector, date)` and `S_t` by `date`. A ticker whose sector
  is not a VAR series gets `NaN` (not 0 — no silent neutral fill; the GBM handles NaN natively).
- **Leakage argument:** the feature at row-date `t` is a function only of `{parkinson_variance_{i,τ} :
  τ <= t}`. Perturbing any vol dated `> t` cannot change it (unit-tested in
  `test_dy_spillover.py::test_feature_is_causal`).

## 4. Feature list
| feature           | meaning                                             | source                |
|-------------------|-----------------------------------------------------|-----------------------|
| `net_spillover`   | `NET_{s,t}` net directional spillover TO stock's sector | trailing-window GFEVD |
| `total_spillover` | `S_t` system-wide total spillover index             | trailing-window GFEVD |

## 5. Design decisions
- **Sector series only, no separate market series.** A market-mean series is a linear combination of the
  sector means ⇒ near-collinear ⇒ singular `Σ`. The common market factor is already spanned by the
  sector series, so the total spillover index captures market co-movement. (Requirements listed the
  market series as *optional*; excluded for VAR conditioning — Simplicity/Anti-Abstraction gate.)
- **Few, well-conditioned series.** `DY_SECTOR_MIN_STOCKS = 10` keeps `K ~ 12` (HOSE) rather than a
  400-node VAR; a 400-variable VAR on a 250-day window is hopelessly rank-deficient (the reference
  paper's instability). `DY_VAR_LAG = 1` keeps `K·p + 1` params per equation small vs `WINDOW`.
- **Fail-soft per window, fail-loud on the whole series.** A single ill-conditioned window ⇒ `NaN` +
  ffill (standard for a rolling estimator, documented). But `run_dy` asserts the merged feature is not
  entirely NaN for the panel (guards the idxvol-style silent all-zero/all-NaN bug per CLAUDE.md
  "no silent degradation").
- **Reuse, no re-implementation:** QLIKE (`M.per_obs_qlike`), DM (`ST.date_clustered_dm`), gamma-GBM
  (`FM.gbm`), panel/target (`FM.panel`), folds/floor (`S1`, `FM.FL`) are imported, not copied. VAR +
  `ma_rep` from statsmodels (already installed, 0.14.6).

## 6. SDD gates
- **Simplicity Gate:** PASS — two scalar features, one VAR system, no new model. No market-series flag.
- **Anti-Abstraction Gate:** PASS — statsmodels `VAR`/`ma_rep` used directly; project QLIKE/DM/GBM reused.
- **Performance/Batching Gate:** the hot loop is the walk-forward GBM (`FM.gbm`, HistGradientBoosting =
  multi-core OpenMP, seed-ensemble) — identical to the delivered SP500/VN champion; no per-item Python
  batch=1 training. The rolling VAR runs once over `~N/STEP` anchors (`~580` for HOSE), each a cheap
  `K x WINDOW` fit — negligible vs the GBM. No GPU applies (scikit-learn/statsmodels CPU). PASS.

## 7. Tasks (each with a verify)
1. `config.py` constants → verify: imported, no hardcoded literals in pipeline modules.
2. `gfevd` + `spillover_scalars` → verify: `test_gfevd_sign_known_var` (B←lagA ⇒ NET_A>0>NET_B).
3. `sector_logvar_panel` → verify: qualifying-sector count + log-mean value on a fixture.
4. `rolling_spillover` + `merge_spillover` → verify: causal perturbation test + broadcast-by-sector test
   + a degenerate-window (all-NaN → ffill) branch test.
5. `run_dy` → verify: smoke on synthetic frames with a valid scored fold (result schema + JSON-serialisable)
   and an all-folds-skip empty path.
6. HOSE real run → verify: JSON written, per-horizon QLIKE + DM reported.
7. Colab notebook for SP500 → verify: committed + gitignore allowlist entry.
