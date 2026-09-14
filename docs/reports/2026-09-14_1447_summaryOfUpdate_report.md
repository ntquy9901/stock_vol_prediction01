# S&P 500 GARCH degenerate-QLIKE fix — summary

## Problem
`run_garch.py sp500` produced degenerate pooled QLIKE: h1=5.899, h5=0.896, h10=6.015, h22=6.410
(~16x the sane SP500 scale; HAR_ref = 0.370/0.436/0.461/0.485 in the same file). HOSE GARCH was fine
(1.65/1.74/1.78/1.84). The h5-vs-rest inconsistency indicated a small number of degenerate ticker-folds
dominating the pooled mean.

## Root cause (candidate (c)+(d): near-IGARCH fits, not a scale/fallback bug)
Instrumented `run_garch(market='sp500')` per horizon, ranked ticker-folds by summed per-obs QLIKE.
A tiny number of ticker-folds carried ~90% of the total QLIKE mass:

- **CEG fold k=2 (dominant, ~90% of mass at h1/h10/h22):** `arch` returned `omega=6.8e-8, alpha=5.9e-10,
  beta=0.9946` on a short (358-obs), newly-listed series. omega~0 and alpha~0 make the conditional-variance
  filter collapse to ~0 and ignore returns: implied unconditional variance = 1.27e-9 = **1.86e-6 x the
  ticker's own sample return variance (6.8e-4)**. Forecasts ~1.3e-9 vs realized ~2.2e-4 -> ratio y/f ~170,000
  -> per-obs QLIKE ~22,000.
- **EXE / PSKY (explosion mirror image):** `phi = alpha+beta ~ 0.99999` (< 1, so it passed the old
  `0 < phi < 1` gate) with sizeable omega -> unconditional variance `omega/(1-phi)` = 1e4-4e8 -> multi-step
  forecast explodes -> `-log(y/f)` blows up (dominant at h5/h10/h22).

The old gate `PERSIST_LO(0) < phi < PERSIST_HI(1)` is too permissive: it admits near-unit-root fits whose
implied unconditional variance is either ~0 (omega~0 collapse) or astronomically large (phi~1 explosion).
Legitimate fits sit near the data (GL ratio=0.83, RF ratio=6.07). The discriminator that separates
degenerate from legitimate is **model-implied unconditional variance vs the ticker's own sample variance**.
HOSE was not visibly blown up only because its larger variance scale (~1.6) and the particular per-horizon
`n_train` windows kept the pooled mean down — the same degenerate fits exist there (30 at h1, ratios up to
1.2e14), they simply did not dominate.

## Fix (files/lines)
- `baselines/2026-09-13_garch/code/config.py`: new `VAR_RATIO_CAP = 1000.0` (single-source constant; 3 orders
  of magnitude — loose enough to keep a legitimate high-persistence fit, tight enough to catch the 1e5-1e14x
  pathologies).
- `baselines/2026-09-13_garch/code/garch_model.py`
  - `fit_params` (after the existing omega/phi gate): reject to fallback when the implied unconditional
    variance `(omega/(1-phi))/SCALE**2` is not within `[sample_var/CAP, sample_var*CAP]` (and requires
    `sample_var>0`, so no divide-by-zero). Catches both the omega~0 collapse and the phi~1 explosion.
  - `forecast` (ML path): `np.clip(f, fallback_var/CAP, fallback_var*CAP)` as a safety net for transient
    one-step/multi-step values on otherwise-accepted fits.
- The guard is anchored on each ticker's OWN train-window sample variance (causal, no leakage), applied
  identically to HOSE and SP500. Degenerate FITS become fallback (flat own-history sample variance); no rows
  are winsorized or deleted, and the estimability exclusion (`MIN_TRAIN_OBS`, dropped from every model) is
  unchanged, so HAR/GARCH/GJR stay on identical pooled rows.

## Results (before -> after)
SP500 pooled QLIKE (GARCH / GJR):
- h1: 5.899/5.896 -> **0.4649/0.4634**  (HAR 0.3697)
- h5: 0.896/5.951 -> **0.5077/0.5035**  (HAR 0.4362)
- h10: 6.015/6.009 -> **0.5218/0.5163** (HAR 0.4612)
- h22: 6.410/6.399 -> **0.5442/0.5366** (HAR 0.4850)

Now stable across horizons, monotone in horizon, in the sane range near the other models, GARCH modestly
worse than HAR (DM -12% to -16%, all p<0.001) — the expected classical-benchmark behaviour (HAR beats GARCH
on QLIKE). fit_diagnostics verdict = "ok" for all models/horizons (was "overfit" from the 16x test blow-up).
`n_excluded` unchanged (2424/2420/2539/2527); `n_fallback` rose (62->79 etc.) as degenerate fits are now
correctly rerouted. Top-12 ticker-folds now carry 1.7% of QLIKE mass (was 92%).

## HOSE (confirmation)
- OLD code re-run in this environment reproduces committed HOSE h1 EXACTLY (1.6539) -> arch is deterministic
  here, no environment drift.
- Fixed code: HOSE GARCH = 1.6563/1.7401/1.7823/1.8366 vs committed 1.6539/1.7397/1.7823/1.836x
  (Delta <= +0.0024, <= 0.14%). Materially unchanged; still sensible; still GARCH < HAR at h1-h10.
- The tiny shift = 30 genuinely-degenerate near-IGARCH HOSE fits at h1 (29 with phi~1.0, ratios 2.6e4-1.2e14;
  1 collapse ratio 7.9e-6) correctly rerouted to fallback. These give good one-step but explosive multi-step
  forecasts, so the guard is net-correct across horizons. The regenerated HOSE JSON is committed for
  code<->result reproducibility.

## Tests
`baselines/2026-09-13_garch/test/test_garch_model.py` (+6 tests): igarch collapse rejected, explosive
unconditional variance rejected, sane uncond accepted, zero-sample-variance rejected (no divide-by-zero),
forecast clip up (collapse) and down (explosion), plus `VAR_RATIO_CAP > 1` assertion.
- `pytest baselines/2026-09-13_garch/test/` -> 31 passed.
- Coverage (garch_model.py + config.py): C0 line 100%, C1 branch 100% (14/14 branches).
- `ruff check --select F` on changed files: clean.
- overfit-evidence gate on both JSONs: OK.

## Commands run
- `python run_garch.py sp500` (full, ~40 min local) -> regenerated `results/gamma_gbm/garch_sp500.json`
- `python run_garch.py hose` -> regenerated `results/gamma_gbm/garch_hose.json`
- `pytest`, `pytest --cov ... --cov-branch`, `ruff check --select F`, `check_overfit_evidence.py`

## Data-quality gate
N/A (no data change) — code + regenerated result JSONs only; no `data/` files touched.

## Risks / follow-ups
- `VAR_RATIO_CAP=1000` is a principled 3-orders-of-magnitude threshold; no legitimate fit observed near the
  boundary except one HOSE VHM fit at ratio 1.3e3 (phi~1.0, still degenerate). No cosmetic tuning to a target
  number was done.
- GJR at long horizons uses `phi=alpha+beta+gamma/2`; the same guard applies to it (verified GJR sane).
