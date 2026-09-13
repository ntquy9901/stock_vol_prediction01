# Summary — Causal Diebold–Yilmaz spillover feature for the per-stock gamma-GBM (Experiment #1)

**Date:** 2026-09-13. **Baseline:** `baselines/2026-09-13_dy_spillover_feature/`.
**Verdict:** NO-GO on HOSE — the spillover feature reduces QLIKE accuracy at every horizon.

## What changed / was added
Tests whether the Diebold–Yilmaz (2012) volatility-spillover index, built **causally** (rolling trailing
window, no look-ahead), adds value to the per-stock gamma-GBM volatility forecast. Files:

| Path | Purpose |
|------|---------|
| `code/config.py` | All tunable constants (window/step/VAR-lag/GFEVD-H/sector threshold, success gate). |
| `code/dy_spillover.py` | Causal feature builder: sector-mean log-var panel → rolling VAR + generalized FEVD → `net_spillover`, `total_spillover`, broadcast onto per-stock panel. |
| `code/run_dy.py` | Walk-forward GBM(own) vs GBM(own+spillover): pooled QLIKE + date-clustered DM + fit diagnostics; writes `results/gamma_gbm/dy_spillover_<market>.json`. |
| `test/test_dy_spillover.py`, `test/test_run_dy.py` | 14 pytest tests (GFEVD sign, causal perturbation, degeneracy paths, run smoke). |
| `notebooks/dy_spillover_sp500_colab.ipynb` | Git-centric Colab runner for the SP500 run (user runs it). |
| `requirements/`, `design/`, `code_review/` | SDD spec, plan, adversarial review. |

## Feature (causal construction)
- `K` sector-mean log-variance series (sector with `>= 10` constituents). HOSE → **15 series**.
- Every `DY_STEP=5` days, fit `VAR(1)` on the trailing `DY_WINDOW=250` rows (dates `<= t` only), compute
  the generalized FEVD (Pesaran–Shin 1998 / Diebold–Yilmaz 2012), derive total spillover `S_t` and net
  directional `NET_{s,t}`; forward-fill between anchors. Per stock in sector `s`: `net_spillover=NET_{s,t}`,
  `total_spillover=S_t`.
- Leakage argument: the row-date-`t` value is a function only of vols dated `<= t`. Verified by
  `test_rolling_spillover_is_causal` (perturbing future rows leaves all values `<= cutoff` bit-identical)
  and `test_gfevd_sign_known_var` (B driven by lagged A ⇒ net A > 0 > net B).

## HOSE results (real run — `results/gamma_gbm/dy_spillover_hose.json`)
15 sector series, 1270 VAR windows (329 failed on thin-market windows, ffilled).

| h | n | GBM(own) QLIKE | GBM+spillover QLIKE | gain % | DM p | fit |
|---|---|---|---|---|---|---|
| 1 | 399,040 | 1.5679 | 1.7680 | **−12.76%** | 0.0718 | ok/ok |
| 5 | 397,428 | 1.6478 | 1.8169 | **−10.27%** | 0.0682 | ok/ok |
| 10 | 395,413 | 1.6842 | 1.7287 | **−2.64%** | 0.000496 | ok/ok |
| 22 | 390,577 | 1.7263 | 1.9115 | **−10.73%** | 0.0216 | ok/ok |

The spillover feature **hurts** at all four horizons. h10 and h22 are DM-significant but in the *wrong*
direction (spillover worse); h1/h5 are worse but not DM-significant. `success=False`. No overfit: train
QLIKE > test QLIKE for both models (verdicts `ok`), so this is a genuine no-lift-plus-noise result — the
GBM overweights the spillover signal and generalises worse, not an estimation artifact. Consistent with the
~15–20% prior. **Honesty note:** at `n ~ 400k` a `|gain| ~ 0.04%` "DM-significant" would NOT be a real win;
here the gains are large-magnitude and negative, so the direction is unambiguous.

SP500 not run locally (too slow) — via the committed Colab notebook, to be run by the user.

## Tests + coverage
- `python -m pytest baselines/2026-09-13_dy_spillover_feature/test` → **14 passed**.
- `--cov --cov-branch` on `code/`: **C0 line 100%, C1 branch 100%** (config 14/14, dy_spillover 92 stmts /
  20 branches, run_dy 63 stmts / 16 branches — 0 miss). New files, so all lines are changed lines.
- `ruff check --select F` → clean. Config-hardcode scan → 0 BLOCK (1 WARN on `DY_VAR_MIN_STD=1e-9`, which
  is the constant's canonical home in the config module — expected, not actionable).

## Code review
Adversarial review focused on VAR/GFEVD-window leakage and sector-mean aggregation. Result + actions
recorded in `baselines/2026-09-13_dy_spillover_feature/code_review/code_review_2026-09-13.md`.

## Performance
Hot path = the walk-forward seed-ensemble HistGradientBoosting (OpenMP multi-core), identical to the
delivered SP500/VN champion — no per-item batch=1 training. The rolling VAR runs once over ~1270 anchors
(cheap `12×250` fits), negligible vs the GBM. No GPU applies (scikit-learn / statsmodels CPU). HOSE run
wall-time ~4 min on the local box.

## Data-quality gate
N/A for new data (no data crawled/appended). The experiment reads the existing, already-gated
`data/processed_enriched/hose` frames (junctioned into the worktree for the local run; the junction is
gitignored data, not committed).

## Risks / follow-ups
- SP500 run pending (Colab). Prior + HOSE both point NO-GO; SP500 will confirm or not.
- 329/1270 HOSE windows failed the VAR (thin-market / NaN-heavy sectors) and were ffilled — documented; a
  larger `DY_STEP` or `DY_SECTOR_MIN_STOCKS` would trade resolution for stability but is unlikely to flip a
  −10% result into a win.

## DoD checklist
- [x] 5 baseline subfolders present. [x] requirements + design with success/go-no-go.
- [x] Code runs (HOSE real run, JSON written). [x] Tests pass, C0=100%/C1=100% on changed lines.
- [x] ruff F clean, config-hardcode 0 BLOCK. [x] Adversarial code review done + documented.
- [x] Colab notebook committed + gitignore allowlist. [ ] Not pushed (per instruction: commit to worktree
      branch for review only).
