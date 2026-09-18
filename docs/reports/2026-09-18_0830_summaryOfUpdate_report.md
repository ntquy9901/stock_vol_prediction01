# Summary of update — DTW self-similarity feature -> GBME (HOSE)

Date: 2026-09-18. Baseline: `baselines/2026-09-18_dtw_feature/`.

## What was built
A new HOSE baseline testing the transferable germ of Nakagawa & Yoshida (2022) "Time-series gradient
boosting tree" — DTW time-series shape matching — adapted to the variance-QLIKE thesis as a **strictly
causal feature** (NOT a new tree split criterion). For each stock-day, a Sakoe-Chiba banded DTW distance
is computed between the stock's trailing window of standardized log-variance and per-band centroid
templates built from that stock's OWN further-past (train) history; the distances are appended to the
champion own-history gamma-GBM (GBME = OWN-8 + real crawled VN earnings).

Arms: **GBME** (base), **GBME+dtw** (real templates + real trailing window), **GBME+dtw_placebo**
(same templates, date-shifted trailing window — the mandatory seasonal-artifact control per the
2026-09-10 lesson). `dtaidistance`/`tslearn` are not installed, so the banded DTW is a small
numpy DP vectorized over the query batch.

## Files (path -> purpose)
- `baselines/2026-09-18_dtw_feature/requirements/requirements.md` — spec, arms, pre-registered kill.
- `.../design/design.md` — data flow, causality guarantees, gates.
- `.../code/dtw_config.py` — single-source tunables (W, band, template bands, placebo shift, kill, spikes).
- `.../code/dtw_feature.py` — banded DTW, causal window/template builders, per-fold feature frame.
- `.../code/run_dtw_feature.py` — walk-forward driver (reuses `full_matrix` champion GBM, `S1` folds,
  `stats`/`metrics`/`overfit_check`); one JSON per horizon, atomic per-fold checkpoint; `--horizon` for
  per-horizon subprocess memory isolation.
- `.../test/test_dtw_feature.py` (+ conftest) — 23 tests.
- `.../code_review/code_review_2026-09-18.md` — adversarial review + empirical verdict.
- `results/gamma_gbm/dtw_hose_h{1,5,10,22}.json` — results (each carries metrics/train/val/fit + DM +
  per-fold + spike robustness).

## Tests + coverage
`.venv_gpu_encode -m pytest` : **23 passed**. Diff-coverage on changed modules (`--cov-branch`):
C0 line = **100%**, C1 branch = **100%** (>=95%) on `dtw_config.py`, `dtw_feature.py`, `run_dtw_feature.py`
(entry `main()` + path bootstrap marked `# pragma: no cover`). Covers: banded-DTW (identical->0, known
pair -> sqrt(3), batch, bad-shape raise), causal windows/templates (front-pad, empty-band else, zero-std
guard, missing-ticker / short-history NaN, FUTURE-perturbation causality), placebo != real on a trend,
DM/verdict/success/spike-mask logic, `_load_earn` (3 paths), and run smokes (hose per-fold+spike, sp500
no-spike, empty-fold skip, all-in-spike omit, horizons override).

## Result — HOSE, all horizons, n_folds==8 VERIFIED (each JSON)
| h | n_folds | GBME QLIKE | GBME+dtw QLIKE | placebo QLIKE | dtw vs GBME | DM p | beats | placebo vs GBME | dtw fit |
|---|---|---|---|---|---|---|---|---|---|
| 1  | 8 | 1.5719 | 1.9779 | 1.5925 | -25.83% | 0.297 | False | -1.31% (p=0.002) | overfit |
| 5  | 8 | 1.6479 | 3.8602 | 1.6549 | -134.25% | 0.221 | False | -0.43% (p=0.003) | overfit |
| 10 | 8 | 1.6856 | 4.7757 | 1.6899 | -183.33% | 0.225 | False | -0.26% (p=0.003) | overfit |
| 22 | 8 | 1.7270 | 4.4743 | 1.7321 | -159.07% | 0.242 | False | -0.30% (p=0.014) | overfit |

Spike-robust (ex-COVID/2022/Apr-2025): GBME+dtw still worse at every horizon (never beats). Pre-registered
SUCCESS = **False** (0 of 4 horizons beat; kill needed >=2, spike-robust, placebo-clean).

## Verdict: NO-GO
The DTW self-similarity feature does not add QLIKE-relevant information beyond HAR/own-history; it
sharply degrades QLIKE and **overfits at every horizon**. The near-neutral date-shifted placebo confirms
the harm is specific to the real feature, not a generic extra-column artifact. This matches the prior
belief: DTW encodes path *shape*, but QLIKE penalises *magnitude*, and vol level is already captured by
the own-history features. Consistent with the exhausted-lever record for non-magnitude signals on HOSE.

## Code review
Adversarial 3-lens review done (`code_review/code_review_2026-09-18.md`): causality (train-only
templates/scaling, past-only windows, future-perturbation test), no silent zero-fill (NaN + native HGBR
NaN handling), valid placebo, vectorized DTW (no batch=1), all tunables in `dtw_config.py` (0 BLOCK on
config-hardcode whole-file scan; `ruff --select F` clean). No CRITICAL/MAJOR open.

## Commands run
- `pytest test/` (23 passed); `pytest --cov=... --cov-branch` (100%/100%).
- `ruff check --select F`, config-hardcode whole-file scan (0 BLOCK).
- `run_dtw_feature.py hose` (h1) + `--horizon {5,10,22}` (per-horizon subprocess for memory isolation).
- `check_overfit_evidence.py` on the 4 real JSONs -> OK (deterministic-boosting result, gate skips).

## Data-quality gate
N/A (no data change): reuses existing `data/processed_enriched/hose` + `results/gamma_gbm/
hose_earnings_combined.parquet`; no crawl/reprocess.

## Performance
DTW vectorized over the query batch (no per-row Python DTW loop); GBM seed-ensemble reuses the champion
HGBR. Per-horizon subprocess isolation resolved a late-fold host-memory exhaustion (one horizon fits in a
fresh process; all 4 completed with 8 folds).

## Notes / follow-ups
- Model names avoid neural tokens, so the over/under-fit pre-push gate classifies the result as a
  non-learned (deterministic-boosting) file and skips it; fit evidence is still carried and reported
  (dtw = overfit is disclosed above, not hidden).
- No git operations performed (coordinator pushes). `archive/` out of scope.
