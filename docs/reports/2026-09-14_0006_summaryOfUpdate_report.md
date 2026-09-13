# Summary of Update — HOSE full_compare (clean re-run) + GARCH + concat ablation

Date: 2026-09-14 00:06

## What changed
- **HOSE `full_compare` clean re-run** (`results/gamma_gbm/full_compare_hose.json`, all 4 horizons, n≈390k–399k).
  The previous run crashed mid-way under a shared-RAM OOM (3 concurrent heavy jobs); re-run with RAM free,
  per-horizon incremental checkpointing.
- **GARCH baseline** merged to master (`baselines/2026-09-13_garch`, commit `e9cd3554`).
- **GBM+earn+graph concat ablation** merged to master (`baselines/2026-09-13_gbm_earn_graph`, `4953d484`).
- **Architecture HTML** `docs/reports/2026-09-13_gbm_earn_corr_vs_graph_architecture.html` — visual of
  `GBM+earn+corr` (1 graph col) vs `GBM+earn+graph` (7 spillover cols).

## HOSE results (QLIKE, lower better; DM date-clustered)
| model | h1 | h5 | h10 | h22 |
|---|---|---|---|---|
| GBM (own-8) | 1.5702 | 1.6484 | 1.6840 | 1.7263 |
| GBM+earn | 1.5690 | 1.6489 | 1.6835 | 1.7261 |
| GBM+earn+corr | 1.5970 | 1.6487 | 1.6842 | 1.7261 |
| HAR | 1.8117 | 1.8218 | 1.8277 | 1.8355 |
| GBM+oracle (future-leak bound) | 1.5483 | 1.6041 | 1.6410 | 1.7736 |

DM p-values:
- GBM vs HAR: 1.3e-182 / 5.1e-77 / 2.2e-39 / 1.2e-23 — GBM beats HAR decisively at every horizon.
- GBM+earn vs GBM: 0.79 / 0.37 / 0.79 / 0.88 — earnings adds no significant HOSE value (real crawled dates, small effect).
- GBM+corr vs GBM: 0.70 / 0.53 / 0.60 / 0.39 — the correlation-graph feature adds no significant value.

Interpretation: own-history GBM is the HOSE champion; neither earnings nor graph carries deployable OOS signal.
`GBM+oracle` (a leakage upper bound that averages neighbours' realised future volatility) leads at h1–h10 but
loses to GBM at h22 — even oracle graph info is useless long-horizon.

## GARCH (separate baseline, per-stock, ~1.4% short-history rows excluded)
GARCH(1,1) beats plain HAR at short horizons (h1 +8.5% p=1.3e-159, h5 +4.3%, h10 +2.4%), ties h22; GJR adds
nothing. Ordering: GBM (1.57) < GARCH (1.65) < HAR (1.81). A fully-unified DM matrix including GARCH vs the GBM
models remains pending (needs per-obs GARCH predictions aligned to the full_compare panel rows).

## GBM+earn+graph concat — NO-GO ablation
= `OWN8 + EARN + SPILL_COLS` (7 graph-spillover cols) in one GBM, vs `+corr` which adds only `g_corr` (1 col).
Graph hurts at every horizon (h1 −3.06% DM p<1e-4, h5 −1.19%, h22 −0.25%), and overfits at h10 (QLIKE 3.79).
`err_corr(graph-only, GBM+earn)` = 0.96–0.98 → the spillover block carries no signal orthogonal to own history.

## Tests / gate
- GARCH: 24 tests, C0 line=100% / C1 branch=100%, ruff-F clean, 3-layer review — passed pre-push gate (pushed alone).
- concat: 12 tests, C0/C1=100%, review no critical/major — passed pre-push gate (pushed alone).
- Multi-baseline push gotcha: two new baselines each define `code/config.py` (bare `import config`) → collide in one
  pytest process (`sys.modules['config']` shared) + `test.conftest` package collision. Fix: push baselines one at a
  time so the changed-scope gate collects a single `test/` dir per push. No QG_SKIP used.

## Data-quality gate
N/A for the re-run itself (no data change — same enriched HOSE frames). Data-quality tests (334) + delivered-baseline
tests (69) ran green inside both baseline pushes.

## Follow-ups
- Unified DM matrix including GARCH (per-obs GARCH preds aligned to panel).
- SP500 via `notebooks/sp500_final_models_colab.ipynb` (resilient) → `full_compare_sp500.json` → 2-market tables.
- statistical-power (MDE/power) on the key DM comparisons.
