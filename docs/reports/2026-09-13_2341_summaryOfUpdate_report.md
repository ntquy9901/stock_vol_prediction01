# Summary — Hybrid feature-concat GBM (own+earn ⊕ graph-spillover), HOSE

Date: 2026-09-13. Baseline: `baselines/2026-09-13_gbm_earn_graph/`. Market run: HOSE (local). Not pushed.

## What was built
A hybrid per-stock gamma-GBM whose feature vector concatenates the own-history+earnings block
(`own8 = FM.OWN \ {rq}` + `FM.EARN`) with a **7-feature graph-spillover block** (`S1.GRAPH`:
`g_nb_vol, g_nb_shock, g_nb_max, g_nb_disp, g_node_minus_nb, g_nb_ret, g_nb_volshock`) aggregated over a
per-fold TRAIN-only correlation top-k graph. Compared, all horizons, against `GBM+earn` and
`GBM+earn+corr` (single-feature `g_corr`), with a residual error-correlation diagnostic vs a standalone
`graph_only` GBM. Follows SDD (requirements → design) → TDD → gate → adversarial review.

Files (all under `baselines/2026-09-13_gbm_earn_graph/`):
- `requirements/requirements.md`, `design/design.md` — spec + plan (architecture, causality argument, gates).
- `code/config.py` — tunables (horizons, min_rows, overfit ratio, success thresholds).
- `code/spillover_features.py` — causal spillover-block builder (read-only reuse of `FM.nb`/`S1.graph_feats`).
- `code/run_hybrid.py` — walk-forward driver; writes `results/gamma_gbm/gbm_earn_graph_hose.json`.
- `test/` — causality (future-invariance), aggregation-formula, fillna, no-mutation, end-to-end smoke,
  earnings-resolution branches, fail-loud guards, success-verdict branches.
- `code_review/code_review_2026-09-13.md` — adversarial review.

Shared modules (`FM`/`S1`/metrics/stats) imported read-only; no shared file modified.

## HOSE results (`results/gamma_gbm/gbm_earn_graph_hose.json`)

| h | n | GBM+earn | GBM+earn+corr | GBM+earn+graph | +graph vs +earn | DM p | +graph vs +corr | DM p | err_corr | fit(+graph) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1  | 399,040 | 1.5690 | 1.5970 | 1.6170 | **-3.06%** | 0.0000 | -1.25% | 0.268 | 0.961 | ok |
| 5  | 397,428 | 1.6489 | 1.6487 | 1.6685 | **-1.19%** | 0.0000 | -1.20% | 0.0000 | 0.965 | ok |
| 10 | 395,413 | 1.6835 | 1.6842 | 3.7866 | **-124.9%** | 0.224 | -124.8% | 0.224 | 0.969 | **overfit** |
| 22 | 390,577 | 1.7261 | 1.7261 | 1.7304 | **-0.25%** | 0.0010 | -0.25% | 0.0004 | 0.975 | ok |

QLIKE lower = better; a negative "gain" means the graph model is WORSE. `success=False`.

## Verdict — NO-GO (prior confirmed)
The richer graph-spillover block adds no out-of-sample value on HOSE and actively **hurts** the earnings
model at every horizon: -3.06% (h1, DM p<0.0001), -1.19% (h5, p<0.0001), -0.25% (h22, p=0.001), and a
catastrophic overfit blowup at h10 (QLIKE 3.79 vs 1.68, verdict=overfit — thin-market GBM instability with
the extra 7 features). It also fails to beat the single-feature `GBM+earn+corr` (worse and DM-significant at
h5/h22; not significant at h1; blowup at h10).

The residual error-correlation between `GBM+earn` and the standalone `graph_only` forecast is **0.96–0.98**
across horizons — the graph branch's errors are ~collinear with `GBM+earn`, so the block carries essentially
no orthogonal signal. This reproduces the prior `GNN ⊕ GBM+earn` ~0.98 finding: cross-stock spillover, at
least as trailing-correlation neighbour aggregates, is redundant with own-history+earnings on this market.

## Causality / leakage
Graph built from train rows only (`S1.build_graph(tr, ...)`, `tr.date < ts - embargo`); neighbour
aggregation uses only the day-t cross-section (target is h-ahead, separate); embargo `int(h*1.6)+5` days.
Future-invariance proven by `test_spillover_causal_future_invariant` (perturbing later-dated rows leaves
earlier-date features byte-identical). Same causal footing as the `g_corr` the paper already ships.

## Tests + coverage
`python -m pytest baselines/2026-09-13_gbm_earn_graph/test/` — 12 passed. Branch coverage on changed code
(`--cov-branch`, `.venv_gpu_encode`): line 100%, branch 100% (`run_hybrid.py` 86 stmts / 22 branch,
`spillover_features.py` 19 stmts / 4 branch, `config.py` 5 stmts — 0 missing). `main`/`_print`/path-bootstrap
lines carry `# pragma: no cover` per repo convention.

## Code review
Adversarial review (Blind Hunter / Edge Case Hunter / Acceptance Auditor + leakage + performance lenses):
no critical/major findings. Three minor observations accepted with rationale (M1 impossible-on-real-data
NaN err_corr; M2 this baseline's `GBM+earn` uses `own8` without `rq` per task spec, so its QLIKE is not
directly comparable to the paper's `GBM+earn`; M3 panel drops `rq`-NaN rows uniformly across models).
See `code_review/code_review_2026-09-13.md`.

## Lint / performance
`ruff check --select F` clean on code + tests. Neighbour aggregation is a vectorised `V @ W.T` matmul per
fold (no per-observation Python loop); GBM fits batch all fold rows via sklearn; only irreducible loops over
folds/horizons/models/seeds. Gamma-GBM is CPU-bound in the shared `FM.gbm` (GPU path would require editing a
shared module — out of scope).

## Data-quality gate
N/A (no data change) — the experiment consumes existing `data/processed_enriched/hose` and the existing
`results/gamma_gbm/hose_earnings_combined.parquet` read-only; it creates no data/features/manifest.

## SP500
Note only — not run locally (too heavy for the RTX 4060; `run_hybrid.py sp500` targets Colab, `min_rows`
30000). Prior evidence (GNN sweep) already indicates the same redundancy on SP500.

## Follow-ups
- None required; the experiment answered its question (NO-GO). The h10 overfit blowup is an additional data
  point on thin-market GBM instability when low-signal cross-sectional features are added.
