# Adversarial code review — hybrid GBM (own+earn ⊕ graph-spillover)

Date: 2026-09-13. Scope: `code/config.py`, `code/spillover_features.py`, `code/run_hybrid.py`, tests.
Lenses: Blind Hunter (hidden bugs), Edge Case Hunter (boundaries), Acceptance Auditor (spec match),
plus the mandated leakage lens (graph construction + neighbour aggregation) and a performance lens.
`archive/` and shared modules (`FM`/`S1`/metrics/stats) are out of scope (imported read-only).

## Verdict
No critical or major findings. Three minor observations, all accepted with rationale. The central
question the review was asked to stress — leakage and spillover-block redundancy — is addressed:
leakage is prevented by construction and proven by test; redundancy with `GBM+earn` is exactly what
the experiment measures (DM + err_corr) and is reported, not hidden.

## Leakage lens (graph construction + neighbour aggregation) — PASS
- **Graph from train only.** `Wc = S1.build_graph(tr, ...)` with `tr = a[date < ts - embargo]`. The
  adjacency never sees test-window rows. `tickers` is the universe ordering only (no values), so
  computing it over the full panel is not leakage (matches `paper_metrics_sp500`/`full_compare`).
- **Aggregation contemporaneous.** Every S1.GRAPH feature and `g_corr` at row (t, i) is
  `sum_j Wc[i,j]·value_j(t)` — a day-t cross-section. The h-ahead target `y = pk.shift(-h)` is separate.
  Identical causal footing to the `g_corr` the paper already ships.
- **Future-invariance proven.** `test_spillover_causal_future_invariant` perturbs only the latest date and
  asserts earlier-date features are byte-identical (and that the perturbed date DID change, so the test is
  not vacuous).
- **Embargo.** `int(h*1.6)+5` trading days between train end and test start, matching the sibling harness.

## Redundancy lens (is the spillover block non-redundant with own-history?) — MEASURED, not assumed
- `g_corr` (single) equals `g_nb_vol` (first S1.GRAPH feature) by construction (same `Wc`, same NaN-fill) —
  verified in `test_spillover_aggregation_formula`. So `GBM+earn+graph` **nests** `GBM+earn+corr` (7-feature
  block ⊃ the single feature), making "does the richer block beat the single one?" a clean nested test.
- Redundancy with `GBM+earn` is quantified two ways and reported honestly: (a) DM of `GBM+earn+graph` vs
  `GBM+earn`; (b) residual `err_corr` between `GBM+earn` and a standalone `graph_only` GBM. If the block
  carries no orthogonal signal, DM is null and err_corr is high — the honest prior (NO-GO) is the expected
  outcome and the code is built to surface it, not paper over it.

## Findings

### Minor
- **M1 — `err_corr` could be NaN if a pooled prediction vector is constant.** `np.corrcoef` returns NaN when
  either residual series has zero variance. On real HOSE data neither `GBM+earn` nor `graph_only` is
  constant, so this cannot arise in the actual run; `json.dumps` (default `allow_nan=True`) would still
  serialise it as `NaN`. Accepted: guarding an impossible-on-real-data case would add dead code (violates
  §2/§Code-hygiene "no error handling for impossible scenarios"). Documented here as the known boundary.
- **M2 — `GBM+earn` here uses `own8` (no `rq`), unlike the paper's `GBM+earn` (`FM.OWN`, with `rq`).** This
  is the task spec (base block = `[c for c in FM.OWN if c != "rq"]` + earn). All three compared models
  share `own8`, so the internal DM is fair; but this baseline's `GBM+earn` QLIKE is NOT directly comparable
  to the paper's `GBM+earn` number. Called out in `requirements.md` / `design.md`. Accepted (spec-driven).
- **M3 — panel drops rows where `rq` is NaN even though `own8` excludes `rq`.** `FM.panel` does
  `dropna(subset=OWN + ["y"])` and `OWN` contains `rq`, so early rows are dropped for all models alike.
  Consistent across the comparison (same panel) and matches the shared harness convention. Accepted.

## Performance lens — PASS
- Neighbour aggregation is a single vectorised `V @ W.T` matmul per fold inside `S1.graph_feats` (no
  per-observation Python loop). GBM fits are batched over all fold rows by sklearn. The only loops are over
  folds, horizons, models and the 3-seed ensemble — all irreducible. No batch=1 anti-pattern. CPU-bound
  gamma-GBM in the shared `FM.gbm`; adding a GPU path would mean editing a shared module (out of scope).

## Acceptance auditor — PASS
- Models, comparisons, output schema, earnings injection, walk-forward params (folds, embargo, min_rows,
  seed-ensemble, pooled QLIKE, date-clustered DM) all match `requirements.md` and the sibling harness.
- Fail-loud on missing earnings and on empty results (no silent degradation). Config single-sourced.
- Coverage C0=100% / C1=100% on changed lines (branch), `ruff --select F` clean.
