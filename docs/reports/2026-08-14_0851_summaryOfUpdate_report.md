# Summary of Update — graph_eda plan coverage gap-fill (Tier 1 + Tier 2)

Date: 2026-08-14

## Context / request

Review the existing `graph_eda/` EDA implementation against the guiding plan
`docs/eda_guide/parkinson_volatility_gnn_eda_experiment_plan1408.md` (byte-identical to the
already-executed `..._plan.md`), find sections skipped or done differently, and fill the
Tier-1 + Tier-2 gaps. The purpose is to complete the evidence base for the "how to use a GNN
correctly" question. The GNN verdict itself was already settled (Conclusion C — market factor
dominates); none of these gaps change it. This pass adds completeness / paper-figure
deliverables and one additional per-K predictive diagnostic.

## What changed

### New analytical functions (unit-tested)
| File | Function | Plan § |
|---|---|---|
| `graph_eda/graphs.py` | `clustering_metrics` (networkx modularity/communities/degree/components/clustering-coeff) | §23 |
| `graph_eda/graphs.py` | `sector_purity`, `mean_edge_strength` | §50 |
| `graph_eda/graphs.py` | `snapshots_to_long`, `multi_window_edge_panel` (leakage-safe trailing windows) | §18/§46 |
| `graph_eda/relationships.py` | `bootstrap_ci_mean` (percentile bootstrap CI) | §56 |

### New orchestrator outputs (`graph_eda/run_eda.py`)
- `ticker_coverage.csv`, `daily_stock_features.parquet` (§46)
- `return_corr_spearman.csv`, `volume_change_corr.csv` (§8/§10)
- `contemp_vshock_to_pk.csv`, `contemp_return_to_pk.csv` — same-day cross-feature (§11)
- `return_leadlag_{1,2,5}.csv`, `volume_to_pk_lag{1,2,5}.csv` (§13/§14)
- `dynamic_edge_features.parquet` — multi-window (20/60/120) rolling PK/return/volume edge panel, 30,096 rows × 13 cols (§18/§46)
- `graph_clustering.csv` (§23), `topk_search.csv` — per-K (3/5/8/10) density/stability/sector-purity/edge-strength/predictive (§50)
- bootstrap CIs on headline means, rendered in the report (§56)
- `docs/eda/graphs/*.png` — node-link graph visualizations: PK Top-5 at low/normal/high-vol dates, directed lead-lag, multi-edge (§48)

### Report / recommendation
- `docs/eda/reports/EDA_GRAPH_REPORT.md`: added CIs to Parkinson/Sector sections and a
  "Graph Clustering & Multi-Window Dynamics" section. Verdict regenerated unchanged
  (MAYBE → Conclusion C).
- `docs/eda/graph_recommendation.json`: additive new evidence keys.

## New evidence produced (not verdict-changing, but informative)
- **Top-K search (§50):** OOS predictive gain over HAR+market is positive only at **K=3
  (+0.03%)** and turns negative for K≥5 (−0.78% / −0.89% / −1.25%). Sparser is better; even
  the best K barely helps — consistent with Conclusion C.
- **Clustering (§23):** PK Top-5 graph forms 4 greedy-modularity communities
  (modularity 0.306, clustering-coeff 0.548), single connected component.
- **Confidence intervals (§56):** mean |PK corr| 0.3778, 95% CI [0.3655, 0.3903];
  same-sector 0.5003 [0.4738, 0.5285] vs cross-sector 0.3450 [0.3342, 0.3564] —
  non-overlapping.

## Files (path → purpose)
- `graph_eda/graphs.py`, `graph_eda/relationships.py`, `graph_eda/parkinson.py`,
  `graph_eda/run_eda.py` — implementation.
- `graph_eda/tests/test_graphs.py`, `test_relationships.py`, `test_parkinson.py`,
  `test_run_eda.py` — unit + integration/smoke tests.
- `docs/eda/tables/*`, `docs/eda/figures/*`, `docs/eda/graphs/*`,
  `docs/eda/reports/EDA_GRAPH_REPORT.md`, `docs/eda/graph_recommendation.json` — regenerated deliverables.

## Tests + coverage
- **76 tests pass** (baseline 63; +13 new), ruff clean on `graph_eda/`.
- **diff-cover (C0) = 100%** on changed lines (`diff-cover coverage.xml --compare-branch=HEAD
  --fail-under=100`), branch (C1) ≥ 80% — key branches (dyn None/not, regime fill/skip,
  clustering empty/non-empty, NaN-edge skip, bootstrap empty, k/tau guard) exercised by tests.
- Smoke: `pytest graph_eda/tests/test_run_eda.py -m smoke` passes (full pipeline on the real
  33-ticker universe end-to-end).
- Commands run: `python -m pytest graph_eda/tests/ --cov=graph_eda --cov-branch --cov-report=xml`,
  `diff-cover coverage.xml --compare-branch=HEAD --fail-under=100`, `ruff check graph_eda/`,
  `python -m graph_eda.run_eda`.

## Code review (3-layer adversarial, before done)
Blind Hunter + Edge Case Hunter + Acceptance Auditor (parallel subagents). Leakage-safety
verified PASS by all three (trailing-window multi-window panel with per-snapshot
no-look-ahead assertion; train-only neighbour selection; k=0 contemporaneous branch correct).

Findings — all addressed:
- **MEDIUM** bootstrap CI computed on 1056 duplicated off-diagonal entries → ~30% too narrow.
  Fixed: bootstrap over the 528 unique pairs (`pk_pairs["corr"]`). CI widened to the correct value.
- **MEDIUM** report hardcoded "non-overlapping CIs" string. Fixed: computed from the CI bounds.
- **MEDIUM** `build_features` emitted `inf` for a nonpositive-price row (SSI 2006-12-15 low=0)
  into `daily_stock_features.parquet`. Fixed: sanitize inf→nan (+ `np.errstate`) mirroring the
  sibling `_pk_var_wide`. Verified: no inf in the 106,540-row parquet.
- **LOW** `_edge_graph` TypeError when neither k nor tau given → raises clear ValueError.
- **LOW** Top-K branch of `_edge_graph` didn't skip NaN-corr edges → added guard.
- **LOW** multi-window panel crashed on a ≤119-row panel → extracted to
  `multi_window_edge_panel`, returns typed-empty frame (unit-tested).
- **LOW** `mean_edge_strength` RuntimeWarning on all-NaN edges → filter non-finite first.
- **LOW/perf** per-K density recomputed full community detection → replaced with direct
  edge-count density.
Acceptance Auditor: all Tier-1+2 deliverables present, no over-claiming, no scope creep.

## Data-quality gate (Pandera + Evidently)
**N/A (no model-data change).** `graph_eda/` reads only `data/raw/prices`; it does not modify
`data/processed`, the training manifest, or model features (the Pandera schema / Evidently
drift targets). EDA data-quality is covered by the pipeline's own checks
(`data_quality_summary.csv`: 0 high<low, 1 nonpositive-price, 0 duplicate dates across 33
tickers) and the leakage assertions in `graph_eda/leakage.py`, exercised by the test suite.

## Risks / follow-ups
- Tier-3 plan-optional items intentionally skipped (extra optional lag/rolling features,
  `config.yaml`, mutual-information/distance-correlation) — plan marks them optional; no
  evidentiary value for the GNN decision.
- Verdict unchanged (Conclusion C): the correct GNN direction remains market-factor +
  volume-shock **node features** on the LSTM/HAR backbone; cross-stock message-passing adds
  no OOS value. The added Top-K search reinforces this (predictive gain positive only at K=3,
  +0.03%).

## DoD checklist
- [x] Code satisfies request (Tier-1+2 gaps filled); no unrelated refactor
- [x] Tests written/run; diff-cover C0=100% on changed lines
- [x] Lint run (ruff clean)
- [x] Smoke test pass (full pipeline)
- [x] 3-layer code review run; all findings fixed
- [x] Summary report (this file)
- [x] Data-quality gate: N/A documented with reason
- [ ] Commit + push origin master (next)
