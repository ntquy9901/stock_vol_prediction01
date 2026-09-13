# Requirements — Hybrid feature-concat GBM: own+earn ⊕ graph-spillover block

Date: 2026-09-13. Follows CLAUDE.md §5 SDD (Specify → Clarify → Plan → Tasks → Implement → Validate).

## 1. Objective
Test whether a **richer graph-spillover feature block** adds out-of-sample value to the per-stock
gamma-GBM when **concatenated** with the own-history + earnings block. This is the project's tabular
paradigm (feature concat into one gamma-GBM), not a two-branch neural blend.

Intuition: `GBM+earn` is per-stock and cannot see cross-stock spillover; a graph block adds neighbour
propagation. **Honest prior: NO-GO.** Prior project results:
- `GBM+earn+corr ≈ GBM+earn` (a single spillover feature `g_corr` adds ~0).
- A prior `GNN ⊕ GBM+earn` blend failed with residual error-correlation ~0.98 (graph branch errors
  ~collinear with `GBM+earn`, so no orthogonal signal).

This experiment is a clean, richer-spillover re-test that either confirms the null or finds a small
exception, reported honestly under Diebold-Mariano (DM).

## 2. Models (all gamma-GBM, seed-averaged over `FM.SEEDS`, gamma-loss HistGradientBoosting)
Base own+earn block for every model: `own8 = [c for c in FM.OWN if c != "rq"]` + `FM.EARN`.
- `GBM+earn`       = gamma-GBM(`own8 + FM.EARN`) — the key baseline.
- `GBM+earn+corr`  = gamma-GBM(`own8 + FM.EARN + [g_corr]`) — single-feature spillover (project's existing).
- `GBM+earn+graph` = gamma-GBM(`own8 + FM.EARN + <7-feature spillover block>`) — the new/richer model.
- `graph_only`     = gamma-GBM(`<7-feature spillover block>`) — diagnostic-only, for error-correlation.

`own8` = `[har_daily, har_weekly, har_monthly, mr_change, mr_slope5, mr_slope10, mr_dev5, mr_z22]` (8).
`FM.EARN` = `[earn_prox, earn_soon, earn_pre, earn_post]`.

## 3. Spillover block (the richer part)
Reuse the project's causal graph machinery **read-only** (no edits to `FM`/`S1`):
- `S1.build_graph(train_rows, tickers, rng)` → per-fold correlation top-k adjacency `Wc`, built on
  **train logpk only** (leakage-safe).
- `S1.graph_feats(fold, tickers, Wc, "")` → the 7-feature `S1.GRAPH` block:
  `g_nb_vol, g_nb_shock, g_nb_max, g_nb_disp, g_node_minus_nb, g_nb_ret, g_nb_volshock` (neighbour-mean
  volatility, neighbour shock, neighbour max, neighbour dispersion, node-minus-neighbour, neighbour
  return, neighbour volume-shock).
- `FM.nb(fold, tickers, Wc)` → the single-feature `g_corr` (neighbour-mean parkinson_variance).

## 4. Evaluation (mirror the sibling harness `paper_metrics_sp500` / `full_compare` / `run_gbm`)
- Walk-forward over `S1.FOLDS`, expanding train from `S1.TRAIN_START`, embargo `int(h*1.6)+5` days.
- Per-fold graph from **train only**; fold panel spans `[TRAIN_START, tend)`.
- Fold gate: `len(te) > 0` and `len(tr) >= min_rows` (`3000` for hose, `30000` for sp500).
- Pooled per-observation QLIKE (`M.per_obs_qlike`, floor `FM.FL`); seed-averaged GBM predictions.
- Date-clustered DM (`ST.date_clustered_dm`) at each horizon h ∈ {1,5,10,22}.
- Real VN earnings injected for HOSE from `results/gamma_gbm/hose_earnings_combined.parquet`
  (as `paper_metrics_sp500` does).

### Comparisons (all horizons)
1. `GBM+earn+graph` vs `GBM+earn` — key test: does the spillover block add OOS value over earnings?
2. `GBM+earn+graph` vs `GBM+earn+corr` — does the richer block beat the single-feature one?
3. Residual **error-correlation** between `GBM+earn` and `graph_only` — the 0.98 orthogonality check.

## 5. Output
`results/gamma_gbm/gbm_earn_graph_<market>.json`, per horizon:
`{n, qlike: {model: q}, dm: {"GBM+earn+graph_vs_GBM+earn": p, "GBM+earn+graph_vs_GBM+earn+corr": p},
  err_corr, train_metrics, test_metrics, fit_diagnostics}`.

## 6. Acceptance criteria (go/no-go)
- Code runs HOSE end-to-end, writes the JSON with all four horizons.
- Spillover block proven causal (graph from train only; aggregation uses data ≤ t; feature at t
  unchanged when future rows are perturbed) — enforced by tests.
- Tests pass under `.venv_gpu_encode`; changed-line coverage C0=100% / C1≥95%.
- **Economic go** (spillover retained) only if EVERY horizon has a positive, sign-consistent QLIKE
  gain ≥ `SUCCESS_MIN_GAIN_PCT` with DM `p < SUCCESS_DM_P`, no overfit verdict, AND the richer block
  beats `GBM+earn+corr`. Otherwise **NO-GO** — reported honestly. (Prior expectation: NO-GO.)
- At n~large a ~0.04% "DM-significant" gain is NOT a real economic win — report the effect size, not
  just the p-value.

## 7. Non-goals / scope
- SP500 is a note only (Colab; not run locally — too heavy for the 4060).
- Do NOT modify shared modules (`FM`/`S1`/metrics/stats) — import read-only.
- `archive/` out of scope.
- Do NOT push; commit to the worktree branch for review.
