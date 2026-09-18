# Design — Expected-schedule earnings (leakage-safe), 2026-09-19

## Data flow
```
actual dates                                 expected schedule (leakage-safe)
─────────────                                ────────────────────────────────
SP500: sp500_earnings.parquet ─────────────► expected_schedule(dates)
HOSE:  hose_disclosures.csv ─► quarterly ──► expected_schedule(quarterly)
        (earliest-across-scope, drop FY/H1)
                                              │
full_matrix.load(market) ─► frames           │  (edates injected via _load_earn override)
                                              ▼
run_leaf_graph_paper.run(market,"full", out_dir=results/gamma_gbm/expected_schedule/)
  walk-forward 8 folds × 3 seeds → XGB+E (base) + XGB+E+LG (leaf-graph) + DM + spike + fit evidence
  → leaf_graph_paper_<market>_full_h{1,5,10,22}.json  (NEW dir, no overwrite)
```

## Key design decisions
1. **Reuse, don't reimplement.** The 2026-09-18 runner already produces gate-valid docs (evidence, DM,
   spike, leaf-graph, checkpointing). We inject the earnings source by overriding the module-global
   `_load_earn` at runtime (`R._load_earn = expected_load_earn`). This does NOT edit the sibling
   baseline's file (hard-isolation preserved) and guarantees the expected-vs-actual comparison is
   apples-to-apples (identical machinery, only the date list differs).
2. **New output directory** `results/gamma_gbm/expected_schedule/` so the committed actual-date JSONs
   are preserved (constraint: keep old numbers retrievable).
3. **HOSE quarterly reduction** uses the SSC labels (`quarter` ∈ {Q1..Q4}, drop FY/H1) and takes the
   EARLIEST disclosure per (ticker, fiscal-year, quarter) across scopes — parent/consolidated/combined
   are released the same day for 95% of cases (median 0-day gap), and this preserves coverage (79 HOSE
   tickers file only parent/standalone). Scope choice is therefore immaterial; earliest-across-scope is
   the "when did the quarter's earnings first hit the market" signal that drives volatility.
4. **XGB (no-earn) is earnings-independent** → not re-run; the new paper cites the committed
   `leaf_graph_paper_<market>_noearn_h*.json` for that column.

## Gates
- **Simplicity:** one small pure module + a thin injecting runner; no new abstractions.
- **Anti-abstraction:** reuses `run_leaf_graph_paper`, `full_matrix`, `leaf_graph` directly.
- **Performance/batching:** inherits the runner's batched 3-seed XGBoost; no per-item loop added.

## Causality / no-leakage argument
`predict_schedule` computes, for event `i`, `actual[i-1] + median(gaps[:i])` using only gaps strictly
before `i` — every predicted date depends only on releases already observed at the forecast origin.
The panel then measures distance from `t+h` to the nearest such predicted date. No realized future date
enters, so the earnings feature is point-in-time valid.

## Files
- `code/expected_schedule.py` — predict_schedule, expected_schedule, hose_quarterly_dates,
  expected_vs_actual, summarize (pure, unit-tested).
- `code/run_expected.py` — injecting runner (`main(smoke)`), writes to the new output dir.
- `test/test_expected_schedule.py` — unit + real-data (HOSE quarterly reduction) tests.
