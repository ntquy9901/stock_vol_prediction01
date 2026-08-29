# Sector-graph ablation for the LSTM+GAT volatility model — requirements

**Date started:** 2026-08-29
**Status:** CPU-prep + HNX directional run (GPU run for scale-up deferred until the RTX 4060 frees)

## Objective

Test whether a STATIC sector-defined graph edge beats the shipped STATISTICAL edge (directed
volume-shock→Parkinson, `adj_vol2pk`) and the no-graph LSTM on out-of-sample QLIKE, under the exact
`MaskedRichNet` / HAR-X pipeline already used by the delivered results.

## Motivation

Read-only EDA (`docs/reports/2026-08-29_gat_signal_eda_and_harm_analysis.md`, graph handoff) found the
statistical edges persist only ~9–30% train→test (unstable OOS), the market factor is already a node
feature (`market_pk`, so the graph is partly redundant), and own-history persistence dominates. A
sector graph is built from static metadata: no OOS drift, no leakage. It is the natural stable
alternative to probe.

## Scope (after 2026-08-29 course change)

- **Panels:** HNX (primary), VN100 (secondary, if time permits). S&P500 code retained but deprioritized.
- **Target:** Parkinson variance (the shipped primary target).
- **Three configs:** sector-GAT (`adj = sector`) vs stat-GAT (`adj = adj_vol2pk`) vs no-graph LSTM.
- **Compute:** CPU only (the overnight GPU run owns the RTX 4060). HNX N≈154 at 5–10 epochs / 1–3
  seeds trains on CPU in minutes.

## Inputs

- Raw OHLCV: `data/raw/prices/hnx_vnstock/` (VN100: `.../vn100_vnstock`); processed Parkinson panels
  under `submission/soict_lstm_gat/data/` and `data/processed/hnx`.
- Sector labels: vnstock ICB `Listing().symbols_by_industries()` → `vn_sectors.csv` (VN); the datahub
  GICS table → `sp500_gics_sectors.csv` (S&P500).

## Outputs

- `sector_adjacency.build_sector_adjacency(tickers, sector_map, top_k)` — the tested, panel-agnostic
  adjacency builder (drop-in for `MaskedRichData.adj_*`).
- Ticker→sector CSVs with provenance (source + fixed date string).
- `results/sector_gat_ablation/sector_ablation_<panel>_h<h>.json` — 5 metrics (MSE/RMSE/MAE/QLIKE/R²)
  per config + date-clustered Diebold–Mariano (sector-GAT vs stat-GAT, sector-GAT vs no-graph).

## Acceptance criteria

1. Sector-adjacency builder passes TDD property tests (block-diagonal, self-loop, symmetry,
   no cross-sector edge, shape `[N,N]`, singleton for unmapped, Top-K cap).
2. HNX sector-label coverage reported (≥90% expected); unmapped → singleton own-sector.
3. CPU smoke: adjacency aligns to `D.tickers`, `MaskedRichNet(use_graph=True)` yields finite output
   on a tiny batch — no training loop, no GPU.
4. HNX 3-way comparison at 5–10 epochs produced on CPU, all 5 metrics + DM, clearly labelled a
   DIRECTIONAL check (not a final number).
5. No edit to the live-training-path files (`masked_rich`, `run_masked_rich`,
   `estimator_forecast_ablation`, `run_yz_robustness`, `config`) — import read-only.
6. Pre-push gate green (C0 line 100% / C1 branch ≥95% on changed lines, ruff-F clean, lessons,
   data-quality N/A no data change).

## Go / No-Go

- **Go to scale-up (GPU, more seeds/epochs, VN100)** if the CPU directional check shows sector-GAT
  competitive with or beating stat-GAT / no-graph on QLIKE.
- **No-Go / report negative** if sector-GAT does not improve on the stat edge or no-graph at the
  directional setting — a clean negative is a valid ablation outcome (consistent with the graph
  consistently not helping in prior runs).

## Non-goals

- No final publishable numbers (needs GPU, 5 seeds, both horizons).
- No edit to shipped pipeline / no new target estimator.
- Not fabricating sector labels: if VN labels were unavailable, escalate as the blocker (they were
  available via vnstock ICB — 98.8% HNX coverage).
