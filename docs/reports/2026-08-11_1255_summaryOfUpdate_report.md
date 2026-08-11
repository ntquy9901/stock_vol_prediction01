# Summary of Update - Graph/GNN EDA on Daily VN30 Parkinson Volatility

Date: 2026-08-11. Task type: code (analysis module + evidence deliverables).

## What was done

Executed the experiment plan `docs/eda_guide/parkinson_volatility_gnn_eda_experiment_plan.md`
(Priorities 1-4) on daily OHLCV for the 33 VN30 tickers to determine, with leakage-safe
evidence, whether a GNN/GAT is justified for cross-stock Parkinson-volatility forecasting,
and if so which graph. Priority 5 (actual GNN training) was already done by the project's
G1 ladder/sweep and is referenced, not retrained.

## Verdict

**Use GNN: MAYBE** - a standard correlation GNN will NOT beat HAR, but ONE specific,
non-obvious edge (directed volume->PK lead-lag) has a small positive out-of-sample lift over
HAR+market and is worth building + DM-testing. The market factor dominates cross-stock
Parkinson structure (Conclusion C). Full evidence: `docs/eda/reports/EDA_GRAPH_REPORT.md`;
machine-readable recommendation + config: `docs/eda/graph_recommendation.json`.

### Recommended GNN config (to build next and test vs HAR with Diebold-Mariano)
- Node features: pk_daily, pk_weekly, pk_monthly (HAR core) + volume_zscore_20.
- Edge: directed volume->PK lead-lag, Top-K = 5, dynamic; edge features volume_to_pk_lag1/5.
- Expected OOS RMSE lift: +0.24% over HAR, +0.17% over HAR+market (67% of 33 stocks, sign
  p = 0.08 - marginal, NOT significant at 0.05). Honest expectation: may or may not clear HAR
  under DM; it is the strongest evidenced shot, but the lift is small.
- Do FIRST: add MarketPK (median PK) as a global node feature to HAR - it is the single largest
  cross-stock signal and the ablation the G1 sweep skipped.

### Key numbers (all from generated tables, common panel = 1296 trading days, 2021-03-24 to 2026-06-09)
- Cross-stock PK dependence is real: mean |PK corr| = 0.378, far above the shuffle-null 95th
  pct 0.023 (empirical p = 0.002); 99.4% of pairs FDR-significant. (Gate 1 passes.)
- **Market-factor adjustment (key test):** mean R^2 of each stock's PK on median-market PK
  (train) = 0.424. After removing the market factor, mean |pairwise corr| falls 0.378 ->
  0.087, i.e. **only 23% retained**. Most cross-stock structure is one common factor.
- **Node-feature ranking over HAR (incremental OOS, h=1):** the ONLY own feature that beats
  HAR is volume_zscore_20 (+0.55% RMSE, 31/33 stocks, sign p=1.3e-7). pk_lag2, pk_std_20,
  pk_vol, log_return_1d, return_5d all HURT.
- **Edge-definition ranking over HAR+market (incremental OOS, h=1):** directed volume->PK
  +0.17% (p=0.08, only positive edge); market-adjusted residual-corr -0.92% (p=0.035);
  directed PK lead-lag -1.15%; **plain PK-correlation edge -2.20% (p=0.014, significantly
  WORSE)**; multi-edge -2.21%. The plain PK-corr edge is the one G1's kNN-8 GAT uses.
- **Neighbour predictive test OOS (Gates 6-7):** all baselines on an identical shared sample
  (n=6369). C(own+PK-corr-neighbours) does NOT beat own-only (Gate 6 fail; C<A in 24% of
  stocks, p=0.005 = worse) nor own+market (Gate 7 fail; C<B in 42%, p=0.49). Same at h=5.
- Lead-lag: PK L(1) off-diag max 0.487 ~ own lag-1 autocorr 0.471, asymmetry mean 0.024 - no
  stable directionality for PK edges. Sector: same 0.500 vs cross 0.345 (p<1e-4) but residual
  same-sector corr collapses to 0.091 after market adjustment.

### How this explains the project's G1-null (direct measurement)
G1 (kNN-8 correlation GAT) tied HAR and the beat-HAR sweep failed. Direct data reason: the
plain PK-correlation edge G1 uses is the WORST edge construction tested - it changes OOS RMSE
by -2.20% vs HAR+market (sign p=0.014), i.e. significantly worse. ~77% of cross-stock PK
correlation is one market factor HAR already captures, so a correlation GAT re-learns it and
adds noise. Directed PK lead-lag also fails the market bar; only the directed volume->PK edge
(never used by G1) shows a marginal positive lift. Conclusion C (market factor dominates), with
a concrete, testable next step.

## Files

| Path | Purpose |
|---|---|
| `graph_eda/io_data.py` | robust OHLCV load (tz-aware VPB/VRE normalised), common-panel alignment, chronological 70/15/15 split |
| `graph_eda/parkinson.py` | Parkinson variance/vol + leakage-safe per-ticker features |
| `graph_eda/data_quality.py` | plan section-5 per-ticker quality checks |
| `graph_eda/relationships.py` | pairwise corr + BH-FDR + permutation null, vectorised lead-lag, market-factor residuals |
| `graph_eda/graphs.py` | Top-K neighbours, stability, turnover, density, random control |
| `graph_eda/predictive.py` | OOS baselines A/B/C (Gates 6-7), shared-mask + per-stock sign test |
| `graph_eda/incremental.py` | incremental-over-HAR ranking of node features + edge definitions |
| `graph_eda/sectors.py` | documented VN30 sector map (no metadata file exists) |
| `graph_eda/leakage.py` | automated leakage/integrity assertions (plan sections 31/55/65) |
| `graph_eda/run_eda.py` | orchestrator: writes all tables/figures/report/json |
| `graph_eda/tests/` | 45 pytest tests incl. real-data smoke |
| `docs/eda/tables/` | 27 CSVs incl. node_feature_ranking_h1, edge_definition_ranking_h1, predictive_per_stock_h1/5 |
| `docs/eda/figures/` | 16 PNGs (plan section-47 set) |
| `docs/eda/reports/EDA_GRAPH_REPORT.md` | executive summary, decision matrix, RQ1-10, conclusion |
| `docs/eda/graph_recommendation.json` | machine-readable recommendation (computed, not hard-coded) |
| `ruff.toml` | added `graph_eda/run_eda.py` E501 per-file ignore (report/figure generator; matches existing precedent) |

## Tests + coverage

- `pytest graph_eda/tests`: 63 passed (incl. 2 `smoke` tests: real-universe alignment + a full
  end-to-end orchestrator run into a tmp dir with reduced permutation counts).
- **diff-cover vs origin/master: C0 = 100% on changed lines** (`coverage.xml`); every module
  incl. `run_eda.py` shows 100% - the orchestrator is line-covered by the end-to-end integration
  test (CLAUDE.md "test the run_*() runner, not just pure helpers"). C1 branch coverage high
  (verdict, lift-evidence, and leakage-boundary branches unit-tested).

## Commands run
- `python -m graph_eda.run_eda` (full pipeline, ~85s; regenerated all deliverables).
- `python -m pytest graph_eda/tests -q` -> 45 passed; `-m smoke` -> 1 passed.
- `python -m pytest graph_eda/tests --cov=graph_eda --cov-report=xml`.
- `ruff check graph_eda` -> All checks passed.
- Pandera `check_schema()` -> PASS 34/34 processed artifacts (existing data unaffected).

## Data-quality gate
- Pandera schema on `data/processed`: PASS 34/34 (evidence that the existing processed data is
  intact; this change adds a read-only EDA module and does not modify `data/processed`, any
  manifest, or the training pipeline).
- Evidently drift: N/A - no change to training features/pipeline (per CLAUDE.md "N/A (no data change)").

## Code review
- Adversarial review (Blind Hunter + Edge Case Hunter + Acceptance Auditor) run over the module.
  Full record: `docs/eda/reports/code_review_2026-08-11.md`. Verified safe: lead-lag alignment,
  vectorised correlation formula, chronological split, train-only betas/scalers/neighbour-selection,
  determinism, trailing-only snapshots (no look-ahead leakage found).
- 2 MAJOR fixed: M1 - baselines now share one valid mask so A/B/C are pooled over identical
  (stock, test-row) pairs (verified n=6369 identical); M2 - added per-stock win-rate + sign test
  and a >0.5% margin + stock-majority gate rule, so the verdict rests on tested evidence not a
  bare point comparison. 3 MINOR fixed: N1 per-stock DirAcc, N4 QLIKE excludes zero-variance days,
  N3 std==0 guard. N2/N5/N6 accepted with documented rationale. All fixes covered by tests.

## Risks / follow-ups
- Single split, single market-factor definition (median PK). The verdict is robust across the
  three decisive lines (null-separation, market-adjustment, OOS neighbour test) but a multi-seed
  / alternative-market-factor sensitivity check would further harden it.
- Recommended next experiment: add MarketPK as a global/node feature to HAR/LSTM and re-check any
  residual graph gain beyond it (the ablation the G1 sweep skipped).

## Definition of Done
- [x] Code satisfies the request (plan Priorities 1-4, leakage-safe, decision gates + minimum evidence).
- [x] Tests written first (TDD) and green (45 pass); library C0=100%.
- [x] Lint clean (ruff).
- [x] Smoke test passes.
- [x] Data-quality: Pandera PASS 34/34; Evidently N/A (no data change) with reason.
- [x] Adversarial code review run + findings addressed.
- [x] Summary report (this file) + evidence deliverables generated.
- [x] Ledger entry + dashboard regenerated; committed and pushed.
