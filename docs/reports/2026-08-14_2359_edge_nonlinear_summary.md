# Nonlinear edge discovery — gradient-boosting ΔQLIKE + MI-residual (summary)

Date: 2026-08-14. Branch `feature/edge-discovery-eda`. Modules `graph_eda/edge_nonlinear.py` +
runner `graph_eda/run_edge_nonlinear.py`; refactor `graph_eda/edge_discovery.py` (predict_fn hook).
Outputs `docs/eda/edge_discovery/`.

## Motivation
The linear supervised ΔQLIKE edge test returned STOP. User asked to probe the remaining angle: a
NONLINEAR / tail cross-stock relationship (only extreme source volatility drives a target's future
volatility) that a linear model would underrate. Two leakage-safe probes, conditioned on the E2 node
features (HAR + MarketPK + volume_zscore_20), h=5:
- **GB ΔQLIKE** — the same val-select / test-confirm / floor / median-gate harness as the linear
  test, but the per-pair predictor is a regularized gradient-boosting regressor.
- **MI-residual** — mutual information of source i(t) with the E2-residual of target j(t+5) vs a
  source-permutation null.

## Result — verdict: STOP
- **GB val→test generalization:** 378 validation-selected edges → test-positive **exactly 50.0%**
  (binomial p=1.0), median test ΔQLIKE ≈ 0 (Wilcoxon p=0.67). Even a nonlinear/tail predictor finds
  no cross-stock edge that helps OOS QLIKE beyond E2, in aggregate.
- **MI-residual is a market-leftover CONFOUND, not an edge:** 626/1056 (59.3%) pairs exceed the
  source-permutation null p95 — but per-source mean MI correlates **0.771** with |corr(source,
  MarketPK)| (67% above-null for high-market-corr sources vs 52% for low). The target residual is
  de-meaned by MarketPK(t) but lands at t+5 still carrying the market factor at t+5 (not removable
  leakage-safe); market-co-moving sources share that leftover. The supervised GB test (base already
  contains MarketPK) correctly finds no incremental value.
- **Honest caveat (not "zero dependence"):** a thin ~10-edge large-cap cluster (Vingroup: BVH→VIC,
  VIC→VHM, …) does replicate on test (top-10 val-ranked ~80% test-positive) but is tail-driven and
  factor-consistent (source/target market-correlation ≈ panel median), i.e. a residual sector/market
  factor, not idiosyncratic spillover — and the sibling DM-tested build (`2026-08-11_eda_gnn`) already
  showed that graph does not beat HAR (QLIKE p=0.116). So the aggregate is at chance and **no
  buildable graph is justified**.

## Method note (raw vs residual, linear vs nonlinear)
Adding raw PK_i(t) to a base already containing MarketPK is equivalent to adding its market-residual
(ridge/GB orthogonalize) — so this is the market-conditioned edge test. GB extends it to nonlinear
interactions/tails; the synthetic unit test proves the exact production config recovers a planted
tail edge, so a null here is not a detection failure.

## Tests + coverage
- 12 tests pass across `test_edge_nonlinear.py` (GB recovers planted tail edge; MI > null; runner
  smoke via ridge-swap; empty guards) and `test_edge_discovery.py` (predict_fn refactor intact).
- ruff clean; **diff-cover C0 = 100%** on changed lines (`edge_discovery.py`, `edge_nonlinear.py`,
  `run_edge_nonlinear.py`, tests). GB runner `main()` validated by the real 3-split run (CSVs +
  report); its pure helpers unit-covered via monkeypatched smoke.

## Code review (adversarial validity, before done)
One focused layer: "is the STOP a false negative (GB over-regularized / overfit) and is the
MI-confound correct?" Verdict: **STOP robust.** Evidence: the production GB config provably detects a
planted tail edge (unit test); **loosening regularization pushes val→test BELOW chance (44.7%,
p=0.018)** — so 50% is the charitable case, not suppression; the 50% is genuine (real prediction
spread, 0/378 exact-zero in the selected set); every MI-confound number reproduces exactly (0.77,
67%/52%, 59.3%); no material leakage. The reviewer's one honesty note — phrase as "no broad/buildable
edge; aggregate at chance", not "zero dependence" — has been applied to the report. Record:
`docs/eda/edge_discovery/reports/code_review_nonlinear_2026-08-14.md`.

## Data-quality gate
N/A (read-only EDA on `data/raw/prices`; no `data/processed`/manifest/feature change).

## Net conclusion for the project
Across the full edge investigation — correlation, vol→PK, supervised linear ΔQLIKE, and now
supervised nonlinear (GB) ΔQLIKE + MI — **no cross-stock edge construction adds usable out-of-sample
value beyond the E2 node features**. The only lever that beats HAR (on QLIKE) is node features
(MarketPK + volume_zscore_20). This is the strongest stop-GNN evidence available on daily-OHLCV VN30
data.

## DoD checklist
- [x] Module + runner + tests (TDD); diff-cover C0=100%; ruff clean
- [x] Real run; honest STOP verdict + confound explanation + large-cap-tail caveat
- [x] Adversarial validity review (false-negative + MI-confound); findings applied
- [x] Data-quality gate: N/A documented
- [ ] Commit + push branch (next)
