# Summary of update — classical econometric baselines (Track-B, h5)

## What changed
Added a suite of classical econometric volatility baselines evaluated on the EXACT same
data / target / split / held-out observations as the consistent Track-B ladder
(`docs/reports/ladder_consistent_h5_2026-08-09_154402.json`), so they drop into a combined paper
table alongside the deep-model rungs P0 -> P1 -> P2 -> P3 -> G1.

New baseline folder `baselines/classical_baselines/` (requirements / design / code / test /
code_review). Baselines: Persistence (random walk), EWMA/RiskMetrics (lambda=0.94), HAR, HARQ
(daily range-based RQ proxy), log-HAR, and per-ticker GARCH(1,1) / GJR-GARCH / EGARCH (`arch`).

## Files
- `baselines/classical_baselines/code/classical_baselines.py` — baseline library + `run_all`.
- `baselines/classical_baselines/code/run_classical_baselines.py` — CLI; writes the canonical report.
- `baselines/classical_baselines/test/test_classical_baselines.py` — 18 tests (17 unit + 1 smoke).
- `baselines/classical_baselines/{requirements,design,code_review}/` — spec, plan, adversarial review.
- Canonical results: `docs/reports/classical_baselines_h5_2026-08-09_182129.{json,md}`
  (JSON mirrors the ladder `rung_metrics` schema for a drop-in combined table).

## Basis / fair comparison
- Observations: EXACTLY the pooled val (14418) / test (14464) keys and raw Parkinson-variance
  targets scored by the ladder; verified by a smoke test asserting the counts.
- Scorer: the identical `train.evaluate_records` (round-trip through the ladder store's per-ticker
  `target_scaler`, lossless), so all 6 metrics are computed by the ladder's own code.
- Leakage-safe: GARCH params estimated on the train sample only (frozen); HAR/HARQ/logHAR fit on
  train samples only; EWMA/persistence are causal. Same 70/15/15 chronological split.
- Target clarification: the processed `parkinson_volatility` column is numerically the Parkinson
  VARIANCE sigma^2 (verified corr=1.0 vs raw OHLCV); all baselines forecast that quantity and
  GARCH's conditional return variance is compared on the same units.

## Results (h5, TEST, 3299-char canonical file for full val+test)
| baseline | rmse | r2 | qlike | dir_acc | n_test |
|---|---|---|---|---|---|
| Persistence | 0.00277 | 0.658 | 4151* | 48.01 | 14464 |
| EWMA (RiskMetrics) | 0.00231 | 0.762 | 0.601 | 48.03 | 14464 |
| HAR | 0.00229 | 0.767 | 0.579 | 48.40 | 14464 |
| HARQ (proxy) | 0.00229 | 0.767 | 0.574 | 48.38 | 14464 |
| log-HAR | 0.00237 | 0.750 | 0.779 | 48.83 | 14464 |
| GARCH(1,1) | 0.00476 | 0.003 | 1.761 | 48.67 | 14292 |
| GJR-GARCH | 0.00477 | 0.001 | 1.824 | 48.65 | 14292 |
| EGARCH | 0.00478 | -0.003 | 1.874 | 48.83 | 14292 |

Ladder anchors (TEST): P0 rmse 0.002289 / r2 0.7668 / qlike 0.5676; G1 rmse 0.002305 / r2 0.7635 /
qlike 0.5759. (*) Persistence QLIKE is degenerate — inflated by H==L days (Parkinson variance = 0
-> floored prediction); its RMSE/R^2/DirAcc remain valid.

## How the classical baselines stack vs P3/G1 (reviewer defense)
- **HAR-family ties the deep models on point accuracy.** Per-ticker HAR (test rmse 0.002290,
  r2 0.7667) reproduces the ladder's P0 (0.002289 / 0.7668) and is on par with G1 (0.002305 /
  0.7635); HARQ nudges QLIKE to 0.5737 (vs G1 0.5759, P0 0.5676). The deep GNN (G1) does NOT beat a
  plain HAR on test point metrics — consistent with the ladder's own graph verdict B.
- **The deep models and HAR-family both beat GARCH decisively.** GARCH/GJR/EGARCH land at rmse
  ~0.0048, r2 ~0, QLIKE ~1.8 — roughly 2x worse RMSE and far worse QLIKE than HAR/G1. A return-based
  GARCH is a weak forecaster of the range-based realized-variance target (expected: HAR on the RV
  series dominates return-GARCH for RV-proxy targets).
- **Directional accuracy is ~48% for every model, classical or deep** — at/below chance, matching
  the documented structural anti-persistence ceiling; no baseline escapes it.

Net: the deep ladder is defensible against GARCH (clear win) and honestly reported as competitive-
but-not-superior to HAR/HARQ on this daily VN30 RV target.

## N/A / discrepancies
- **HARQ** is an approximation: no intraday realized quarticity in a daily dataset, so RQ_d is
  proxied by RV_d^2 (range-based). Flagged in the report; not the canonical BPQ-2016 HARQ.
- **GARCH family coverage 32/33 tickers** (14247 val / 14292 test): LPB has no raw OHLCV anywhere,
  so returns cannot be formed. Reported explicitly per baseline (`n_obs`, `garch_excluded_tickers`);
  vol-only baselines keep exact 14418/14464 ladder alignment.
- **Horizons other than h5 not run** (h1/10/22): the code accepts a `horizon` arg, but each run is
  ~5-6 min and the machine is under load from the multi-horizon training agent; h5 is the priority
  and the only horizon with a matching ladder artifact for comparison.

## Tests / gates
- `pytest baselines/classical_baselines/test` — 18 pass (17 unit + 1 smoke). Metric-correctness vs
  `evaluate_records` (abs<=1e-12), obs-alignment smoke (14418/14464), per-ticker GARCH real-data
  smoke (garch/gjr/egarch), date-normalization regression, run_all + CLI integration.
- ruff: clean on `baselines/classical_baselines`.
- diff-cover (origin/master...HEAD): C0 = 98% on changed lines (259 lines, 3 missing); the 3
  uncovered lines are the two sys.path bootstrap guards and the `__main__` entry (idiomatic,
  non-logic). pytest-cov + diff-cover were installed for this task. C1 >= 80% (branch logic in
  `_har_design`, GARCH spec dispatch, coverage assertions all exercised by tests).
- Data-quality gate (Pandera/Evidently): reads processed data read-only and derives no new data
  artifact; recorded N/A (no data change) — no `data/processed` or manifest was written.
- Code review: adversarial 3-layer self-review in `code_review/code_review_2026-08-09.md`; 3 real
  bugs found and fixed during development (variance-vs-vol GARCH units, VPB/VRE tz-aware date
  mismatch, log(0) in log-HAR) plus documented edge cases. No open HIGH/MEDIUM.

## DoD checklist
- [x] Baseline folder with requirements/design/code/test/code_review.
- [x] Runnable code (sys.path bootstrap), canonical JSON+MD in ladder schema.
- [x] Same observations / target / split / scorer as the ladder (verified).
- [x] TDD (RED->GREEN), pytest green, ruff clean, real diff-cover recorded.
- [x] Comparison note vs P3/G1.
- [x] Ledger entry + dashboard regenerated; commit + push.
