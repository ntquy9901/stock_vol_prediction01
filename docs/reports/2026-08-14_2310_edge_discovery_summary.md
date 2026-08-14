# Edge-Discovery EDA — supervised OOS ΔQLIKE directed edges (summary)

Date: 2026-08-14. Branch `feature/edge-discovery-eda`. Module `graph_eda/edge_discovery.py` +
runner `graph_eda/run_edge_discovery.py`. Outputs `docs/eda/edge_discovery/`.

## Motivation
An external suggestion (relayed by the user): the prior EDA was strong at node-feature discovery
but shallow on EDGE relationships — it ranked edges by RMSE gain at h=1, whereas the deep models
beat HAR on **QLIKE**. So an edge useful on QLIKE could be missed. This pass answers the sharper
question, on the metric that matters and conditioned on the winning node features (E2 = HAR +
MarketPK + volume_zscore_20):

> Does knowing source i's history LOWER target j's out-of-sample QLIKE (t+5) after j's own HAR +
> the market factor + j's volume z-score are already known?

Scope (user-chosen): X4 supervised ΔQLIKE directed edge matrix + X2' market-residual lead-lag +
X6' regime-conditioned; build+DM only if a positive, OOS-stable edge set emerged.

## Method (leakage-safe)
Per directed pair (i→j): base ridge = HAR_j + MarketPK + volz_j predicting PK_j(t+5); augmented =
base + PK_i(t). ΔQLIKE = QLIKE_base − QLIKE_full (positive = source helps). Predictions floored at
the train 1st-percentile of positive targets (identically for base and full). Coefficients fit on
TRAIN; edges SELECTED on VALIDATION only (ΔQLIKE_val>0); generalization judged on the untouched
TEST. Verdict gates on the sign-count val→test binomial (across pairs) + a robust median centre.

## Result — verdict: STOP
- 449/1056 directed pairs are validation-selected (ΔQLIKE_val>0).
- On the held-out test they are **45.9% test-positive** (binomial vs 50% p=0.089 — indistinguishable
  from chance, marginally below), mean test ΔQLIKE **−0.012**, median **−0.0002** (Wilcoxon
  p≈0.0006 — the median selected edge significantly *hurts* OOS).
- Validation selection is not broken (val-selected test ΔQLIKE −0.012 vs non-selected −0.033, Welch
  p=0.027) — it selects a real but **net-harmful** regressor: the cross-stock feature inflates OOS
  QLIKE variance more often than it helps.
- Regime: apparent "help under stress" in the mean is a tail artifact — high-vol-regime median is
  negative (Wilcoxon p≈0.019), edges hurt *more* under stress at the median.
- Market-residual lead-lag L(5): mean |off-diag| ≈ small (reported in the EDA report).

**The naive counts (206 sign-stable, 47 "FDR-significant") were a test-peeking + multiple-testing
artifact** (both use the test set in selection; ~1000 pairs × ~14k obs cross the line by chance).
They do NOT survive the honest val→test split and are labelled as context-only in the report.

## Phase B decision
**Not built.** The EDA verdict is STOP, so building a ΔQLIKE-edge GNN is not justified — the
conditional (a positive, OOS-stable edge set) was not met. This is a stronger stop-GNN result than
the prior RMSE-only ranking: even correcting the metric (QLIKE), the confound (market factor via the
E2 base), and the direction (supervised, directed), no edge definition recovers conditional
cross-stock signal beyond E2, and under robust statistics the candidate edges significantly hurt.

## Tests + coverage
- 7 tests pass (`python -m pytest graph_eda/tests/test_edge_discovery.py`), incl. a smoke that runs
  the full runner on the real 33-ticker panel + synthetic tests proving the method finds a *true*
  planted lead-lag edge (S→T) and rejects a noise source. ruff clean; **diff-cover C0 = 100%** on
  changed lines.

## Code review (2-layer adversarial + inline acceptance)
- **Leakage / edge-case layer:** no leakage found (train-only fits, strict-future target, identical
  per-pair mask for base/full, train-only floor + regime thresholds). Fixes applied: empty-edge
  `_summarise`/`_heatmap` crash guarded; verdict gate switched mean→**median** (outlier-robust);
  the per-edge p_test/FDR flagged as horizon-overlap anti-conservative (verdict does not rely on
  them — it uses the pair-level sign binomial).
- **Statistical-validity / false-negative layer:** re-ran 8 methodology variants (floor on/off,
  raw PK_i vs market-residual — **numerically identical** because the base's MarketPK+intercept make
  ridge orthogonalize the edge against the market; λ=1e-4 vs 1.0; sign vs val-significance selector;
  mean vs median/Wilcoxon). **STOP is robust in every configuration** (45.6–47.6% test-positive);
  under robust statistics it is *stronger*. No bug produces a false negative. Full record:
  `docs/eda/edge_discovery/reports/code_review_2026-08-14.md`.

## Data-quality gate
N/A (read-only EDA on `data/raw/prices`; no `data/processed`/manifest/feature change). Leakage
controls are the EDA's own train/val/test discipline + `graph_eda/leakage.py` assertions.

## Risks / follow-ups
- Verdict is a documented STOP; no build planned. If revisited, the only untried angles are
  non-linear conditional dependence (conditional MI given MarketPK) and news-as-edge co-mention —
  but the linear supervised ΔQLIKE test already conditions out the market factor and finds nothing,
  so the prior is strongly null.

## DoD checklist
- [x] Module + runner + tests; TDD green; diff-cover C0=100%; ruff clean
- [x] Real run on 33-ticker panel; honest STOP verdict; report + figures + tables
- [x] 2-layer adversarial review (leakage + false-negative); findings fixed
- [x] Data-quality gate: N/A documented
- [ ] Commit + push branch (next)
