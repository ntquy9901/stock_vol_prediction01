# Summary — GBME + leaf-cooccurrence graph baseline (2026-09-18)

## What changed
New baseline `baselines/2026-09-18_gbm_leaf_graph/` testing whether smoothing the champion gamma-GBM's per-stock
volatility predictions over a graph built from the GBM's OWN trees (leaf-index cooccurrence) improves HOSE
out-of-sample QLIKE. XGBoost `reg:gamma` (capacity-matched to the HGBR champion GBME) exposes per-tree leaf
indices; each day's stocks are linked by leaf-Hamming similarity into a per-day kNN graph; the base prediction
is smoothed `ŷ = (1−α)ŷ + α·mean_{kNN}ŷ` with α fit on a per-fold validation slice, frozen for test.

## Files (path → purpose)
- `code/leaf_graph_config.py` — all tunable constants (XGB params, K, α grid, horizons, gates, spike windows).
- `code/leaf_graph.py` — booster fit/predict, leaf matrix, sparse leaf-Hamming similarity, kNN neighbour-mean,
  per-day causal smoothing, val α-fit.
- `code/run_leaf_graph.py` — walk-forward driver (GBME / XGB / XGB+leafgraph), pooling, date-clustered DM,
  verdict, per-fold + spike robustness, atomic per-horizon checkpoints.
- `test/test_leaf_graph.py` (24 tests) + `test/conftest.py`.
- `requirements/requirements.md`, `design/design.md`, `code_review/code_review_2026-09-18.md`.
- Results: `results/gamma_gbm/leaf_graph_hose_h{1,5,10,22}.json` (all n_folds=8).

## HOSE results (8 folds each, seed-ensemble FM.SEEDS, QLIKE floor FM.FL)
Primary comparison isolates the graph: XGB+leafgraph vs the identical un-smoothed XGB base.

| h | GBME | XGB | XGB+leafgraph | gain vs XGB | DM p vs XGB | beats | ex-spike gain | ex-spike p | ex-spike beats | vs GBME p | α_mean | fits |
|---|------|-----|---------------|-------------|-------------|-------|---------------|-----------|----------------|-----------|--------|------|
| 1  | 1.5719 | 1.5689 | 1.5650 | +0.25% | 0.0004 | True  | +0.25% | 0.0014 | True  | 0.0000 | 0.35 | all ok |
| 5  | 1.6479 | 1.6512 | 1.6469 | +0.26% | 0.0000 | True  | +0.20% | 0.0002 | True  | 0.2817 | 0.29 | all ok |
| 10 | 1.6856 | 1.6838 | 1.6817 | +0.12% | 0.0046 | True  | +0.12% | 0.0074 | True  | 0.0133 | 0.26 | all ok |
| 22 | 1.7270 | 1.7270 | 1.7264 | +0.03% | 0.3376 | False | +0.03% | 0.4911 | False | 0.6552 | 0.28 | all ok |

Per-fold α (val-fit): h1 [0,.4,.4,.7,.5,.5,.1,.2], h5 [.2,.2,.2,.6,.6,.4,0,.1], h10 [0,.1,.2,.2,.9,.2,.2,.3],
h22 [.3,.6,.5,0,.8,0,0,0]. Non-zero α on most folds — the graph is weighted in, not driven to 0.

## Verdict — NOT the expected clean NO-GO; a small, denoising-driven marginal win over raw XGB
- **Pre-registered kill criterion (beats XGB at BOTH h1 and h5, gain>0, DM p<0.05, spike-robust): MET.** h1 and
  h5 both beat the un-smoothed XGB base with DM p<0.001 and survive spike-exclusion. h10 also beats; h22 is null.
- **But the effect is tiny and does not beat the deployed GBME champion.** Gains decay with horizon
  (+0.26% → +0.03%) and are practically negligible (<0.3%). XGB+leafgraph is DM-distinguishable from GBME only
  at h1 (p=0.0000) and h10 (p=0.0133); at h5 (p=0.28) and h22 (p=0.66) it is statistically indistinguishable
  from GBME. At h1 the un-smoothed XGB already beats GBME, so part of the h1 "vs GBME" gap is XGB≠GBME, not the
  graph.
- **Mechanism = cross-sectional shrinkage, not new information.** Averaging predictions of stocks the GBM's
  trees already treat as similar is a variance-reduction (denoising) of a noisy point forecast. This is fully
  consistent with the a-priori expectation that the leaf-graph re-encodes the same 8 own-history features and
  carries no new signal — the only benefit is mild forecast shrinkage, which fades by h22 and never separates
  from the HGBR champion at the mid/long horizons.

Conclusion: the leaf-graph does NOT establish that a graph derived from the champion GBM's own trees carries
useful own-history-independent structure. It yields a small, DM-significant point-forecast shrinkage over the
raw XGB base at short/mid horizons (null at h22) that does not beat the deployed GBME champion — i.e. no
practically meaningful lift.

## Tests + coverage
- `python -m pytest test/` → **24 passed**.
- diff-cover proxy (branch coverage on the three code modules): **C0 line = 100%, C1 branch = 100%**
  (247 stmts / 44 branches, 0 miss) — clears the C0=100% / C1≥95% gate.
- Cover: leaf-similarity (identical→1, disjoint→0, partial fraction), kNN top-k + singleton, α-smoothing
  (α=0 identity, α=1 neighbour-mean, monotone, per-day causal), α-fit both directions, booster plumbing +
  cap clip, `_pool_doc` per-fold/spike/keep-empty, `_load_earn` sp500/hose-missing/real-parquet, `_safe_dm`
  degenerate + ValueError, run smoke hose/sp500/skip-empty/omit-robustness.

## Code review
`code_review/code_review_2026-09-18.md` — adversarial 4-layer pass (leakage/causality, numerical safety,
α-fit integrity, config hygiene). No critical/major findings. Causality verified: booster train-only, per-day
cross-section smoothing, α fit on OOS val frozen for test, embargo preserved. Two accepted minors (train-metric
smoothing cost when α>0; seed-0 booster for the graph structure vs seed-ensemble base).

## Gates
- ruff pyflakes (F): clean. config-hardcode scan: 0 BLOCK on all three modules (unique `leaf_graph_config.py`
  avoids the `config.py` sys.modules collision). PostToolUse tier passes.
- Over/under-fit evidence: every JSON carries `metrics`/`train_metrics`/`val_metrics` + `fit_diagnostics`;
  all three models (incl. the learned XGB / XGB+leafgraph) classify `ok` on the pooled 8-fold result.
- Data-quality gate (Pandera/Evidently): N/A (no data change; reads existing processed_enriched/hose +
  hose_earnings_combined.parquet).

## Risks / follow-ups
- Memory: the full 4-horizon run exhausted process RAM at h22 (train-metric smoothing accumulates arrays when
  α>0); h22 was completed in a fresh single-horizon process. A future robustness fix would stream/free per-fold
  train arrays or skip the train-metric smoothing (α=0 fast path already covers the inert case). Not required
  for the result (all 4 JSONs are complete 8-fold pools).
- The marginal short-horizon win is a denoising effect; if pursued, compare against a plain cross-sectional
  mean-shrinkage baseline (no leaf-graph) to confirm the graph structure adds nothing beyond shrinkage.

---

## SP500 cross-market check (added 2026-09-18) — leaf-graph does NOT transfer

Ran the identical leaf-graph pipeline on SP500 (8 folds, val-fit α, per horizon). Result: **NO-GO on SP500** —
the val-fit α collapses to ~0 (mean 0.00–0.05 vs HOSE's 0.26–0.35), so the graph self-disables; gains are ~0 or
slightly negative (h1 −0.003% p=0.66, h5 −0.009% p=0.23, h10 −0.036% p=0.014 slightly worse, h22 0.0% p=1.0).
`results/gamma_gbm/leaf_graph_sp500_h{1,5,10,22}.json`.

**Interpretation:** the leaf-graph is a **thin-market denoising phenomenon**. On HOSE (~400 thin VN stocks) per-stock
forecasts are noisier, so cross-sectional smoothing over GBM-leaf-similar stocks helps (+0.12–0.26%, DM-sig,
spike-robust). On the liquid SP500 the GBM forecasts are already low-noise, so smoothing adds nothing and the
causal α-selection correctly drives α→0 (no overfitting to a useless graph — validates the method's honesty).
Scope for the paper: the lever is HOSE-specific, not universal.

## v2 RF-GAP / KeRF (added 2026-09-18) — no improvement over the simple v1 kNN
The principled proximity upgrades (`baselines/2026-09-18_leaf_graph_rfgap`, RF-GAP row-stochastic soft mean +
KeRF `1/leaf_population` down-weighting) are all significantly WORSE than v1's hard leaf-Hamming top-k kNN at
every horizon (RF-GAP vs kNN −0.09…−0.27%, DM-sig), and KeRF is NOT more spike-robust (v1 kNN has the lowest
ex-spike QLIKE). The soft weighted mean over all leaf co-members dilutes more than the hard top-k. **v1's simple
kNN remains the champion leaf-graph version.**
