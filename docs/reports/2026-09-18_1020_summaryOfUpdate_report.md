# Summary of update — leaf-graph v2 (RF-GAP / KeRF): NO improvement over the v1 simple graph

Date: 2026-09-18. Baseline: `baselines/2026-09-18_leaf_graph_rfgap/`.

## What changed
Built and ran **leaf-graph v2**, the literature-grounded upgrade of the v1 leaf-Hamming kNN smoother
(`baselines/2026-09-18_gbm_leaf_graph`, which passed on HOSE at h1/h5/h10). v2 adds two RF-proximity weightings
selectable via config, all smoothing the SAME shared un-smoothed XGBoost gamma base so the graph effect is
isolated:
- **RF-GAP** (arXiv 2307.01077): a proper regression proximity — a leakage-safe leave-one-out neighbour
  distribution over the boosted-tree leaves, renormalised so each stock's neighbour weights sum to 1 and no
  stock smooths itself (a soft weighted mean, not a hard kNN).
- **RF-GAP + KeRF** (arXiv 2601.02735): the same, with each tree's contribution scaled by
  `KERF_FUNC(leaf_population)` (`1/pop`) so a giant storm-day leaf is down-weighted — aimed at HOSE
  spike-robustness.

## Files (path -> purpose)
- `code/rfgap_config.py` — all tunable constants (XGB capacity, K, `WEIGHT_SCHEMES`, `KERF_FUNC`, α grid,
  horizons, kill criterion, spike windows). Single source of truth.
- `code/rfgap.py` — booster fit/predict, leaf matrix, leaf-Hamming (knn), RF-GAP/KeRF proximity
  (`day_weights` reference matrix + vectorised `_rfgap_neighbour_mean` hot path, cross-checked), scheme-aware
  smoothing + α fit.
- `code/run_rfgap.py` — walk-forward driver; **streams train metrics** (`_Stream` sufficient statistics) so the
  large train arrays are never all held in memory (fixes the v1 h22 OOM); one horizon per fresh process
  (`--horizon`); pooling, date-clustered DM (vs XGB / GBME / v1 knn), verdict, spike, `v2_vs_v1` decision,
  atomic per-fold checkpoints.
- `test/test_rfgap.py` (+ `conftest.py`) — 36 tests.
- `requirements/`, `design/`, `code_review/` — SDD artifacts.
- `results/gamma_gbm/rfgap_hose_h{1,5,10,22}.json` — full 8-fold result docs (verified `n_folds==8`).

## Tests + coverage
`.venv_gpu_encode\Scripts\python.exe -m pytest baselines/2026-09-18_leaf_graph_rfgap/test/` — **36 passed**.
diff-cover proxy via `--cov-branch` on the new code: **C0 line = 100%, C1 branch = 100%** on all four modules
(rfgap.py 140 stmts / 38 branch, run_rfgap.py 244 / 54, rfgap_config.py 20, `__init__` 0 — zero missing).
Tests cover: RF-GAP proper-normalisation (row-stochastic, self-zero), KeRF large-leaf down-weighting (a neighbour
reached via a size-4 leaf gets 1/9 vs RF-GAP 1/6, small-leaf neighbour 2/3 vs 1/2), vectorised==reference-matrix
consistency (rfgap / kerf inv+invsqrt), per-day causality, streaming-metric==direct-pooled equivalence,
`_pool_doc`/`_dm_block`/`_spike_for`/`_safe_dm`/`_load_earn`/verdict/success, and run smokes (hose + sp500)
carrying the gate evidence keys.

## Commands run (real)
- Full HOSE walk-forward, all 4 horizons, 3 seeds, 8 folds each (one fresh process per horizon; h22 re-run to a
  complete 8-fold pool after the chain wrapper was interrupted). `results/gamma_gbm/rfgap_hose_h*.json`.
- `ruff check --select F` on code + tests → clean.
- `postgen_gate.py` (config-hardcode + ruff-F) on all three `.py` → exit 0.
- `overfit_check.check_result_evidence` on all 4 JSONs → `gate_ok=True`, every model `fit=ok`, `n_folds=8`.

## Final HOSE table (pooled 8 folds, QLIKE; DM date-clustered; positive gain = improvement)
```
h    model            QLIKE     vs XGB    (p)        vs v1-knn  (p)
1    GBME             1.5719
1    XGB (base)       1.5689
1    XGB+knn (v1)     1.5650   +0.248%  (0.0004)*     —
1    XGB+rfgap        1.5684   +0.032%  (0.4764)     -0.217%  (0.0000)
1    XGB+rfgap+kerf   1.5669   +0.130%  (0.0145)*    -0.119%  (0.0059)
5    GBME             1.6479
5    XGB (base)       1.6512
5    XGB+knn (v1)     1.6469   +0.258%  (0.0000)*     —
5    XGB+rfgap        1.6513   -0.010%  (0.4663)     -0.268%  (0.0000)
5    XGB+rfgap+kerf   1.6511   +0.007%  (0.7273)     -0.251%  (0.0000)
10   GBME             1.6856
10   XGB (base)       1.6838
10   XGB+knn (v1)     1.6817   +0.123%  (0.0046)*     —
10   XGB+rfgap        1.6846   -0.050%  (0.0262)     -0.173%  (0.0000)
10   XGB+rfgap+kerf   1.6852   -0.082%  (0.0239)     -0.205%  (0.0000)
22   GBME             1.7270
22   XGB (base)       1.7270
22   XGB+knn (v1)     1.7264   +0.033%  (0.3376)      —
22   XGB+rfgap        1.7280   -0.058%  (0.0325)     -0.092%  (0.0060)
22   XGB+rfgap+kerf   1.7278   -0.050%  (0.0667)     -0.083%  (0.0062)
```
`*` = beats the XGB base (gain>0 AND DM p<0.05). v1's h1/h5/h10 gains reproduce exactly (+0.248 / +0.258 /
+0.123%, h22 ns), confirming the harness.

### Spike-robustness (ex COVID-2020 / 2022 / Apr-2025), pooled QLIKE
```
h    XGB      XGB+knn   XGB+rfgap  XGB+rfgap+kerf
1    1.5536   1.5497    1.5539     1.5528
5    1.6355   1.6323    1.6357     1.6359
10   1.6775   1.6755    1.6785     1.6791
```
Ex-spike, **v1 kNN is still the lowest at every horizon**; KeRF is NOT more spike-robust than kNN (its central
hypothesis fails on HOSE).

## Verdict — pre-registered kill criterion: NO-GO for v2
A v2 variant had to beat v1's `XGB+knn` on QLIKE at ≥1 horizon (gain>0 AND DM p<0.05 vs knn), OR be at least as
good while strictly more spike-robust. **Neither happened at any horizon:**
- RF-GAP is inert-to-slightly-negative vs the XGB base and is **significantly WORSE than v1 kNN at all four
  horizons** (−0.09% to −0.27%, DM p ≤ 0.006).
- RF-GAP+KeRF beats the base only at h1 (+0.13%) but by LESS than kNN, is worse than the base at h10/h22, and is
  **significantly worse than kNN at all four horizons** (−0.08% to −0.25%). It is not more spike-robust than kNN.
- The α grid frequently selected α→0 for RF-GAP/KeRF (graph turned off), while kNN kept α 0.1–0.9.

Interpretation: the "principled" RF-GAP soft weighted mean over *all* leaf co-members dilutes the signal more
than v1's hard top-k neighbour mean; KeRF's large-leaf down-weighting recovers only part of that and never
overtakes the simpler graph. **v1's hard leaf-Hamming kNN remains the champion version; v2 is reported as no
improvement over the simple graph.** Result is honest and gate-clean either way.

## Code review
Adversarial 4-layer review in `code_review/code_review_2026-09-18.md` (leakage/causality, proximity correctness,
numerical safety + the streaming RAM fix, falsification integrity + config hygiene). No HIGH/MAJOR findings; two
accepted MINOR notes (RF-GAP train-smoothing compute cost — mitigated by streaming + per-horizon processes; KeRF
`invsqrt` available but only `inv` run this pass). Key properties are test-backed.

## Performance / batching
Predictions on batched feature matrices; RF-GAP neighbour mean fully vectorised (one `bincount` per day, no
per-pair Python loop); α=0 identity fast path skips graph construction. Train metrics streamed (no cross-fold
train-array accumulation) — the direct fix for the v1 h22 OOM. All 4 horizons completed as full 8-fold pools.

## Risks / follow-ups
- KeRF `invsqrt` sensitivity not run (config switch present) — optional follow-up; unlikely to change the NO-GO
  given `inv` never overtook kNN.
- SP500 not run here (the driver supports `sp500`; a separate agent was running the SP500 leaf-graph job).
- No git operations performed (coordinator pushes). `archive/` out of scope.
