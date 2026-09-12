# Adversarial code review — Complex-Network baseline (2026-09-12)

Own adversarial pass over `code/*.py`, focused on leakage and correctness. Scope excludes `archive/`.

## Leakage / correctness checks

| # | Risk checked | Where | Result |
|---|---|---|---|
| 1 | **Causal correlation window** — a network dated `d0` must use only rows strictly before `d0`. | `topology.build_topo_windows`: window `iloc[i-win:i]`, `d0 = dates[i]` (row `i` excluded). | OK — window is `[i-win, i)`, strictly past. |
| 2 | **Strictly-future target** — Exp A target must lie entirely at/after `d0`. | `market_index.future_targets`: `p = searchsorted(dates, d0, 'left')`, block `p..p+L-1`. | OK — block starts at the first index row `≥ d0`; features come only from stock data before `d0`. The block's first log-return references `close[p-1]` (the return INTO d0) — this is the target series, not a feature, so it introduces no feature leakage (matches paper eq. 4). |
| 3 | **Walk-forward embargo** — train windows whose target period overlaps the test feature date leak. | `run_index.walk_forward`: `mask = target_end ≤ d0_j`; a window is trainable only when its full L-day target completed by the test feature date. | OK — the test row itself (`target_end = d0_j + L > d0_j`) is always excluded; verified by `test_walk_forward_excludes_leaked_future_target` (a poison row with `target_end` just after `d0_j` is provably dropped). |
| 4 | **Scaler fit on train only.** | `run_index.walk_forward`: `StandardScaler().fit(X[mask])`, then `transform` train and the single test row. | OK — no test statistics enter the scaler; a fresh model per fold. |
| 5 | **Min-history guard.** | `MIN_TRAIN_WINDOWS` skip; `MIN_COMMON_TICKERS` skip. | OK — too-small train sets are skipped, not scored on noise. |
| 6 | **Exp B target shift + fold embargo.** | `FM.panel` shifts `parkinson_variance` by `-h`; fold train `date < ts - embargo`, `embargo = int(h·1.6)+5` days. | OK — inherits the delivered `full_matrix` protocol. |
| 7 | **Identical QLIKE floor across compared models (DM validity).** | `run_gbm`: both GBM and GBM+topo scored with `M.per_obs_qlike(..., floor=FL)`, same `FL = FM.FL = 1e-8`; both DM loss series aggregated over the same `dates`. | OK — same floor, same basis. |
| 8 | **Market-level broadcast is not cross-sectional leakage.** | Topology 7-vector is identical for every stock on a date and built only from past windows. | OK — no per-stock future information. |
| 9 | **Config-hardcode gate.** | All tunable constants live in `config.py`; modules read `config.*` at call time. | OK — post-generate gate passed on every file. |
| 10 | **Overfit-evidence gate.** | Both result JSONs carry `train`/`test` metrics + `fit_diagnostics` per model. | OK — Exp A: `train_r2`/`test_r2` + verdict per model×target; Exp B: train/test QLIKE + verdict per model. |
| 11 | **Module-name shadowing.** | FM's import chain prepends `submission/soict_lstm_gat` (which has its own `config.py`) to `sys.path`. | FIXED — local `config`/`topology`/`market_index` are imported BEFORE `full_matrix`, binding the names to this baseline; confirmed by a real run. |
| 12 | **Refinement 1 — REAL `ln(volume)`, not `volume_zscore_22`.** | `volume_io.load_log_volume` reads raw OHLCV volume, masks `volume<=0 → NaN`, returns `ln(volume)`; `topology.build_topo_windows` builds the volume-correlation matrix from it (aligned to the return calendar). | OK — the paper's exact volume transform; `volume_zscore_22` is no longer used anywhere in the topology path. Real-file smoke `test_load_log_volume_real_hose_slice` loads a real HOSE ticker. |
| 13 | **Refinement 2 — per-window ticker count N.** | `build_topo_windows` returns `n_by_window` (common-ticker count per window); `run_index` reports `n_tickers_per_window` {min,median,max} over the scored windows. | OK — sample size visible; NaN-safe `_n_stats`. |
| 14 | **Refinement 4 — feature importance is causal (per-fold, train-only).** | `_feature_importance` uses RF `feature_importances_` or standardized LR coef (`coef_ × train-feature-std`) computed on the SCALED TRAIN matrix of each fold, then averaged over folds. | OK — no test data enters the importance; shape == 7 (verified by `test_walk_forward_rf_importance_shape_seven`). |
| 15 | **Refinement 3 — WIN vs WIN_ROBUST vs α-grid distinct.** | Headline WIN=66; robustness re-runs α∈{0,0.5,0.7,1.0} at WIN and WIN_ROBUST=132 at ALPHA — three distinct quantities (WIN, STEP=22, L=22) never collapsed. | OK — matches design §7. |

## Performance

- Topology recomputed once per `STEP=22` days (not per day); betweenness/eigenvector bounded by node count
  (≤~620 HOSE / ≤~500 S&P 500) × ~90-130 windows per build. Robustness reruns the build for each α and for
  win=132 — the dominant cost — but each is a one-off feature build, then cached.
- Exp B reuses the vectorised seed-ensembled gamma-GBM; no batch=1 main-thread hot loop over samples.

## Verdict (self-review)

No HIGH/MEDIUM leakage or correctness findings after the shadowing fix and the four refinements (real
`ln(volume)`, per-window N, WIN=66/132, feature importance). Tests: 25 passed, 100% line / 100% branch on
changed `code/*.py` (config, volume_io, topology, market_index, run_index, run_gbm).

## Independent adversarial review (second reviewer, 2026-09-12)

A separate reviewer that did not write the code re-ran the leakage/correctness hunt. Found **0 HIGH**, **2
MEDIUM**, **5 LOW**. Resolution:

| # | Sev | Finding | Resolution |
|---|---|---|---|
| M1 | MEDIUM | `run_gbm._merge_topo` fill used a plain `.ffill()` over `FM.panel`'s ticker-blocked concat, which can drag one ticker's late-dated topology into the next ticker's early rows (cross-series, non-causal). Not exploitable in the current config (leaked rows are pre-2015, dropped by `TRAIN_START`) but redundant with the per-date ffill already done upstream. | **FIXED** — `run_gbm.py` now uses `a.groupby("ticker")[c].ffill().fillna(0.0)` (per-ticker), removing the cross-boundary fill. |
| M2 | MEDIUM | No real-data-sample smoke exercised the topology/walk-forward core (only the two loaders had real smokes). | **FIXED** — added `test_build_topo_windows_real_hose_slice`: builds topology on 12 real HOSE frames + real `ln(volume)` and asserts finite metrics, strictly-increasing window dates, and N∈[5, n_tickers]. |
| L1 | LOW | `_feature_importance` multiplied LR `coef_` by `x_train.std(axis=0)`, but `x_train` is already StandardScaler-transformed (std≈1), so the factor was a no-op and the docstring overstated it. | **FIXED** — `run_index.py` returns `coef_` directly (already the standardized, per-1-SD coefficient); docstring corrected. LR importance values unchanged. |
| L2 | LOW | Silent `0.0` fallbacks (eigenvector `except`, `np.nan_to_num` on the corr matrix) coerce zero-variance / suspended tickers to isolated nodes without logging. | **Accepted** — defensible network semantics (an in-window flat ticker genuinely has no correlation edge); documented in `topology.global_feats`. No result impact. |
| L3 | LOW | Walk-forward embargo `target_end <= d0[j]` compares to the feature-window END, 1 day looser than the design prose (feature START). Still leak-free (at most a same-day touch on `d0_j`, which is the test target's first day). | **Accepted** — leak-free; the 1-day touch is on the target's opening day, not a feature. |
| L4 | LOW | `test_walk_forward_returns_aligned_arrays` uses `target_end == d0` (self-eligible fixture); it only asserts alignment/finiteness. | **Accepted** — causality is proven by `test_walk_forward_excludes_leaked_future_target` (a poison future-target row must be excluded or the prediction diverges). |
| L5 | LOW | `robustness()` rebuilds the topology 5× (α-grid + win132); the α-grid rebuilds recompute identical correlation matrices (topology at fixed WIN is α-independent before the blend). | **Accepted** — perf only; sample is ~90 windows, cost tolerable. |

Reviewer's "tried to break and could not" list (Exp-A target-into-feature leakage, `searchsorted` boundary,
scaler/model/importance leakage, `common`-ticker intersection, Exp-B DM validity, the 7-metric definitions)
matches the self-review.

## Verdict (final)

0 HIGH, 0 unresolved MEDIUM. M1/M2/L1 fixed; L2–L5 accepted with rationale. Tests: 33 passed, 100% line /
100% branch on all changed `code/*.py` (adds `diag_gbm`, `build_topo_diag_html`). No result-changing defect;
the M1 fix leaves the current HOSE numbers unchanged (leaked rows were already outside the train window).
