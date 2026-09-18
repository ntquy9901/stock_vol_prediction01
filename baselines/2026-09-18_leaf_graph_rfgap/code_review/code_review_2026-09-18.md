# Adversarial code review — leaf-graph v2 (RF-GAP / KeRF), 2026-09-18

Scope: `code/rfgap_config.py`, `code/rfgap.py`, `code/run_rfgap.py`, `test/test_rfgap.py`.
Reviewed adversarially for leakage/causality, proximity correctness, numerical safety, the streaming-metric
RAM fix, and config hygiene. `archive/` and vendored trees out of scope.

## Layer 1 — Leakage / causality (central risk for a contemporaneous graph)
- **Booster trained on the past only.** `fit_booster` sees `trf_e` = train window minus the val slice, strictly
  before `ts - embargo`. Test/val leaf-vectors are computed by *applying* that fixed booster (`leaf_matrix`),
  never by refitting on future rows. No target leakage into the tree structure. PASS.
- **Per-day cross-section only.** `smooth_all` groups by `date` and smooths each date independently. Under the
  temporal split train/val/test occupy disjoint date ranges, so a test day's smoothed prediction depends only on
  other stocks *on that same test day*. Verified by `test_smoothing_is_per_day_causal` (perturbing another day's
  rows leaves the target day unchanged). PASS.
- **RF-GAP self-exclusion.** Every scheme excludes self: `day_similarity` diagonal is masked in `knn_neighbour_mean`;
  RF-GAP masks the diagonal (`np.fill_diagonal(same, False)`) and the LOO mean subtracts the stock's own value
  `(GS - pred)/(GC-1)`. `test_rfgap_weights_are_row_stochastic_and_self_zero` asserts `W[i,i]=0`. So a stock
  never smooths toward itself — the RF-GAP leave-one-out property. PASS.
- **α fit on genuine OOS val.** The val slice is excluded from `trf_e`; α is chosen per scheme to minimise val
  QLIKE then frozen for test (`_fold_predictions`). No test information touches α. PASS.
- **Embargo preserved.** `embargo = int(1.6h)+5` days (inherited from the sibling battery). PASS.

## Layer 2 — Proximity correctness (the v2 contribution)
- **Two implementations, cross-checked.** `day_weights` (explicit m×m reference) and `_rfgap_neighbour_mean`
  (vectorised bincount hot path) are asserted equal for rfgap and rfgap_kerf (inv/invsqrt) over random leaves in
  `test_vectorised_neighbour_mean_matches_day_weights` — guards against the two paths silently diverging. PASS.
- **RF-GAP is a proper proximity.** `test_rfgap_weights_are_row_stochastic_and_self_zero`: rows sum to 1, self 0,
  non-negative. Derivation: per valid tree each row's LOO distribution sums to 1, so `num` row-sum = `den`, and
  `W = num/den` is row-stochastic. PASS.
- **KeRF down-weights large leaves.** `test_kerf_downweights_neighbours_reached_through_large_leaves`: a neighbour
  reached only via a size-4 leaf gets strictly less weight under KeRF(inv) than under raw RF-GAP (1/9 vs 1/6),
  while the small-leaf neighbour gains (2/3 vs 1/2); both stay row-stochastic. Exact hand-computed values
  asserted. This is the mechanism targeting HOSE storm-day noise. PASS.
- **Isolated stock keeps its base prediction.** A stock with no leaf co-member in any tree gets an all-zero `W`
  row / zero `den`; `neighbour_mean` returns its own base (not a shrink-to-zero). Tested
  (`test_rfgap_isolated_stock_row_is_zero`, `test_singleton_cross_section_returns_base`). PASS.
- **knn vs rfgap distinction is honest.** Hard top-k always assigns k neighbours even to a disjoint stock (v1
  behaviour), whereas RF-GAP leaves a co-member-less stock unsmoothed. `test_smooth_day_alpha0_identity_and_
  alpha1_neighbour_mean` asserts both behaviours explicitly, so the difference is documented, not hidden. PASS.

## Layer 3 — Numerical safety & the RAM fix
- **Gamma exp-link overflow.** XGB `reg:gamma` clipped to `[FL, PRED_CAP]` in `predict_booster`; the smoothed
  convex combination re-clipped in `_fold_predictions` (`test_predict_booster_clips_to_cap_and_seed_ensemble`).
  A convex combination of in-range values stays in range; the clip is belt-and-braces. PASS.
- **No divide-by-zero in weights.** In `_rfgap_neighbour_mean`, `GC>=1` always (a stock is in its own leaf), so
  `1/GC`/`1/sqrt(GC)` are finite; invalid trees (`GC==1`) are masked to weight 0; `den==0` (fully isolated) →
  own pred. `denom_loo` uses `np.where(valid, GC-1, 1)` so the `GC==1` slot never divides by zero. PASS.
- **QLIKE floor consistency.** Every model scores QLIKE with the SAME floor `FM.FL` (single source), so the
  comparison basis is identical across GBME / XGB / all smoothed variants (H2 lesson from eda_gnn). PASS.
- **Streaming metrics reproduce pooled metrics exactly.** `_Stream` accumulates sufficient statistics and
  `finalize()` matches direct `_metrics5` to `rtol=1e-9` (`test_stream_matches_direct_pooled_metrics`), incl. the
  constant-`y` R²=0 guard (`test_stream_and_metrics5_constant_y_r2_zero`). This is the fix for the v1 h22 OOM:
  train arrays are folded in then dropped, never all held at once. One horizon per fresh process adds isolation.
  PASS.
- **DM degeneracy.** `_safe_dm` returns p=1.0 for identical loss series and for the HLN/HAC ValueError cases, so a
  degenerate fold (e.g. all α=0 → identical to XGB) cannot crash the pool. Tested. PASS.

## Layer 4 — Falsification integrity & config hygiene
- **Shared base isolates the graph and the v2-over-v1 delta.** XGB and every smoothed variant share the identical
  seed-ensembled base; `vs_XGB` measures each scheme's smoothing alone and `vs_knn` measures RF-GAP(+KeRF) over
  the v1 graph. PASS.
- **Pre-registered success is explicit.** `success()` requires a v2 variant to beat XGB+knn on QLIKE+DM at ≥1
  horizon, OR be as-good-and-more-spike-robust — coded in `v2_vs_v1`, not decided post-hoc. Tested for both the
  beats path and the robustness path (`test_verdict_and_success`). PASS.
- **α grid includes 0.** `ALPHA_GRID` starts at 0.0, so any scheme can be selected OUT entirely; ties keep the
  smaller α (ascending grid, strict `<`). The observed smoke fold picked α=0 for knn/rfgap (inert) and α=0.10 for
  rfgap_kerf — the mechanism can turn each graph off. PASS.
- **Config single-sourced.** All tunables (XGB capacity, K, schemes, KeRF func, α grid, horizons, gates, spike
  windows) live only in `rfgap_config.py`; no bare magic numbers in the pipeline. `postgen_gate` config-hardcode
  scan is satisfied (inline-commented constants). PASS.

## Findings
No HIGH/MAJOR issues. Two MINOR notes (accepted, not blocking):
- **M1 (accepted):** RF-GAP train smoothing over the full train window is the compute cost (vectorised O(m·T)
  per day); mitigated by streaming + per-horizon processes. Not a correctness issue.
- **M2 (accepted):** KeRF uses `KERF_FUNC='inv'` for the reported run; `'invsqrt'` is available as a config
  switch but not run this pass (would be a follow-up sensitivity, not required by the pre-registration).

## Verdict
Code is causal, numerically guarded, memory-safe (streaming), and the RF-GAP/KeRF proximities are correct
(row-stochastic, self-excluded, large-leaf down-weighted) with the two implementations cross-checked. 36 tests
pass; C0=100% / C1=100% (branch) on all four new modules. Approved to run on HOSE; the pre-registered v2-vs-v1
verdict is read from the completed 8-fold JSONs (below in the summary report).
