# Adversarial code review — GBM + leaf-cooccurrence graph (2026-09-18)

Scope: `code/leaf_graph_config.py`, `code/leaf_graph.py`, `code/run_leaf_graph.py`, `test/test_leaf_graph.py`.
Reviewed adversarially for leakage/causality, numerical safety, α-fit correctness, and config hygiene.
`archive/` and vendored trees out of scope.

## Layer 1 — Leakage / causality (the central risk for a contemporaneous graph)

- **Booster trained on the past only.** `fit_booster` sees `trf_e` = train window minus the val slice, strictly
  before `ts - embargo`. The leaf-vectors of test/val rows are computed by *applying* that fixed booster
  (`leaf_matrix`), never by refitting on future rows. No target leakage into the tree structure. PASS.
- **Per-day cross-section only.** `smooth_all` groups rows by `date` and smooths each date's cross-section
  independently. Because train/val/test occupy disjoint date ranges under the temporal split, a test day's
  smoothed prediction depends only on other stocks *on that same test day* — no future rows, no train/test
  mixing. Verified by `test_smoothing_is_per_day_causal` (perturbing another day's rows leaves the target day
  unchanged). PASS.
- **α fit on genuine OOS val.** The val slice is out-of-sample for the booster (excluded from `trf_e`), and α is
  chosen to minimise val QLIKE then frozen for test (`_fold_predictions`). No test information touches α. PASS.
- **Embargo preserved.** `embargo = int(1.6h)+5` days between train end and test start (inherited from the
  sibling battery) prevents horizon-`h` target overlap at the boundary. PASS.

## Layer 2 — Numerical safety

- **Gamma exp-link overflow.** XGB `reg:gamma` predictions are clipped to `[FL, PRED_CAP]` in
  `predict_booster`; the smoothed convex combination is re-clipped in `_fold_predictions`. A convex combination
  of values in `[FL, PRED_CAP]` stays in range, and the explicit clip is a belt-and-braces guard
  (`test_predict_booster_clips_to_cap`). PASS.
- **Singleton / small cross-sections.** `knn_neighbour_mean` returns the base prediction when `m == 1`
  (no neighbours); `kk = min(k, m-1)` caps the neighbour count so `argpartition` never indexes out of range on
  a day with fewer than `k+1` stocks. Covered by the singleton tests. PASS.
- **QLIKE floor consistency.** All models score QLIKE with the SAME floor `FM.FL` (single source), so the
  comparison basis is identical across GBME / XGB / XGB+leafgraph (H2 lesson from the eda_gnn baseline). PASS.
- **DM degeneracy.** `_safe_dm` returns p=1.0 for identical loss series (`np.allclose`) and for the HLN/HAC
  ValueError cases (too few dates), so a degenerate fold cannot crash the pool. PASS.

## Layer 3 — α-fit correctness & falsification integrity

- **Shared base isolates the graph.** XGB and XGB+leafgraph use the identical seed-ensembled base; the DM
  `XGB+leafgraph vs XGB` therefore measures the smoothing alone (not an XGB-vs-HGBR artefact). PASS.
- **α grid includes 0.** `ALPHA_GRID` starts at 0.0, so the graph can be selected OUT entirely; ties keep the
  smaller α (strict `<` update in `fit_alpha`). If the graph is useless, α=0 wins and XGB+leafgraph ≡ XGB — an
  honest inert outcome, not a masked failure. Verified both directions
  (`test_fit_alpha_prefers_lower_qlike...`, `..._can_select_positive_alpha...`). PASS.
- **α reported per fold.** `doc["alpha"]["per_fold"]` + `mean` are recorded so a reader can see whether the
  graph was ever weighted, independent of the DM verdict. PASS.
- **Pre-registered kill criterion.** `verdict` requires gain>`GAIN_MIN` AND DM p<`DM_ALPHA`; `success` requires
  BOTH h1 and h5 to beat XGB. No post-hoc horizon cherry-picking. PASS.

## Layer 4 — Config hygiene & style

- All tunable constants live in `leaf_graph_config.py` with inline comments; the config-hardcode scanner reports
  0 BLOCK on every module. No bare magic numbers in the pipeline. PASS.
- Unique config module name `leaf_graph_config.py` (not bare `config.py`) avoids the `sys.modules` collision
  with `submission/soict_lstm_gat/config.py` documented in the project feedback. PASS.
- ruff pyflakes (F) clean. Matches sibling baseline style (`;`-joined statements tolerated house-style). PASS.

## Findings
No critical or major findings. Minor/observations:
- (Minor, accepted) When a fold selects α>0, train-row smoothing is recomputed over the full train date range
  for the train-metric block; this is honest but the dominant cost. The α=0 identity fast path keeps the
  expected NO-GO case cheap. No action.
- (Minor, accepted) The leaf-graph uses the seed-0 booster's tree structure while the base is a 3-seed
  ensemble; this is a deliberate design choice (fixed deterministic graph) documented in `design.md`. No action.

## Verdict
Approved. Causality and numerical guards are sound; the falsification is honest (α can go to 0, kill criterion
pre-registered, DM date-clustered and spike-robust). Final numbers reported in the summary report.
