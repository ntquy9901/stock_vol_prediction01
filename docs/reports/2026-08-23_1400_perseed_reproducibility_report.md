# Summary of update — per-seed reporting (F1) + pinned reproducible config (2026-08-23)

## Trigger
External review (Codex rerun-03) on `deliverables_20260822`, 4 findings. Verified against current code:
- **F1 (major, CONFIRMED):** the tables' "5-seed means" were metrics of the seed-AVERAGED ensemble
  prediction (`_ens` then `_metrics`), not the mean of seed-level metrics. The ensemble is generally lower,
  hiding seed sensitivity; the gap is of the same order as the graph effects (e.g. VN30 h1 LSTM+GAT QLIKE
  ensemble 0.607 vs per-seed mean 0.660 ± 0.064).
- **F2 (important):** GARCH random-sign pseudo-returns — largely rebutted (symmetric GARCH(1,1) variance
  depends only on squared residuals; measured sign-seed variation < 5%); disclosed.
- **F3 (important):** no multiple-comparison adjustment — disclosed (headline results are nulls, robust;
  significant graph-vs-LSTM contrasts labelled unadjusted exploratory).
- **F4 (minor):** LSTM vs LSTM+GAT not capacity-matched — disclosed.

## Secondary finding (reproducibility)
While fixing F1, the stored deep-model numbers were found **not to reproduce** from the committed config:
HAR-X / GARCH / data reproduce exactly (GARCH basis-guard passed; training is deterministic — same seed+cfg
gives identical output), but the deep ensemble did not match at any tried batch size (16/32/512). The
original run's `batch_size`/`epochs` were never recorded. Root cause: unrecorded training config.

## Fix
- `run_masked_rich.py`: added `seed_metric_stats()` (per-seed mean/std/min/max + raw per-seed; TDD test in
  `test_masked_rich.py`); `run()` now records a `config` block (batch_size, epochs, patience, min_epochs) and
  a `metrics_per_seed` block in every result.json. Ensemble metrics + DM kept for the DM tests.
- Pinned **batch_size = 32** for all panels and re-ran all 20 cells (5 panels × 4 horizons). Deterministic,
  documented, reproducible. HOSE/HNX/SP500 reproduce their prior ensemble (they were already batch=32); the
  deterministic HAR-X/GARCH basis-guard passed for every cell. VN30/VN100 deep numbers change (now
  reproducible from the recorded config).
- Papers (all four): learned rows → per-seed **mean**, QLIKE column shows **mean ± per-seed std**; DM tables
  relabelled "on the 5-seed ensemble forecast … unadjusted"; abstract/intro/method/results/discussion/
  conclusion/captions rewritten to the new (stronger HAR-X-dominant) picture; F2/F3/F4 + batch=32
  reproducibility disclosures added. Corrected a propagated claim (VN100 h1: HAR-X, not LSTM+GAT, has the
  lowest MSE/RMSE/R²; LSTM+GAT only the lowest MAE).

## New picture (VN30/VN100, per-seed)
HAR-X has the lowest QLIKE at every horizon on both panels; no learned model is significantly lower on QLIKE
anywhere. Deep models have the lowest MAE at some short horizons on VN100. The LSTM+GAT has a significantly
lower QLIKE than the no-graph LSTM at VN30 h1, VN100 h1 and VN100 h5 (not VN30 h5 / h10 / h22) — an
ensemble-level effect of the same order as the learned models' per-seed QLIKE std (0.02–0.07). Cross-market
(extended/crossmarket): only on the less-liquid HNX do the deep/graph models reach a significantly lower QLIKE
than HAR-X; on VN30/VN100/HOSE/SP500 they cut point error (MAE/MSE) but not QLIKE.

## Verification
- Tests: `test_masked_rich.py` 11/11 (incl. new seed_metric_stats test); full runner/GARCH/submission suites
  pass in the pre-push gate.
- Re-run: 20/20 cells have `config.batch_size==32` + `metrics_per_seed`; each passed the HAR-X basis guard.
- Numeric: submitted 160, extended 352, crossmarket 440, with_sp500 196 VN table cells ALL MATCH result.json;
  DM cells verified. Compiles: 8 / 11 / 13 / 10 pages, 0 LaTeX errors.
- Parallelism note: a 2-worker GPU re-run was attempted per user request but measured ~2.4× slower per cell
  (compute contention on one GPU), so the run was completed sequentially (matches the known VRAM/compute
  bottleneck; parallel training is counterproductive here).

## Code review
Runner change is additive (records config + per-seed stats; ensemble/DM/GARCH paths unchanged) and TDD-tested;
GARCH cap covered separately. No leakage introduced. Papers verified cell-by-cell against result.json.

## Data-quality gate
N/A (no data/feature/pipeline change — model outputs on already-validated processed data).

## Commits
`9ac2c8a` (F1 fix + submitted paper), `b12467a` (3 supplementary papers + VN100 h1 fix). Deliverables folder
`deliverables_20260823` refreshed separately (untracked artifact).

## Follow-ups
- Crossmarket version is 13 pages (discussion track); the submitted paper is 8 pages, within the 12-page limit.
