# Backlog — Rolling-origin recalibration (P4 robustness)

**Status:** deferred by user 2026-08-16 ("đưa vào backlog, review làm sau"). NOT delivered to reviewers
until executed and approved.

## What
Full rolling-origin evaluation (paper protocol, Zhang et al. 2308.01419): instead of the current
single static per-ticker 70/15/15 split, slide a window forward through time and RETRAIN
(recalibrate) the model at each step, testing on the next unseen block; aggregate the block test
predictions into a multi-period OOS series. Paper uses train 36m / val 12m / test 1m, roll monthly,
~10y OOS.

## Why
- The static split tests only one recent tail (2021–2026). The temporal-stability check
  (`subperiod_report.py`) already showed the full model's edge is time-varying (concentrated in
  middle sub-periods; recent block favours HAR at h5/h10/h22) — rolling would show whether periodic
  retraining keeps the model competitive.
- Strongest robustness evidence for reviewers.

## Cost / why deferred
Trains dozens of times (one per window) × 4 horizons × 5 ablation rungs = many hours on the single
RTX 4060 laptop GPU. Needs a scoping decision (expanding vs sliding window; number of folds; which
horizons; how many seeds) before running.

## How (starting points)
- Split logic to generalize: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py`
  `chronological_split` (currently fixed 0.7/0.15/0.15) — parameterize a movable fold boundary.
- Reuse the trained pipeline: `baselines/2026-08-15_volatility/code/run_ablation.py` per fold.
- DM per window: reuse `dm_report` / `subperiod_report`.
- Lighter check already done: `subperiod_report.py` (temporal stability on the fixed test window).

## Related
- Results report: `docs/reports/2026-08-16_1600_gnnhar_p1p2p3_results_report.md` (Follow-up 3b).
- Paper addendum: `docs/paper/volatility_paper_addendum_gnnhar_p1p2p3.md` (Section G).
