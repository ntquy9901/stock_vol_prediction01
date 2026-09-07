# QLIKE-loss / HAR-X-anchor on VN30 — results (GO: cross-market confirmation)

Experiment `baselines/2026-09-07_qlike_anchor` on VN30 (33 nodes, 7 folds, 5 seeds, lookback 10),
completing the QLIKE-loss study begun on VN100. Configs: loss=qlike × anchor∈{none, harx}, all
horizons. Ticker-date pooled OOS QLIKE; DM date-clustered vs HAR-X. All 8 result JSONs carry
train/val/test metrics + fit_diagnostics (fit=ok for LSTM/VolGA at every horizon) + learning curves.

## Outcome: GO — QLIKE-loss makes the deep models beat HAR-X on QLIKE at h1, same as VN100
Training LSTM/VolGA on a **QLIKE loss** (anchor=none) makes both **significantly beat HAR-X on QLIKE
at the daily horizon** on VN30, mirroring VN100. This reverses the MSE-loss ranking, where the VN30
VolGA QLIKE advantage over HAR-X was not significant (edgehm DM p=0.280).

## QLIKE by horizon (VN30, anchor=none)
| h | HAR-X | LSTM (MSE) | LSTM (QLIKE) | VolGA (MSE) | VolGA (QLIKE) | DM QLIKE-loss vs HAR-X (LSTM / VolGA) |
|---|---|---|---|---|---|---|
| 1 | 0.4801 | 0.4787 | **0.4634** | 0.4740 | **0.4639** | **p=0.004** / **p=0.006** |
| 5 | 0.5602 | 0.5738 | 0.5585 | 0.5672 | 0.5576 | p=0.865 / 0.780 (n.s.) |
| 10 | 0.6091 | 0.6048 | 0.6037 | 0.6061 | 0.6045 | p=0.572 / 0.620 (n.s.) |
| 22 | 0.6782 | 0.6832 | 0.6777 | 0.6882 | 0.6782 | p=0.980 / 0.999 (n.s.) |

- **h1:** QLIKE-loss LSTM (0.4634) and VolGA (0.4639) both significantly beat HAR-X (0.4801),
  DM p=0.004 / 0.006.
- **h5/h10/h22:** QLIKE-loss deep models match HAR-X (at or below, not significant) — versus MSE-loss
  where they were worse at h5/h22. QLIKE-loss removes the disadvantage across horizons.
- **anchor=harx** (secondary): also beats HAR-X at h1 (LSTM p=0.029, VolGA p=0.006), slightly weaker
  than anchor=none. The loss is the lever; the anchor is secondary (as on VN100).

## MSE / RMSE / MAE cost — negligible
Switching MSE→QLIKE loss changes MSE/RMSE by under 1.1% and raises MAE by at most ~2.5% at every
horizon on VN30 (same as VN100). The QLIKE gain is nearly free on the squared/absolute-error metrics.

## Cross-market summary (both panels, QLIKE-loss anchor=none, h1)
| panel | HAR-X | LSTM (QLIKE) | VolGA (QLIKE) | DM LSTM / VolGA vs HAR-X |
|---|---|---|---|---|
| VN100 | 0.5000 | 0.4840 | 0.4833 | p=0.003 / 0.004 |
| VN30 | 0.4801 | 0.4634 | 0.4639 | p=0.004 / 0.006 |

The QLIKE-loss finding is consistent across both VN panels: training directly on the evaluation loss
lets the deep models overtake HAR-X on QLIKE at the daily horizon, at negligible squared-error cost.

## Added to the paper
`docs/paper/soict_harlstmgat_2026-09-07_final.tex` §Ablation → "Training objective: QLIKE-loss
training" (Table~\ref{tab:qlikeloss}, both panels) + abstract/discussion sentences. PDF recompiled
(12 pages) and re-uploaded to Drive `gdrive:luanvan_papers/`.
