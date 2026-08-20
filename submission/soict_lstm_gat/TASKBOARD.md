# SOICT HAR-LSTM-GAT — Task Board (autonomous build 2026-08-21)

Quality-gate legend: **Tests** = module pytest green · **Gate** = pushed through pre-push quality gate ·
**Review** = /code-review done.

| # | Task | Owner | Status | Tests | Gate | Review |
|---|------|-------|--------|-------|------|--------|
| 0 | Scaffold (config, conftest, deps, taskboard) | main | done | n/a | ✅ | pending |
| 1 | metrics.py (5 metrics + QLIKE floor + DM) | agent | done | 14 ✅ | ✅ | pending |
| 2 | data_utils.py (HAR feats, windows, 80/10/10, scaler) | agent | done | 8 ✅ | ✅ | pending |
| 3 | edges.py (graphical-lasso train-only adjacency) | agent | done | 3 ✅ | ✅ | pending |
| 4 | model.py (HAR-LSTM-GAT + use_graph toggle) | main | done | 3 ✅ | ✅ | pending |
| 5 | baselines.py (HAR OLS + GARCH, arch 8.0.0) | agent | done | 4 ✅ | ✅ | pending |
| 6 | snapshots.py (common-date fixed-N + universe select) | main | done | 2 ✅ | ✅ | pending |
| 7 | train.py (pooled MSE loop, early-stop, curves, GPU) | main | done | smoke ✅ | ✅ | pending |
| 8 | evaluate.py + run_all.py (metrics + DM + orchestrate) | main | done | e2e ✅ | ✅ | pending |
| 9 | Ship data + reproduce.sh + EXTRACTION_LOG + README | main | done | n/a | ✅ | pending |
| 10 | Run full suite on GPU (6 VN configs; sp500 OOM) | main | done | n/a | ✅ | n/a |
| 11 | Paper markdown draft + .svg architecture diagram | main | done | n/a | ✅ | pending |
| 12 | LaTeX (SOICT) + /code-review + summary report | main | partial | n/a | ✅ | pending |

**Submission tests:** 34 passed (metrics 14 + data_utils 8 + edges 3 + baselines 4 + model 3 + snapshots 2).

**Experiment suite (6/8 configs, 20 epochs, 5 seeds):** VN30 lb10/lb22 × h1/h5 + VN100 lb10 × h1/h5.
S&P500 EXCLUDED — GAT attention is O(N²), OOM at 500 nodes on 8 GB VRAM (even batch 16).

**HONEST RESULT (negative):** HAR is the best model at every config; HAR-LSTM-GAT does NOT beat HAR
(significant by DM except VN100-h5 tie); the GAT graph consistently HURTS (leave-one-out favours LSTM
w/o GAT); all learned models beat GARCH. Report: `docs/reports/2026-08-21_0141_soict_results_report.md`.
Paper draft: `docs/paper/soict_harlstmgat_draft.md`. Diagram: `docs/paper/diagrams/soict_harlstmgat.svg`.

**Pending for user review:** (a) accept honest negative result / reframe paper, or (b) re-run deep
models under per-stock per-observation split (fairer, beat HAR on S&P500 elsewhere). SOICT LaTeX +
/code-review still to do. **Design deviation:** global-date snapshot split (documented in report).
