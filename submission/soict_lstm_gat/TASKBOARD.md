# SOICT HAR-LSTM-GAT — Task Board (autonomous build 2026-08-21)

Quality-gate legend: **Tests** = module pytest green · **Gate** = pushed through pre-push quality gate
(pytest+data-quality+baseline) · **Review** = /code-review done.

| # | Task | Owner | Status | Tests | Gate | Review |
|---|------|-------|--------|-------|------|--------|
| 0 | Scaffold (config, conftest, deps, taskboard) | main | done | n/a | pending | — |
| 1 | metrics.py (5 metrics + QLIKE floor + DM) | agent | done | 14 ✅ | pending | — |
| 2 | data_utils.py (HAR feats, windows, 80/10/10 split, scaler) | agent | done | 8 ✅ | pending | — |
| 3 | edges.py (graphical-lasso train-only adjacency) | agent | done | 3 ✅ | pending | — |
| 4 | model.py (HAR-LSTM-GAT + use_graph toggle) | main | done | 3 ✅ | pending | — |
| 5 | baselines.py (HAR OLS + GARCH, arch 8.0.0) | agent | done | 4 ✅ | pending | — |
| 6 | snapshots.py (common-date fixed-N, global-date 80/10/10) | main | done | via smoke ✅ | pending | — |
| 7 | train.py (pooled MSE loop, early-stop, curves, GPU) | main | done | via smoke ✅ | pending | — |
| 8 | evaluate.py + run_all.py (metrics + DM + orchestrate) | main | done | e2e smoke ✅ | pending | — |
| 9 | Ship data + reproduce.sh + EXTRACTION_LOG + README | main | pending | — | — | — |
| 10 | Run full suite on GPU (main + ablation + 3 variation × h1/h5 × 5 seeds) | main | pending | — | — | — |
| 11 | Paper markdown draft + .svg architecture diagram | main | pending | — | — | — |
| 12 | LaTeX (SOICT) + /code-review + summary report | main | pending | — | — | — |

**Submission tests:** 32 passed (metrics 14 + data_utils 8 + edges 3 + baselines 4 + model 3).
**E2E smoke (VN30, 2ep/1seed):** 33 nodes, 1050 train / 132 test snapshots, 184 edges, n_test=4356 —
full pipeline (HAR-LSTM-GAT + LSTM w/o GAT + HAR + GARCH + DM) runs on GPU.

**Design note / deviation:** the graph model uses COMMON-DATE fixed-N snapshots with a GLOBAL-date
80/10/10 split (a cross-sectional graph needs per-date snapshots; this also gives a common test
window). This deviates from the spec's per-stock 80/10/10 — flagged for user review. GPU:
`.venv_gpu_encode`. Parallel: leaf modules built by 4 concurrent agents; experiment suite runs
configs as concurrent processes.
