# Beat-HAR Solution Sweep — Results (C1–C6, 20-epoch, 3-seed)

Date: 2026-08-10. Branch: `feature/beat-har-sweep`. Run TS: `2026-08-10_0130`.
Plan: `docs/reports/2026-08-10_0033_beat_har_solution_plan.md`. Code:
`baselines/2026-08-10_beat_har_sweep/`.

## 1. Verdict

No configuration Diebold–Mariano-significantly beats the P0 pooled-HAR anchor (test QLIKE 0.5676) on
any metric. C2 (HAR + additive graph residual) statistically **ties** P0 on both QLIKE (paired-t
p=0.562) and RMSE (per-seed DM p≥0.09) — the downside-protected design held. C1 (QLIKE-loss GAT+news)
ties P0 on RMSE but is significantly worse on QLIKE. C3 (directed spillover edges), C5 (spillover +
omit-self + k-sweep) and C6 (learned adjacency) are worse than P0 on QLIKE. This is a documented null:
the graph / QLIKE-loss / spillover / learned-adjacency / HAR-residual levers do not beat a
well-specified HAR on VN30 daily range-based (Parkinson) variance at 33-asset scale, matching the
plan's stated honest expectation and the located literature (GNNHAR/DJIA-30, graph component null).

Configs run: 5 (C1, C2, C3, C5[k=8,16], C6), 3 seeds each (42/123/2026), 20 graph-head epochs.
C4 (HAR-RV-X features) deferred (§5); C7 (news-as-edge) infeasible (§5).

## 2. Fair basis (identical to the consistent ladder)

Masked kNN-8 mutual-correlation graph, leakage-safe graph-bound train window, train-only per-ticker
scalers, temporal 70/15/15, Parkinson-**variance** target `shift(-5)`, positivity floor, present-node
masking. The held-out observations are identical to `ladder_consistent_h5_2026-08-09_154402.json`
(n_val = 14418, n_test = 14464, 33 tickers) — asserted by `build_basis`'s one-basis invariant. Each
config keeps the SAME MSE-trained P3 backbone as the ladder and retrains only the graph-stage
message-passing + head (which sets the prediction level) under its loss / adjacency / head variation
(design note: isolates the graph-stage lever on an identical basis; QLIKE applied at the graph-head
stage — see `code_review/code_review_2026-08-10.md` F4). Leakage invariants verified: spillover VAR and
learned-A embeddings estimated on the train window only and frozen; no val/test date enters the graph
structure.

## 3. Results — all configs × 6 metrics (mean over 3 seeds)

HAR reference bar (test), from the plan / cited JSONs:
P0 QLIKE **0.5676**, RMSE 0.0022893, R² 0.76679, MAE 0.0006027, DirAcc 48.53; classical per-ticker HAR
QLIKE 0.5793; HARQ RMSE **0.0022891**, R² **0.76682**.

### TEST (mean over seeds 42/123/2026)

| Config | QLIKE | RMSE | R² | MAE | MSE | DirAcc |
|---|---|---|---|---|---|---|
| **P0 HAR anchor** (bar) | **0.5676** | 0.0022893 | 0.76679 | 0.0006027 | — | 48.53 |
| HARQ (RMSE/R² wall) | 0.5737 | **0.0022891** | **0.76682** | 0.0006289 | — | 48.38 |
| C1 QLIKE-loss GAT+news | 0.57296 | 0.0022922 | 0.76619 | 0.0006053 | 5.254e-06 | 48.36 |
| C2 HAR+graph-residual | **0.56625** | 0.0022896 | 0.76671 | 0.0006032 | 5.243e-06 | 48.28 |
| C3 spillover edges | 0.59076 | 0.0023283 | 0.75872 | 0.0006215 | 5.422e-06 | 47.90 |
| C5 spillover+omit-self (best k=16) | 0.57476 | 0.0023057 | 0.76343 | 0.0006091 | 5.316e-06 | 48.03 |
| C6 learned adjacency | 0.59031 | 0.0023335 | 0.75763 | 0.0006240 | 5.447e-06 | 48.24 |

### VALIDATION (mean over seeds)

| Config | QLIKE | RMSE | R² | MAE | MSE | DirAcc |
|---|---|---|---|---|---|---|
| C1 | 0.51076 | 0.0014694 | 0.74050 | 0.0004741 | 2.159e-06 | 48.53 |
| C2 | 0.50612 | 0.0014618 | 0.74318 | 0.0004707 | 2.137e-06 | 48.64 |
| C3 | 0.51507 | 0.0014699 | 0.74034 | 0.0004775 | 2.161e-06 | 48.96 |
| C5 (k=16) | 0.50503 | 0.0014620 | 0.74313 | — | — | 49.11 |
| C6 | 0.51297 | 0.0014803 | 0.73663 | 0.0004785 | 2.191e-06 | 48.86 |

Per-seed test QLIKE: C1 [0.5715, 0.5746, 0.5728]; C2 [0.5668, 0.5694, 0.5625]; C3 [0.5720, 0.5914,
0.6089]; C6 [0.5855, 0.5942, 0.5913]; C5 k=8 [0.5720, 0.5813, 0.5776], k=16 [0.5728, 0.5767, 0.5748].

## 4. Per-config success check + significance (vs P0, DM h=5 + across-seed paired-t)

"Beat" = test QLIKE < P0 0.5676, all-3-seeds consistent sign, per-seed DM p<0.05 AND paired-t p<0.05.

| Config | QLIKE Δ vs P0 (seeds) | all-neg? | paired-t p | per-seed DM p (QLIKE) | RMSE DM p | Verdict |
|---|---|---|---|---|---|---|
| C1 | +0.0039, +0.0070, +0.0051 | no | 0.027 | 0.016 / 0.0 / 0.0005 | 0.98 / 0.81 / 0.74 | QLIKE **worse** (sig); RMSE ties |
| C2 | −0.0008, +0.0018, −0.0051 | no | **0.562** | 0.57 / 0.14 / 0.0 | 0.43 / 0.20 / 0.09 | **Ties P0** (QLIKE & RMSE) |
| C3 | +0.0044, +0.0238, +0.0413 | no | 0.162 | 0.008 / 0.0 / 0.0 | 0.42 / 0.21 / 0.018 | QLIKE **worse**; RMSE mostly ties |
| C5 (k=16) | +0.0052, +0.0091, +0.0072 | no | 0.024 | 0.005 / 0.0 / 0.0 | 0.34 / 0.42 / 0.30 | QLIKE **worse** (sig); RMSE ties |
| C6 | +0.0178, +0.0266, +0.0236 | no | 0.013 | all ≈0 | 0.53 / 0.0004 / 0.036 | QLIKE **worse** (sig); RMSE mixed |

- **C2** is the only config whose seed-mean QLIKE (0.5662) is below P0, but the sign is not consistent
  (seed 123 above) and the paired-t is far from significant (p=0.562): C2 is statistically
  indistinguishable from P0 on QLIKE and on RMSE — a clean tie, not a win.
- **C1** ties P0 on RMSE (DM p≥0.74) but is significantly worse on QLIKE.
- **C3 / C5 / C6** are significantly worse than P0 on QLIKE; none beats P0/HARQ on RMSE or R².
- Against the weaker classical per-ticker HAR (QLIKE 0.5793), C1 (0.573) and C2 (0.566) are lower, but
  the effective bar is the stronger P0 anchor (0.5676), which none clears DM-significantly.

No full-win (beat P0 AND HARQ on QLIKE+RMSE+R²) and no partial-win (DM-significant QLIKE beat) was
achieved. The documented null is confirmed.

## 5. C4 and C7 status

- **C4 (HAR-RV-X range/overnight node features).** Deferred. The estimator math (Garman–Klass,
  Rogers–Satchell, overnight variance, in σ² units) is implemented and unit-tested (`rvx_features.py`,
  7 tests) and the raw OHLC is available (`data/raw/prices/*_ohlcv.csv`), but full wiring needs a
  6-feature backbone, which requires extending the pilot per-ticker preprocessor
  (`TickerPreprocessor.fit` hard-asserts 1 feature == target) — a change to shared pilot code beyond
  this run's isolation scope and its other in-flight consumers. Not run; not claimed.
- **C7 (news-as-edge co-mention).** Infeasible on the current data. The precomputed news panel is
  per-(ticker, date) PhoBERT vectors with no article-level multi-ticker structure, so co-mention edges
  cannot be built from it (verified: panel columns are per-ticker embeddings only). Dropped per the
  plan's feasibility gate.

## 6. Relation to the plan's expectation and the literature

The plan's honest expectation was a QLIKE-only DM-significant partial win at best, or a documented null,
with RMSE/R² tying HARQ and no precedent for a GNN beating a well-specified HAR on daily range-variance
at ~30-asset scale. The observed outcome is the null branch: the QLIKE-loss lever (C1) improves over the
MSE-trained G1 (0.5793 → 0.573) and beats classical HAR but not the pooled-HAR anchor; the graph edge
constructions (correlation kNN, directed spillover, learned) do not add level-metric value and the
directed-spillover / learned variants degrade it (noisy VAR/graph-structure-learning on 33 daily
series, as flagged). C2's structural HAR anchor delivers the best (tie) outcome, consistent with the
"additive HAR-residual floors at HAR" hypothesis.

## 7. Reproducibility / evidence

- Seeds actually run: 42, 123, 2026 (all configs). Epochs: 20 graph-head, on the frozen 5-epoch (warm
  4 + safe 1) MSE P3 backbone, dropout 0.2, Adam wd 1e-5, grad clip 1.0.
- Per-config predictions + metrics: `results/beat_har_{C1,C2,C3,C5,C6}_2026-08-10_0130/seed*/`
  (`results.json`, `predictions_test.json`, `predictions.json`, `learning_curve.png`).
- Significance: `results/beat_har_sweep_2026-08-10_0130/analysis.json` (P0 recomputed on the basis,
  per-seed DM h=5 + across-seed paired-t). Summary: `.../report_summary.json`.
- Tests: 28 unit + 6 smoke green (`baselines/2026-08-10_beat_har_sweep/test/`); ruff clean on the
  baseline. Adversarial code review: `baselines/2026-08-10_beat_har_sweep/code_review/`.
- Data-quality gate: N/A for the sweep code itself (no data/manifest change; reuses the frozen fair
  basis). The pre-push gate runs Pandera schema (34/34 valid) + Evidently drift.
