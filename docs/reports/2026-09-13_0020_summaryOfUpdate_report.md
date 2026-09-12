# Complex-network topology — HOSE feature-importance debug + eig removal

Date: 2026-09-13. Baseline: `baselines/2026-09-12_complex_network`. Market: HOSE (VNINDEX for the index
target). SP500 runs separately (background / Colab) and is reported when it completes.

## 1. What changed

| File | Change |
|---|---|
| `code/feature_screen.py` (new) | Causal per-fold feature screen: Pearson, Spearman, mutual information, VIF for the 15 features (3 HAR + 6 own-history + 6 topology) vs `y` = future Parkinson variance. Statistics estimated separately on each walk-forward training fold (`date ∈ [TRAIN_START, fold_start − embargo)`); fold gate mirrors `run_gbm` (`len(te)==0 or len(tr)<min_rows`; min_rows 30000 sp500 / 3000 else). MI on a fixed-seed ≤40000-row subsample (flagged per horizon). |
| `code/build_feature_debug_html.py` (new) | Standalone HTML with embedded charts: per-horizon QLIKE (own vs +topo), train-vs-test QLIKE, the feature screen (MI / Pearson·Spearman / VIF per feature, grouped), keep/drop table, and the Experiment-A topology→index-volatility OOS R². |
| `code/config.py` | Dropped `eig` from `TOPO` (7→6). Added `SCREEN_*` constants. `RF_KW` gains `n_jobs=-1` (no result change). |
| `code/topology.py` | `global_feats` returns 6 metrics (eigenvector centrality removed). |
| `code/{run_index,run_gbm,diag_gbm}.py` | Docstrings updated for the 6-metric set (behaviour unchanged — all read `config.TOPO`). |
| `notebooks/complex_network_sp500_colab.ipynb` (new) | Colab notebook to run the SP500 suite faster (git-clone code + two Drive data bundles + push JSONs). `.gitignore` allowlists it; `colab_bundle_*.zip` ignored. |

`eig` was removed on the screen evidence: at every horizon its mutual information with the target was the
lowest of all 15 features (≈0.0006–0.007) and its VIF was `inf` (perfectly collinear with the other metrics).

## 2. HOSE results (6-feature set)

### Experiment B — do the topology metrics help the per-stock gamma-GBM? (pooled QLIKE, lower better)
| Horizon | GBM (own) | GBM+topo | gain % | DM p |
|---|---|---|---|---|
| h1  | 1.5679 | 1.5785 | −0.67 | 4.5e-09 (topo significantly **worse**) |
| h5  | 1.6478 | 1.6517 | −0.24 | 0.077 (worse, marginal) |
| h10 | 1.6842 | 1.6814 | +0.17 | 0.278 (tie) |
| h22 | 1.7263 | 1.7279 | −0.09 | 0.845 (tie) |

Adding topology lowers the **train** QLIKE but not the **test** QLIKE — the signature of fitting in-sample
noise with uninformative features. Prediction correlation own-vs-(own+topo) = 0.88–0.996.

### Causal feature screen — mutual information, graph vs own-history
| Horizon | mean MI (graph, 6) | mean MI (own+HAR, 9) | ratio |
|---|---|---|---|
| h1  | 0.0372 | 0.1955 | 5.3× |
| h5  | 0.0345 | 0.1486 | 4.3× |
| h10 | 0.0353 | 0.1338 | 3.8× |
| h22 | 0.0362 | 0.1172 | 3.2× |

Per-feature (h22): the strongest predictors are `har_daily` / `mr_dev5` / `mr_slope*` (MI ≈ 0.13, Spearman up
to 0.33 for HAR). Every graph feature sits at the bottom (MI 0.007–0.045, Pearson ≈ 0, Spearman ≤ 0.11).

### Experiment A — topology metrics → future VNINDEX volatility (OOS R², 227 monthly windows)
| Model | idx_vol (headline) | idx_ret | idx_lnvol |
|---|---|---|---|
| LinearRegression | −0.045 | −0.333 | +0.260 |
| RandomForest | +0.153 | −0.136 | +0.282 |

On the headline target (std of log-returns) LR fails OOS (R²<0); RF reaches only +0.15. Signal appears only on
the log-volume target and is largely in-sample (train R² ≫ test R²).

## 3. Answers to the two analysis questions

**Q: graph MI at h1/h5 looks smaller than at h10/h22 — does the graph help more at long horizons?**
No. Graph MI is essentially flat across horizons (0.0345–0.0372; h1 is actually the highest). What changes is
the *ratio* to own-history (5.3×→3.2×), and it shrinks because **own-history MI decays** with horizon
(0.196→0.117 — forecasting 22-day-ahead variance from recent history is harder), not because graph MI grows.
Pearson ≈ 0 and Spearman ≤ 0.11 for every graph feature at every horizon — no linear and only trivial monotone
association. The GBM confirms it: topology is significantly worse at h1, marginally worse at h5, and a tie at
h10/h22. **Conclusion: the topology metrics carry no usable information about future per-stock volatility at
any horizon.** The narrower long-horizon ratio is degradation of the good predictor, not emergence of a graph
signal. (Absolute MI ≈ 0.035 is near the kNN-MI estimator's positive-bias floor; a shuffled-`y` permutation
null would make the floor explicit and is a recommended follow-up.)

**Q: any further redundant/noisy graph features to drop?**
Yes. After `eig`, the screen flags (h22 VIF): `dens` 405 and `avg_deg` 368 are near-perfect duplicates
(density ∝ average degree / (n−1)) — keep at most one; `clus` VIF 20.9 is redundant; `diam` has the lowest MI
of all remaining features (0.007) — noise. `betw` is the least-bad graph feature (highest graph MI 0.042,
Spearman 0.109, VIF 5.8). A minimal non-redundant graph set would be ≈ {one of (dens, avg_deg), avg_w, betw}.
However, since the full 6-feature block already adds no OOS value (GBM ties/worse at all horizons), the
recommended action is to **drop the entire topology block from the per-stock forecaster** and keep it only as a
documented negative result / for the faithful paper replication (Experiment A). Note the own-history collinearity
(`rq`, `har_weekly` VIF ≈ 127–130) is harmless for the tree model and those features carry real signal (high
MI), so they are kept.

## 4. Artifacts
- `results/gamma_gbm/complex_network_hose.json` (Exp B), `..._index_hose.json` (Exp A), `..._hose_screen.json`
  (feature screen), `..._hose_diag.json` (permutation importance).
- `docs/reports/2026-09-13_complex_network_hose_feature_debug.html` (charts: QLIKE, screen, keep/drop, index R²).
- `docs/reports/2026-09-12_complex_network_hose_why_topo_hurts.html` (Experiment-B diagnostic).

## 5. Definition of Done
- Tests: 49 pass (`baselines/2026-09-12_complex_network/test`). New modules TDD-style with synthetic
  known-signal/known-noise/collinear fixtures.
- Coverage: C0 line = 100% and branch = 100% on all 10 code modules (pytest-cov, `--cov-branch`).
- Code review: 2 adversarial rounds. Round 1 found 1 CRITICAL (notebook f-string newline) + 1 MAJOR (fold-gate
  mismatch with run_gbm) — both fixed. Round 2: no CRITICAL/MAJOR; minor doc-drift fixed, fold-parity guard added.
- Post-generate gate (ruff-F + config-hardcode): pass.
- Data-quality (Pandera/Evidently): N/A — no `data/` change (code + result JSONs only).
- Performance: screen batches correlations/MI/VIF on numpy arrays per fold; RF uses all cores (`n_jobs=-1`);
  HistGBM uses OpenMP. No per-item batch=1 loop.
