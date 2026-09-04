# Manifest — files in review scope (exact paths)

Paths are repo-relative. Read in place. `archive/` is out of scope.

## 1. Single-source-of-truth config
| Path | Purpose |
|---|---|
| `submission/soict_lstm_gat/pipeline_config.py` | ALL tunable constants (windows, floors, seeds, walk-forward K/val/test_frac, VOLUME_ZSCORE_WINDOW=22, LOOKBACK). Everything imports from here. |

## 2. Data pipeline (ETL clean + enrich → causal columns)
| Path | Purpose |
|---|---|
| `scripts/data_pipeline/run_pipeline.py` | P1→P6 runbook (raw-quality → audit → clean → enrich → data-quality gate → freeze); `--market`, `--incremental`, `--dry-run` |
| `scripts/etl_audit/etl_cleaning.py` | Cleaning ops (widen_range/clip_oc/reconstruct_nonpositive/backadjust_splits/cut_to_listing/…), each returns (df, info) |
| `scripts/eda/volatility_estimators.py` | `estimators_from_ohlcv`: parkinson / garman_klass / rogers_satchell / windowed yang_zhang (test-vs-formula) |
| `baselines/2026-08-31_enriched_processed/` | Baseline that builds `data/processed_enriched/` (17 cols incl parkinson_variance + HAR + market_pk + volume_zscore_22 + dirty flags); tests + EDA |
| `data/processed_enriched/vn30/`, `.../vn100/` | The enriched panels the models read (VN30 ⊂ VN100). NOTE `market_pk` is market-specific (differs between dirs). |

## 3. VolGA walk-forward (headline results)
| Path | Purpose |
|---|---|
| `baselines/2026-08-31_walkforward_volga/code/run_volga_walkforward.py` | `run_fold`/`run_walkforward` (HAR/HAR-X/LSTM/VolGA), CLI `--market/--horizon/--lookback/--folds-target/--epochs/--batch` |
| `baselines/2026-08-31_walkforward_volga/code/wf_enriched_panel.py` | Enriched reader: `build_enriched_panel`, `frozen_universe`, `pack_fold` (per-fold TRAIN-only scalers + vol→PK graph) |
| `baselines/2026-08-31_walkforward_volga/code/tests/` | 6 test modules (leakage/no-lookahead, 3-model run, horizon plumbing, real-data smoke, overfit-evidence) |
| `baselines/2026-08-31_walkforward_volga/code_review/code_review_2026-08-31.md` | prior adversarial review |
| `baselines/2026-08-31_walkforward_volga/{requirements,design}/` | spec + design (SDD) |

### Reused read-only by VolGA (also in scope — the actual model + trainer)
| Path | Purpose |
|---|---|
| `baselines/2026-08-21_har_anchored_residual/code/masked_rich.py` | `MaskedRichNet` (parallel per-node LSTM + GAT over vol→PK graph), `_directed_vol2pk`, `MaskedRichData`. NOTE: the readers add both `submission/soict_lstm_gat` and this dir to `sys.path`; `masked_rich.py` exists ONLY here, so `import masked_rich` resolves to this file (there is no `submission/soict_lstm_gat/masked_rich.py`). |
| `baselines/2026-08-21_har_anchored_residual/code/run_masked_rich.py` | `train_masked_rich` (batched GPU trainer), `_pred_dict/_ens/_metrics/seed_metric_stats/_dm_all` |
| `baselines/2026-08-30_walkforward_harx_lstm/code/run_walkforward.py` | `_har_ols_preds` (HAR 3-feat + HAR-X 5-feat OLS, refit per fold), `training_config` |
| `baselines/2026-08-30_walkforward_harx_lstm/code/wf_folds.py` | `make_folds`, `assert_no_leakage` |

## 4. Pooled/transfer VN30 ablation
| Path | Purpose |
|---|---|
| `baselines/2026-09-04_pooled_transfer_vn30/code/pooled_panel.py` | `vn30_index`, `screened_universe`, `restrict_fold` (train-node mask + graph isolation), `score_mask` |
| `.../code/run_pooled_arm.py` | `run_arm` (one arm's walk-forward) |
| `.../code/run_pooled_ablation.py` | `_build` + `run_ablation` (both arms, paired DM, diff-in-diff, JSON) |
| `.../code/tests/` | 8 test modules incl. **isolation** + **alignment** |
| `.../code_review/code_review_2026-09-04.md` | adversarial review |
| `docs/superpowers/specs/2026-09-04-pooled-transfer-vn30-design.md` | design spec |
| `docs/superpowers/plans/2026-09-04-pooled-transfer-vn30.md` | implementation plan |

## 5. Results (JSON — the numbers to trust)
| Path | Purpose |
|---|---|
| `results/walkforward_volga/walkforward_volga_vn30_h{1,5,10,22}.json` | VN30 headline (metrics_pooled? no — `metrics`, `dm_date_clustered`, `per_fold`, fit evidence) |
| `results/walkforward_volga/walkforward_volga_vn100_h{1,5,10,22}.json` | VN100 headline |
| `results/pooled_transfer_vn30/pooled_vn30_h*.json` | ablation (arm0/arm1/paired_dm/diff_in_diff) — h1 done, rest landing |

## 6. Reports / dashboards / paper evidence
| Path | Purpose |
|---|---|
| `docs/reports/2026-08-31_volga_walkforward_vn100_dashboard.html`, `..._vn30_dashboard.html` | full-metric dashboards (MSE/RMSE/MAE/R2/QLIKE + per-seed + DM 3-basis + Section-1 walk-forward schematic) |
| `docs/reports/2026-08-31_volga_walkforward_vn100_multihorizon.md`, `..._vn30_vs_vn100.md` | multi-horizon + cross-market reports |
| `docs/reports/2026-09-04_pooled_transfer_vn30_report.md` | ablation report (auto-generated per horizon) |
| `docs/reports/2026-09-04_ablation_data_organization.html` | data-organization illustration (Arm0/Arm1 vs standalone) |
| `docs/reports/2026-08-23_1600_volatility_estimator_research.md` | estimator choice (why Parkinson) — test-vs-formula |
| `docs/reports/appendix/2026-09-03_overnight_tail_appendix.md` + `.csv` | data-quality: split-artifact / overnight tail (not split-adjusted) |
| `docs/papers/README.md` | HAR / HAR-X literature refs (Corsi 2009, Clements 2024, GNAR-HARX 2025) |

## 7. Quality gate (how CI enforces the above)
| Path | Purpose |
|---|---|
| `scripts/git_hooks/pre-push` | 6-step gate: TDD, changed-scope tests + diff-cover C0=100/C1≥95, ruff-F, data-quality+lessons, delivered-baseline GPU tests, config-hardcode scan |
| `scripts/quality_gate/` | Pandera schema + Evidently drift + overfit-evidence checks + config-hardcode scanner |
| `CLAUDE.md` | project constitution (quality rules, ablation=leave-one-out, named-estimator=published-formula, no-silent-degradation) |
