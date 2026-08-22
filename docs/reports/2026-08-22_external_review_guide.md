# External review guide — source code, paper, and documents

Purpose: a map for an independent reviewer (human or AI) to verify correctness of the code, paper, and
claims BEFORE any full re-run. Read top to bottom. Priorities are marked **[P1]** (critical path — the code
that produces the paper's numbers), **[P2]** (supporting/statistics), **[P3]** (peripheral/exploratory,
lower priority). `archive/` is out of scope.

Repo is public. Data under `data/` is gitignored (large); code, paper, results JSON, and reports are tracked.

---

## 0. What the paper claims (context)

Multi-horizon daily volatility forecasting on Vietnamese equities (VN30, VN100). Target = Parkinson
**variance** at day t+h (the column `parkinson_volatility` is a variance, not a standard deviation). Three
feature-based models share five node features [daily Parkinson variance, 5-day mean, 22-day mean, market
Parkinson factor, 20-day volume z-score]: a linear **HAR-X** (5-feature OLS = the 3-lag HAR cascade + market factor + volume z-score; the paper's baseline), an **LSTM**, and an **LSTM+GAT** (directed volume→Parkinson Top-5, edge-weighted, two-hop graph).
A **GARCH(1,1)** classical benchmark is also reported. Evaluation: five metrics (MSE, RMSE, MAE, QLIKE, R²)
with equal weight; significance via the **date-clustered Diebold–Mariano** test. Horizons h∈{1,5} primary,
{10,22} extended.

Headline honest findings (verify these against the code/results):
- HAR-X has the lowest QLIKE at every horizon on both panels; **no learned model has a significantly lower
  QLIKE than HAR-X** under date-clustered DM.
- At h1/h5 the deep models have the lowest MAE on VN100; the LSTM+GAT has a significantly lower QLIKE than
  the no-graph LSTM at h1/h5 (graph helps the deep model but does not beat HAR-X).
- At h10/h22 HAR-X leads and the graph effect is not significant.
- GARCH is worse than HAR-X on all five metrics at every horizon.

---

## 1. [P1] Critical-path code — produces the paper's Tables 1–2

The submitted paper's masked-panel numbers come from the **masked-rich runner**, which imports shared
components from the **submission package**. Review these first.

### 1a. Masked-rich runner — `baselines/2026-08-21_har_anchored_residual/code/`
| File | LOC | Purpose | What to check |
|---|---|---|---|
| `masked_rich.py` | 230 | Builds the masked union-of-dates panel: 5 node features, node/target masks, per-node train-only scalers, min-train-rows node drop, directed volume→Parkinson edge + symmetric-corr edge, purge=h | **Leakage:** every scaler/edge estimated on TRAIN rows only (`last_tr_row`, `tok_tr`); `market_pk` = causal cross-sectional median; `volume_zscore` = trailing window (causal). Mask semantics: `node_mask`=valid input window, `target_mask`=valid window AND valid target. |
| `run_masked_rich.py` | ~320 | **Defines the paper's architecture** `MaskedRichNet` + `WeightedGATLayer` (5 node features, 2-hop weight-aware GAT) — THIS is the model the paper's Tables use (NOT `submission/model.py`). Fits **HAR** = classic 3-feature OLS (`D.har_tr`) and **HAR-X** = 5-feature OLS (the 3 HAR features + market factor + volume z-score). **The paper's baseline is HAR-X** (labelled "HAR-X"); the 3-feature HAR is computed but not reported in the paper. Trains LSTM / LSTM+wGAT, evaluates 5 metrics + date-clustered DM, writes `results/masked_rich_floor1e2/<ds>_h<h>/result.json`. | **Output param:** `--output-param {zscore_floor(default)|ratio_exp}`; default floors predictions at `1e-2*t_mean` (relative floor). ratio_exp is a robustness variant (positive-by-construction), see §4. **Loss:** masked MSE on standardized (or ratio) target. **DM:** `_dm_all` aggregates per-obs QLIKE/SE/AE to date level. |
| `masked_snapshots.py` | 138 | Earlier 3-feature masked panel (superseded by masked_rich for the paper) | Lower priority; kept for lineage. |
| `stats.py` | — | `date_clustered_dm(loss_a, loss_b, dates, h)` — aggregates per-obs loss to one value per date, then HLN Diebold–Mariano | **Panel-correct DM:** verify date aggregation (cross-sectional mean per date) and HLN correction; a naive per-obs DM overstates significance ~√N. |

### 1b. Shared components — `submission/soict_lstm_gat/`
| File | LOC | Purpose | What to check |
|---|---|---|---|
| `model.py` | 77 | The **ORIGINAL SOICT model** `HARLSTMGAT` (3 input features, per-stock-pooled experiment). **NOT the paper's model** — the paper (masked panel) uses `MaskedRichNet` defined in `run_masked_rich.py` (5 features, weighted 2-hop GAT). Kept for lineage; do not review as the paper's architecture. | If auditing the original SOICT experiment only. The paper's architecture is in §1a `run_masked_rich.py`. |
| `baselines.py` | 135 | `har_fit`/`har_predict` (OLS + floor); `garch_forecast` (GARCH(1,1) via `arch`, pseudo-returns from variance, fallback to train-variance mean) | **HAR floor**, **GARCH** pseudo-return construction + rescale by 1/1e4, fallback path. |
| `metrics.py` | 134 | MSE/RMSE/MAE/R²/QLIKE + Diebold–Mariano (HLN + HAC lag h−1) | **QLIKE:** shared positivity floor, `r=y/p`, `r−log r−1`. **R²:** `1−SSres/SStot` on the same pooled y,p (SStot uses test mean). **DM:** HLN small-sample correction. |
| `data_utils.py` | 187 | `har_features` (daily/weekly/monthly rolling), pooled data building | Rolling windows produce NaN at warmup/gaps (correctly masked). |
| `edges.py` | — | Directed vol→PK adjacency construction (train-only) | Train-only correlation; Top-K per target; self-loop. |

### 1c. GARCH add-on — `scripts/garch_masked/`
| File | Purpose | What to check |
|---|---|---|
| `compute_garch_masked.py` | Computes GARCH(1,1) on the SAME deterministic masked panel and writes `metrics['GARCH']` into each result.json; asserts recomputed HAR-X QLIKE matches the stored value (basis guard) | Per-node GARCH fit on TRAIN-ONLY series, forecasting the test window (skips the validation block); same per-node floor + qlike_floor; the basis-guard assert. |
| `test_garch_masked.py` | Tests for the above | — |

---

## 2. [P2] Statistics / robustness study (recent, in review)

The deep model's QLIKE was seed-sensitive on S&P 500; a diagnosis + ablation studied the output
parameterization. See report `docs/reports/2026-08-22_output_parameterization_robustness.md`.

| File | Purpose | What to check |
|---|---|---|
| `scripts/garch_masked/ablation_vn_5seed.py` | A/B/C/D output-parameterization ablation on VN30/VN100, 5 seeds, full-precision metrics + per-seed mean±std + date-clustered DM + **bias-matched** link comparison | The 4 configs (z-score/ratio × linear/exp/softplus × floor/no-floor); bias-match constants (exp bias 0, softplus bias log(e−1)); `_flat` date alignment for DM. |
| `scripts/garch_masked/ablation_output_param.py` | Same ablation on S&P 500 h5 (single-cell diagnosis) | — |
| `scripts/garch_masked/exp_logvar_test.py`, `exp_softplus_test.py` | Log-variance+exp (rejected: QLIKE ~126) and softplus experiments | — |
| `scripts/garch_masked/test_ablation.py` | Tests bias-match math, metric consistency, date alignment | — |
| Results: `results/ablation_vn_5seed/{vn30,vn100}_h5.json` and `..._h5_bm.json` | The 5-seed numbers | Cross-check the report's tables against these JSONs. |

Key statistical claims to verify wording of (already applied in the report):
- "DM not significant" is reported as "no statistically significant difference detected", NOT "equivalent".
- HAR keeps a lower mean QLIKE (VN30 +2.23%, VN100 +1.21%) but the gap is not significant.
- R² changes are stated in percentage points AND relative %.

---

## 3. [P3] Peripheral / exploratory code (lower priority)

Under `baselines/2026-08-21_har_anchored_residual/code/`: `experts.py`, `blend.py`, `gate.py`, `folds.py`,
`har_cv.py`, `run_experiment.py`, `run_walkforward.py`, `screen_features.py`, `screen_graph.py`,
`diagnostics.py`, `models.py`, `io_preds.py` — these belong to the earlier HAR-anchored / walk-forward
exploration and the S&P 500 walk-forward robustness check; they are NOT on the submitted paper's critical
path. Review only if auditing the broader exploration. `submission/soict_lstm_gat/` also has `run_all.py`,
`run_alpha.py`, `run_lstm.py`, `snapshots.py`, `train.py`, `evaluate.py` from the original per-stock pooled
experiment (produced `results/soict/`), superseded by the masked panel for the paper.

Throwaway experiment wrappers: `scripts/garch_masked/{run_sp500_seed123, combine_sp500_2seed}.py`,
`scripts/crawl_hose_hnx.py` (data crawl) — utility, not paper logic.

---

## 4. Paper and documents to review

### Paper (two versions)
- **SUBMITTED (VN-only):** `docs/paper/soict_harlstmgat.tex` (+ local `.pdf`, 8 pp) and
  `docs/paper/soict_paper_complete.md` (+ `.docx`). This is the version to treat as final.
- **DISCUSSION version:** `docs/paper/soict_harlstmgat_with_sp500.tex` (+ `.pdf`, 9 pp) and
  `..._with_sp500.md` (+ `.docx`). Adds an exploratory single-seed S&P 500 cross-market section and the
  output-parameterization robustness subsection. NOT submitted; for discussion.
- Figure: `docs/paper/diagrams/soict_harlstmgat.{png,svg,pdf}` (+ generator `generate_arch.py` + test).

What to check in the paper: (1) every number traceable to a `result.json`; (2) the baseline is HAR-X = 5-feature model
consistently; (3) objective wording; (4) statistical claims (DM significance, no equivalence overclaim);
(5) the √N methodology note; (6) GARCH described correctly; (7) horizons h1/h5 primary vs h10/h22 extended.

### Key reports (context, `docs/reports/`)
- `2026-08-22_masked_rich_floor1e2_clean.md` — the clean floor-1e-2 results (source of the paper's numbers).
- `2026-08-22_output_parameterization_robustness.md` — the output-param ablation (P2).
- `2026-08-22_sp500_masked_rich_quick_singleseed.md` — the exploratory S&P 500 single-seed check.
- `2026-08-22_lstm_qlike_blowup_diagnosis.md` — QLIKE floor blow-up diagnosis.
- `2026-08-22_dm_reporting_norms_literature.md` — DM reporting norms.
- `2026-08-22_graph_no_value_analysis.md` / `graph_findings_handoff.md` — graph-adds-no-OOS-value analysis.

---

## 5. Review focus areas (checklist)

1. **Leakage:** train/val/test are chronological; every scaler, edge, HAR/GARCH fit, and the GARCH basis
   use TRAIN rows only; purge = h between splits; per-node scalers not fit on val/test.
2. **Masking:** target scored only where BOTH the input window and the target are valid; loss is mask-aware;
   invalid nodes zero-filled do not enter loss/metrics.
3. **Metrics:** QLIKE positivity floor identical across compared models; R² consistent with MSE (same pooled
   arrays, test-mean denominator — note negative R² for GARCH/long-h is genuine, not a bug); date-clustered
   DM (not naive per-obs).
4. **Output parameterization (open):** the current suite uses z-score target + `1e-2*t_mean` floor; the
   robustness study finds node-scaled ratio parameterization removes QLIKE seed-instability. Decide whether
   this should be adopted before the final re-run (see §6).
5. **GARCH:** pseudo-return construction, rescale, fallback, and the basis guard.
6. **Statistical wording:** no "equivalence" from failure-to-reject; point estimates vs significance
   separated; percentage points vs relative % distinguished.
7. **Architecture description:** GAT reads raw features at t (parallel, SEQ-invariant), matches `MaskedRichNet` in `run_masked_rich.py` (the paper's model, not submission/model.py).

---

## 6. Known open items / in flux (decide before final re-run)

- **Output parameterization:** the ablation recommends node-scaled ratio parameterization (config C =
  ratio+exp, or D = ratio+softplus) over the current z-score+floor. Not yet applied to the full suite. If
  adopted, ALL tables must be re-run (5 seeds × 4 horizons × VN30/VN100, and S&P 500).
- **S&P 500:** only exploratory single-seed masked-rich results exist; a 2-seed mean was computed
  (`results/_seed123_root/`). A full multi-seed S&P 500 run with a scale-aware/positive output is future
  work. QLIKE for the deep models on S&P 500 is inflated by the floor at that scale — do not trust deep
  QLIKE on S&P 500 until re-run.
- **Citations:** verify Dugas et al. (softplus, NeurIPS 13 / 2000) and any ABDL / log-HAR references before
  use; the softplus positivity parameterization is a standard-DL choice, not a volatility-specific standard.

---

## 7. How to run tests / reproduce

- GPU venv: `.venv_gpu_encode/Scripts/python.exe` (torch, arch). Base python 3.14 has no CUDA torch.
- Masked-rich tests: `python -m pytest baselines/2026-08-21_har_anchored_residual/test/ -q`.
- Submission tests: `python -m pytest submission/soict_lstm_gat/tests/ -q`.
- GARCH/ablation tests: `python -m pytest scripts/garch_masked/ -q`.
- Reproduce one masked-rich cell (smoke): `python baselines/2026-08-21_har_anchored_residual/code/run_masked_rich.py vn30 1 --smoke --no-corr --data-root submission/soict_lstm_gat/data --price-dir data/raw/prices`.
- Self-contained reviewer package: `deliverables_20260822/` (code + paper + results + README + REPRODUCE.md,
  verified imports + smoke + 115 tests). Zip and share this for a self-contained review.

---

## 8. Out of scope
`archive/` (any depth) is retired code/data — do not report findings there. Data files under `data/` are
gitignored (obtain via the crawl scripts if needed). `.venv*`, `__pycache__`, `_tmp*` are not for review.
