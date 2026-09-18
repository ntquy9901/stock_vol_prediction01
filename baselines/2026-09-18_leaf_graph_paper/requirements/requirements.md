# Requirements — leaf-graph FINAL paper-grade results (3-seed, XGB-only, earnings ablation)

## Goal
Produce the DEFINITIVE, paper-grade version of the leaf-cooccurrence-graph falsification result. The proof-of-concept
(`baselines/2026-09-18_gbm_leaf_graph`, committed `947296f1`) showed XGB+leafgraph beats the un-smoothed XGB base on
HOSE (h1 +0.25% DM p=4e-4, h5 +0.26% p=3e-5, h10 +0.12% p=0.005; spike-robust) and is NO-GO on SP500 (alpha->0).
The paper needs the result reported for the two headline models ONLY plus an earnings leave-one-out ablation.

## Model set (STRICT)
Report **XGB (base)** and **XGB+leafgraph** ONLY. The champion HGBR `GBME` is **not fit or reported** here
(`FM.gbm` is never called). Fewer models = less RAM (SP500) and a clean two-model paper table.

## Method (unchanged from the committed baseline — reused, not reinvented)
Per test day, an XGBoost gamma booster (capacity-matched to the champion HGBR gamma) produces base predictions; its
per-tree leaf indices form a leaf-Hamming similarity over the same-day cross-section; a per-day kNN(k=10) graph
smooths the base `y_hat = (1-alpha)*y_xgb + alpha*mean_kNN`, with `alpha` fit on a trailing val slice per fold
(causal, frozen for test). Isolate the graph = DM(XGB+leafgraph vs XGB).

## What this run delivers vs the committed baseline
1. **3 seeds** `(0,1,2)` (config `SEEDS`; XGB base seed-ensembled).
2. **Feature-set switch**: `full` = OWN-8 + EARN-4; `noearn` = OWN-8 only (leave-one-out removal of the 4 EARN
   features). `noearn` isolates the earnings contribution on the base AND tests whether the leaf-graph GO survives
   without earnings. Rows are identical across `full`/`noearn` (panel dropna is on OWN+y only) -> directly comparable.
3. **HOSE + SP500** both markets.
4. **Streaming train metrics** (sufficient statistics `_Stream`) so SP500 8-fold x ~500-stock daily graphs do not
   OOM. Only val/test arrays are pooled (needed for DM). One (market, feature-set, horizon) per process (`--horizon`).
5. **Distinct output filenames**: `results/gamma_gbm/leaf_graph_paper_<market>_<full|noearn>_h<h>.json`.

## Inputs / Outputs
- Input: `data/processed_enriched/{hose,sp500_clean}/*.csv`; HOSE real crawled earnings
  `results/gamma_gbm/hose_earnings_combined.parquet`; SP500 earnings from `FM.load`.
- Output: up to 16 JSONs = (market {hose,sp500}) x (feature-set {full,noearn}) x (h {1,5,10,22}). Each doc reports
  XGB + XGB+leafgraph 5-metric train/val/test, fit_diagnostics, date-clustered DM (XGB+leafgraph vs XGB),
  per-fold QLIKE + spike-robustness (HOSE only), fitted alpha per fold + mean, `n_folds`, `seeds`, `features`.

## Acceptance criteria (go/no-go)
- All JSONs written with **`n_folds == 8`** (per-fold checkpoints -> an early read is partial; verify final).
- Result JSONs PASS `scripts/quality_gate/check_overfit_evidence.py` (XGB / XGB+leafgraph fit == ok).
- Tests pass (`.venv_gpu_encode` pytest); diff-cov C0=100% / C1>=95% on new code.
- Adversarial code review with all HIGH/MAJOR fixed.
- Pre-push quality gate PASS (coordinator runs the push; this baseline only creates files + runs).
- Report answers: (a) earnings' marginal effect on the XGB base per market (full vs noearn); (b) does the HOSE
  leaf-graph GO survive the no-earn ablation; (c) is SP500 still alpha->0?

## Non-goals
- Not re-deriving the leaf-graph math (reused). Not tuning k/XGB capacity (frozen = champion-matched). No GBME. Not
  running any git command (coordinator handles git sequentially). `archive/` out of scope.
