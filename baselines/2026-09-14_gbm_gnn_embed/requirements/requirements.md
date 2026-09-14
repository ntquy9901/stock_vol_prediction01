# Requirements — GBME ⊕ 2-layer GNN node-embedding (5th graph falsification)

> Pre-registered FALSIFICATION. Expected NO-GO (four prior graph attempts all failed on OOS QLIKE).
> Correctness/honesty over a positive result.

## Objective
Test whether a **learned** 2-layer GNN node embedding `z_{i,t} ∈ R^{N_HID}`, concatenated as extra
features into the champion `GBM+earn` tree, beats `GBM+earn` on out-of-sample QLIKE. The GNN is a feature
extractor (penultimate hidden of the faithful `GNNHAR`); the gamma-GBM remains the predictor.

```
GBME      = GBM(gamma) <- [ OWN-8 | EARN-4 ]                 (baseline)
GBME+z    = GBM(gamma) <- [ OWN-8 | EARN-4 | z_{i,t} (N_HID) ]   (this experiment)
```

## Inputs
- Enriched per-ticker OHLCV+features: `data/processed_enriched/<market>/*.csv` (via `FM.load`).
- OWN-8 = `config.own_set(FM.OWN)` (FM.OWN minus `rq`), the paper own-history set (single-sourced from
  `baselines/2026-09-13_paper_models/code/config.py`, loaded by path to avoid the bare-`config` collision).
- EARN-4 = `FM.EARN` (SP500 = `sp500_earnings.parquet`; HOSE = `hose_earnings_combined.parquet` if present,
  else earnings features are inert and the baseline degrades to plain `GBM`).
- GNN node features = OWN-8 (matched to the GBM input, so `z` is a learned graph-aware transform of the SAME
  own-history features; the marginal value tested is the graph mixing).
- Target = `parkinson_variance.shift(-h)` (variance), horizons {1,5,10,22}. Identical to the sibling tables.

## Outputs
- `results/gamma_gbm/gnn_embed_<market>_h<h>.json` — ONE FILE PER HORIZON (atomic tmp+replace checkpoint,
  Colab/disconnect-resilient). Each carries, for models `{GBME, GBME+GNN-embed}`:
  - `metrics` (test), `train_metrics`, `val_metrics` — all 5 (mse/rmse/mae/r2/qlike),
  - `fit_diagnostics` (overfit_check.classify_fit verdict for the learned `GBME+GNN-embed`),
  - `learning_curves` (per-epoch train/val QLIKE of the full-train embedding GNN, per seed),
  - `dm` (date-clustered DM: `GBME+GNN-embed_vs_GBME`), `gain_pct`, `verdict`,
  - HOSE also: `per_fold_qlike` + spike-robustness (`qlike_ex_spike`, per-fold, floor sensitivity).

## Critical correctness (no stacking leakage)
`z` for the GBM TRAIN rows is OUT-OF-FOLD (inner temporal K-fold, K=`INNER_K`): every train row is embedded
by a GNN that did NOT see it. `z_test` comes from a GNN trained on the FULL train window. Graph, feature
scaler, GNN, and GBM all fit on train / inner-train rows only; target embargo at the train/test boundary.

## Success / kill criterion (pre-registered)
`success = True` iff `GBME+z` beats `GBME` (gain>0 AND date-clustered DM p<`DM_ALPHA`) at BOTH h1 and h5.
Otherwise NO-GO. Do NOT chase in-sample or MAE-only wins. HOSE spike-robustness must hold before any
positive HOSE claim (verdict sign must survive excluding the COVID/2022/Apr-2025 shock folds).

## Acceptance criteria (Definition of Done)
- Unit tests pass (embed shape, OOF coverage + causality, signal-recovery + noise-neutrality of the z→GBM
  plumbing, run-structure smoke with the required evidence keys, verdict logic, atomic checkpoint).
- diff-cover C0=100% / C1≥95% on changed lines; ruff-F clean; config-hardcode clean.
- 3-layer adversarial code review (incl. a leakage lens + a performance lens) with critical/major fixed.
- HOSE run at FULL config (8 folds × 3 seeds × 4 horizons) with full evidence + honest verdict.
- SP500 Colab notebook runnable on a fresh clone (all imported modules tracked), resilient committer.

## Go / No-Go
- GO to report a positive result ONLY if the pre-registered criterion holds AND (HOSE) survives spikes.
- NO-GO (expected): report the measured QLIKE gaps + DM p-values + fit diagnostics honestly.
