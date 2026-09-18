# LSTM-feature / GBME-blend baseline — requirements

## Objective
Close, concretely and on out-of-sample variance QLIKE, the direction "deep sequential model (+) tree
ensemble" — the transferable germ of arXiv:2505.23084 ("GBDT + LSTM for Investment Prediction") — adapted to
this repo's variance-QLIKE thesis. A small per-fold LSTM reads the OWN-8 (+earnings) own-history feature
sequence and predicts the h-step-ahead log-variance; that prediction is handed to the champion own-history
gamma-GBM (GBME) as one extra causal feature (GBME+lstmfeat), and also blended convexly with GBME. The test:
does the deep sequential signal add anything ORTHOGONAL to the tree?

## Prior (why this is expected NO-GO)
This repo already found GNN/deep (+) GBM blends fail with error-correlation ~0.98 (own-history dominates), and
a constrained-stacking-with-diverse-bases attempt was just falsified. This baseline runs the LSTM variant to
document the same closure on QLIKE, not to fabricate a win.

## Inputs / outputs
- Input: HOSE (primary) enriched panel via `full_matrix.load` / `panel`; OWN-8 = `paper_models.own_set(FM.OWN)`;
  earnings = real crawled VN announcement dates (`hose_earnings_combined.parquet`) if present, else none.
- Models compared per horizon h in {1,5,10,22}: `GBME` (champion, [OWN-8|EARN]), `GBME+lstmfeat`
  ([OWN-8|EARN|lstmfeat]), `blend` (w*GBME + (1-w)*LSTM, w fit on validation QLIKE per fold).
- Output: `results/gamma_gbm/lstm_blend_hose_h<h>.json`, one per horizon, atomic per-fold checkpoint. Each
  carries `metrics`/`train_metrics`/`val_metrics` (all 5: mse/rmse/mae/r2/qlike) + `fit_diagnostics` +
  `learning_curves` for the LEARNED model, `dm`, `gain_pct`, `verdict`, `blend_weights`, `lstm_standalone`
  (standalone LSTM QLIKE + error-correlation vs GBME), and (HOSE) `per_fold_qlike` + `spike_robustness`.

## Success / kill criterion (pre-registered)
- A horizon "beats" iff QLIKE gain(GBME+lstmfeat vs GBME) > GAIN_MIN AND date-clustered DM p < DM_ALPHA.
- Direction SURVIVES iff GBME+lstmfeat beats GBME at BOTH KILL_HORIZONS (1, 5) AND is spike-robust (verdict
  keeps sign when the COVID/2022/Apr-2025 windows are excluded). Otherwise NO-GO (expected).
- The standalone-LSTM<->GBME error-correlation is reported to explain any null (a ~0.9+ correlation means the
  LSTM re-encodes the same own-history signal the tree already has).

## Leakage / rigor constraints
- LSTM trains on the causal train window only (train-window rows as targets), val slice for early stop; test
  predictions are strictly out-of-sample.
- Train-row lstmfeat is produced by OUT-OF-FOLD inner temporal K-fold (no stacking leakage); test-row lstmfeat
  by one LSTM trained on the full train window. Feature/target standardisation and the early-stop split use
  train / inner-train rows only.
- Sequences are causal (each uses only its ticker's own past feature rows).
- QLIKE floor FL and clip [FL, PRED_CAP] identical across all compared models.

## Go / no-go gates
- pytest green; diff-cover C0=100% / C1>=95% on changed lines.
- Pre-push overfit-evidence gate passes (learned model `GBME+lstmfeat` carries train/val/test + learning_curves
  and is not over/under-fit when pooled over folds).
- Full HOSE run verified `n_folds == 8` per horizon before the table is trusted.
- Performance: LSTM batched on GPU (no batch=1); batched inference so a HOSE train window does not OOM.
