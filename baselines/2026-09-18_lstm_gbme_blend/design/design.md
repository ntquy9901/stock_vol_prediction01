# LSTM-feature / GBME-blend baseline — design / plan

## 1. Data flow
```
full_matrix.load(market) -> per-ticker enriched frames (S1._feat: logpk, rq, mr_*)
  -> _load_earn: swap in real crawled VN announcement dates for HOSE (else none)
FM.panel(frames, edates, h) -> pooled panel `a` with OWN-8(+EARN) features + y = pk.shift(-h)
lstm_feat.build_sequences(a, cols_base, SEQ_LEN) -> X_all [N, SEQ_LEN, F] (causal per-ticker windows),
                                                   logvar_all [N] = log(max(y, FL))
per walk-forward fold k (S1.FOLDS, TRAIN_START, embargo = 1.6h+5 days):
  pos_train = rows in [TRAIN_START, ts-embargo);  pos_test = rows in [ts, tend)
  lf_train = oof_train_feat(...)   # inner temporal K-fold OOF LSTM predictions (variance space)
  lf_test  = test_feat(...)        # one LSTM on the full train window, predicts test rows (+ learning curves)
  trf[lstmfeat]=lf_train; tef[lstmfeat]=lf_test; split trf into trf_e / vaf (trailing VALID_LEN dates)
  GBME     = seed-ens gamma-GBM on [OWN-8|EARN]         (FM.gbm, fit on trf_e, predict combo)
  GBME+lstmfeat = seed-ens gamma-GBM on [OWN-8|EARN|lstmfeat]
  blend    = w*GBME + (1-w)*standaloneLSTM,  w = argmin_w val QLIKE (per fold)
pool over folds -> 5-metric train/val/test, fit_diagnostics(FEAT), date-clustered DM, verdict, err-corr,
                   per_fold_qlike + spike_robustness (HOSE). One JSON per horizon, atomic checkpoint per fold.
```

## 2. Why an LSTM feature, not a latent embedding
The germ of arXiv:2505.23084 is "hand the deep sequential model's OUTPUT to the tree". Unlike the GNN-embed
sibling (2026-09-14), whose penultimate hidden lives in an arbitrary rotation/permutation basis (per-fold OOF
vs full-train embeddings drift, detonating long-horizon QLIKE), the LSTM here emits a CALIBRATED scalar
(log-variance in the target space). OOF-train and full-train-test predictions are therefore estimates of the
SAME quantity on the SAME scale — no basis-drift, so a straightforward per-fold OOF (not the frozen-basis
workaround) is leakage-safe and consistent.

## 3. Leakage-safe cross-fitting (design decision)
Train-row lstmfeat by inner temporal K-fold (INNER_K blocks): each block is predicted by an LSTM whose
training dates exclude that block. Test-row lstmfeat by one LSTM trained on the full train window. This is the
established repo pattern (embed.py) that keeps the GBM's train feature honest, so the pre-push overfit gate
sees a genuine val->test gap rather than an inflated in-sample one.

## 4. Gates (SDD)
- **Simplicity Gate:** reuses FM.gbm / metrics / stats / overfit_check / S1 walk-forward unchanged; the only new
  code is the LSTM extractor + the blend/reporting driver. No new abstraction beyond the sibling template.
- **Anti-Abstraction Gate:** plain `torch.nn.LSTM`; no wrapper framework. GBM hyper-params + FL single-sourced
  from `full_matrix`; every LSTM/walk-forward tunable single-sourced in `lstm_blend_config.py`.
- **Performance / Batching Gate:** sequences are batched tensors on GPU; training steps over BATCH sequences
  (no batch=1); inference is chunked (`_forward_batched`) so a hundreds-of-thousands-row HOSE train window does
  not OOM (measured: the whole-set forward tried to allocate ~20 GiB and was replaced by bounded chunks).

## 5. Files
- `code/lstm_blend_config.py` — single source of tunable constants (SEQ_LEN, HIDDEN, LR, WD, BATCH, EPOCHS,
  PATIENCE, MIN_EPOCH, VALID_LEN, LSTM_SEEDS, INNER_K, BLEND_GRID, HORIZONS, MIN_ROWS, PRED_CAP, kill criterion,
  SPIKE_WINDOWS).
- `code/lstm_feat.py` — SeqLSTM, batched train/infer, sequence builder, temporal val split, inner-block OOF
  cross-fitting (oof_train_feat / test_feat), variance clip.
- `code/run_lstm_blend.py` — walk-forward driver, blend-weight selection, pooling / DM / verdict / err-corr,
  atomic checkpoint.

## 6. Kill criterion
Config: GAIN_MIN=0, DM_ALPHA=0.05, KILL_HORIZONS=(1,5). Success iff GBME+lstmfeat beats GBME at both, spike
-robust. Expected: NO-GO (own-history dominates; LSTM re-encodes it — high err-corr).
