# Design — QLIKE-loss / HAR-X-anchor

## Data flow
Reuse the delivered walk-forward read-only: build_enriched_panel + make_folds + pack_fold (lb10, folds=7),
HAR-X via _har_ols_preds, metrics/DM via run_masked_rich (_metrics/_dm_all/_pred_dict/_ens/OF.classify_fit),
VolGA edge via directed_vol2pk_hmatched, cell dump via _cell_rows/_split_dates (all imported, not modified).

## New pieces
- `qa_train.train_deep(D, cfg, seed, use_graph, adj, loss, anchor, harx, clip)` — vendors MaskedRichNet + the
  scaling/early-stop scaffolding and swaps the loss/forecast:
  - anchor=none forecast = clamp(z_out*std+mean, floor); anchor=harx forecast = max(HAR-X*exp(clip(z)), floor).
  - loss=mse: masked MSE on the z-scored target (none) or on z_true=log(y/HARX_oof) (harx);
    loss=qlike: masked QLIKE on the positive forecast. Early-stop on the validation value of the training loss.
  - Pure forecast/loss math (node_floor, zscore_forecast, anchor_forecast, qlike_np, residual_target) is unit-tested;
    the torch loop is # pragma: no cover (GPU, smoke-tested via the baseline run, like train_masked_rich).
- `qa_config.py` — ANCHOR_CLIP + OOF split params + the {loss}x{anchor} option lists (single source of truth).
- Anchor's train-cell HAR-X = OOF HAR-X (xgb_oof.oof_harx, fit strictly on earlier anchors); warm-up NaN cells
  fall back to in-sample HAR-X (train-only; val/test use the fold HAR-X, so evaluation stays leakage-free).

## Gates
Simplicity/Anti-abstraction: reuse delivered machinery, one small trainer. Performance: batched GPU training,
5-seed ensemble, no per-item loop. Passes.

## Files
code/{qa_config.py, qa_train.py, run_qlike_anchor.py}; results/qlike_anchor/qa_<market>_<loss>_<anchor>_h<h>.json
(+ cells/ parquet with --dump-cells). Colab: notebooks/train_qlike_anchor_sp500_colab.ipynb.
