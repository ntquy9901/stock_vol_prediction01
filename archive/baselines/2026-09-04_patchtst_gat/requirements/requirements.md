# Requirements — PatchTST + GAT volatility baseline (2026-09-04)

## 1. Objective

Replace the **LSTM temporal backbone** of the VolGA model (`MaskedRichNet`, in
`baselines/2026-08-21_har_anchored_residual/code/run_masked_rich.py`) with a from-scratch
**PatchTST** encoder (Nie et al. 2023, ICLR — *"A Time Series is Worth 64 Words"*: patch the
lookback series into subseries patches, channel-independent, transformer encoder), and run it
**in parallel** with the existing vol→PK GAT spatial branch, concatenated at the head — exactly
where the LSTM branch (`h_lstm` [B,N,64]) currently sits.

Deliver:
- `PatchTSTEncoder` (patch embedding + `nn.TransformerEncoder` + projection → per-node embedding
  of width `hidden`=64, identical to the LSTM branch width).
- `PatchTSTRichNet` (mirror of `MaskedRichNet`; LSTM submodule swapped for PatchTST; **same** GAT
  branch + head; supports the two VolGA variants: no-graph = PatchTST only, and PatchTST+GAT).
- `train_patchtst` trainer (mirror of `train_masked_rich`; batched, masked MSE, early stop,
  learning curves, over/under-fit evidence).
- `run_patchtst_walkforward.py` CLI (mirror of `run_volga_walkforward.py`) so the coordinator can
  launch the GPU walk-forward sweep **later**.

## 2. Input / Output

- **Input per fold** (from `wf_enriched_panel.pack_fold`): `MaskedRichData` with
  `X_* [B,N,lookback,5]`, masks `nmask_*`/`tmask_*`, targets `y_*`, per-fold TRAIN-only
  `t_mean/t_std`, and the TRAIN-only vol→PK graph `adj_vol2pk`.
- **5 node features** (channels): `[parkinson_variance, har_weekly, har_monthly, market_pk,
  volume_zscore_22]`. **Target** = `parkinson_variance` (a variance σ², per the project note) at t+h.
- **Output**: `results/walkforward_patchtst/walkforward_patchtst_<market>_h<h>.json` with the same
  schema as the VolGA walk-forward result (metrics, metrics_per_seed, DM, train/val/test evidence,
  fit_diagnostics, learning_curves, per_fold), learned model keys = `PatchTST`,
  `PatchTST_wGAT_vol2pk`.

## 3. Success criteria (acceptance)

1. `PatchTSTEncoder` maps `[B,N,22,5] → [B,N,64]` (shape test passes).
2. Patch count equals `floor((seq - patch_len)/stride) + 1` (patching-math test passes).
3. Full net `PatchTSTRichNet.forward(x, adj_b)` returns `[B,N]` predictions for both variants
   (graph / no-graph).
4. No-lookahead / leakage sanity on a tiny real-data slice: `pack_fold` + `assert_no_leakage`
   pass (reused guards); train scalers + graph estimated on TRAIN rows only.
5. Real-data CPU smoke: 1 fold, epochs=2, ≤8 tickers runs without exception and returns a
   **finite** pooled QLIKE for every model.
6. All tests pass under **CPU only** (`CUDA_VISIBLE_DEVICES=`), no `pip install`, no new heavy deps.

## 4. Non-goals / out of scope

- **No full GPU training sweep** (GPU is busy with an overnight walk-forward chain). Only tiny CPU
  smokes. The GPU sweep is launched later by the coordinator.
- No edits to any existing file under `baselines/` (except this new folder), `submission/`, `src/`,
  `scripts/`. Shared modules are imported read-only.
- No hyperparameter tuning; defaults are documented, parsimony-biased choices.

## 5. Go / No-Go

- **GO** to hand off for coordinator review + GPU sweep when acceptance 1–6 all pass on CPU and the
  adversarial self-review findings are resolved or recorded.
- **NO-GO** if any test fails, if the net touches the GPU during smokes, or if any shared file is
  modified.

## 6. Honest prior / risk (stated up front, per §1 Think Before Coding)

The project's standing finding is that **HAR is very hard to beat** on this VN panel, and prior
**non-LSTM backbones failed badly** (CryptoMamba test R² ≈ −1.10). Transformers are **more
overfit-prone** on a small VN panel (~100 tickers, short daily history). The expectation is that
PatchTST is **unlikely to beat HAR/HAR-X** here; this baseline is a *negative-control-quality*
architecture probe, not a presumed winner. Design choices bias toward parsimony (small d_model,
depth=2, dropout, few patches) to give it the fairest shot without overclaiming.
