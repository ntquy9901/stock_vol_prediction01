# Design — GBME+DAE (Denoising-Autoencoder representation → gamma-GBM)

## 1. Overview
Falsification baseline. Structural template = `baselines/2026-09-14_gbm_gnn_embed` (the closest analogue: a
frozen neural embedding → gamma-GBM with DM + spike + fit evidence). This baseline swaps the learned **GNN
node embedding** for an **unsupervised denoising autoencoder (DAE)** bottleneck over the tabular OWN-8+EARN
feature set (the Jahrer / Kaggle-Grandmaster paradigm), keeping the rest of the protocol identical so the
result sits in the same table.

## 2. Modules (`code/`)
| File | Purpose |
|------|---------|
| `dae_config.py` | Single source of truth for all tunable constants (swap rate, K, hidden, epochs, patience, lr, wd, batch, val len, seed, horizons, MIN_ROWS, PRED_CAP, GAIN_MIN, DM_ALPHA, KILL_HORIZONS, SPIKE_WINDOWS). GBM hyper-params and the QLIKE floor are reused from `full_matrix` (NOT duplicated). |
| `dae.py` | `DAE` (symmetric MLP AE), `swap_noise`, `standardize`, `val_len`/`val_split`, `train_dae` (batched GPU DAE training + learning curves + early stop), `default_trainer`, `frozen_z` (frozen single-basis embedding of every row). |
| `run_dae.py` | Walk-forward driver: GBME vs GBME+DAE, per-horizon JSON, DM, per-fold, spike robustness, verdict. Reuses `full_matrix` (load/panel/gbm/OWN/EARN/FL/SEEDS), `gnnhar_sp500._metrics5`, `vn_gbm_graph_stage1` folds, `stats.date_clustered_dm`, `overfit_check.classify_fit`. |

## 3. Data flow (per horizon `h`)
```
frames + earnings ──FM.panel──> panel a (rows dropna OWN+y, y = pk.shift(-h))
for each walk-forward fold k:
  trf = a[TRAIN_START .. ts-embargo)      # causal train
  tef = a[ts .. tend)                     # test
  if first eligible fold:                 # FROZEN BASIS (built once per horizon)
     burnin_dates = trf.dates
     zmap, curves = dae.frozen_z(a, feats=OWN-8+EARN, burnin_dates, K, ...)
        └─ standardize on burn-in cells ─ train ONE DAE (swap noise) ─ embed EVERY row
  z_train = zmap[trf rows]; z_test = zmap[tef rows]     # single shared basis, no per-fold refit
  fit GBME     = FM.gbm on [OWN-8 | EARN]        (seed-ensemble, clip [FL, PRED_CAP])
  fit GBME+DAE = FM.gbm on [OWN-8 | EARN | Z]    (seed-ensemble, clip [FL, PRED_CAP])
  score train / val(hold-out) / test
pool → date-clustered DM(GBME+DAE vs GBME) → gain → verdict → per-fold + spike robustness
```

## 4. Frozen-basis rationale (the central design decision)
The sibling `2026-09-14_gbm_gnn_embed` found that OOF-cross-fitting a NEURAL embedding produces an ill-posed,
drifting latent basis (a hidden representation is only defined up to rotation/permutation/sign, so stitching
`z` from separately-fitted networks across inner folds is ill-posed) which **detonated** long-horizon QLIKE
(design §9, `frozen_z`). This baseline therefore uses the **frozen-basis** approach from the outset: ONE DAE
trained on the burn-in window, frozen, embeds every row → a single shared basis, no per-fold refit, no basis
drift. This is the standard frozen-feature-extractor setup.

**Causality:** the DAE is UNSUPERVISED (reconstructs features, never sees a label `y`), and its weights +
feature standardiser are fit on burn-in rows only. Embedding a later fold's rows is out-of-sample to the
extractor; the only rows in-sample to the extractor are the burn-in rows themselves, which carries NO label
leakage (inherent to transfer learning). Verified by `test_frozen_z_single_basis_covers_all_rows` (the DAE
trains on burn-in dates only) and `test_outer_fold_causality` (mutating post-test-end rows leaves the fold
result unchanged).

## 5. Gates
- **Simplicity Gate:** reuses `full_matrix.gbm` unchanged; the DAE is a plain 3-hidden-layer MLP AE; no OOF
  machinery (frozen only), no graph. Passes.
- **Anti-Abstraction Gate:** direct use of `full_matrix`, `gnnhar_sp500._metrics5`, `stats`, `overfit_check`,
  `torch`. No wrappers. Passes.
- **Performance/Batching Gate:** the DAE trains on GPU-resident tensors in `C.BATCH` (4096) minibatches with
  fully-vectorised swap noise — no batch=1, no per-item main-thread loop, no per-step host↔device sync in the
  hot loop. The GBM is `HistGradientBoostingRegressor` (native multithreaded). Passes.

## 6. Numerical guards
- Prediction floor `FL = full_matrix.FL` (1e-8, single source) and upper cap `PRED_CAP = 10.0` (>> max
  observed HOSE variance ~5.3): both applied IDENTICALLY to GBME and GBME+DAE so the guard cannot bias DM.
- Standardiser guards zero-variance columns (std→1, maps to 0 not NaN).
- `_safe_dm` returns p=1.0 for identical loss series (tree ignores Z) or too-few-dates (h ≥ n_dates after
  spike exclusion).

## 7. Over/under-fit evidence (gate)
Each result JSON carries `metrics` (test) + `train_metrics` + `val_metrics` (all 5 metrics, both models) +
`fit_diagnostics` (classify_fit verdict for GBME+DAE) + `learning_curves` (per-epoch DAE reconstruction
MSE). `val_metrics` is a TRUE hold-out (GBM fit on `trf_e` = train minus the last `VALID_LEN` dates).
Note: the model name `"GBME+DAE"` is not in `overfit_check._LEARNED_PATTERNS`, so the pre-push gate’s
`_is_masked_rich_result` SKIPS the file (not a recognised learned result) → no block; the full evidence is
nonetheless present and passes `check_result_evidence(learned=("GBME+DAE",))` if checked. A shared module was
deliberately NOT edited to add a "dae" pattern (hard isolation, §3.F).

## 8. Expected result
NO-GO (matching the GNN-embed falsification). A learned embedding of an 8-feature own-history + earnings set,
fed back into the same gamma-GBM, is expected to be redundant/noise and to overfit rather than help OOS QLIKE.
