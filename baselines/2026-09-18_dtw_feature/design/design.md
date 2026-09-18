# Design — DTW self-similarity feature -> GBME

## Data flow (per horizon h, per outer walk-forward fold k)
1. `full_matrix.load("hose")` -> per-ticker frames (each already carries `logpk` via `S1._feat`),
   sectors, earnings. `_load_earn` swaps in real crawled VN dates from
   `results/gamma_gbm/hose_earnings_combined.parquet`.
2. `full_matrix.panel(frames, edates, h)` builds the pooled panel `a` with `y = pk.shift(-h)` and the
   4 causal earnings features.
3. Walk forward over `vn_gbm_graph_stage1.FOLDS` from `TRAIN_START`, embargo `int(1.6h)+5` days. Per
   fold: `trf` (train, `date < ts - embargo`), `tef` (test, `[ts, tend)`); train minus the trailing
   `VALID_LEN` dates -> `trf_e` / `vaf` (a real val slice for fit diagnostics).
4. **DTW columns** (`dtw_feature.fold_features`): built for the concatenated `combo = [tef, trf_e, vaf]`
   using per-ticker series from `frames`, with the fold train boundary `ts - embargo`:
   - Templates + scaling come ONLY from train-region rows (`date < boundary`).
   - Query window for a row ends at that row's own date (past-only). Placebo query window ends
     `DTW_PLACEBO_SHIFT` business days earlier (still `<=` the row date -> causal).
   - Rows of a ticker with too little history to form templates -> NaN feature (HGBR handles NaN
     natively; no silent zero-fill that would fabricate a "0 distance").
5. Three arms, each the champion `full_matrix.gbm` (HGBR gamma, 300 trees, lr 0.05, 31 leaves, l2 1),
   seed-ensembled over `FM.SEEDS`:
   - GBME cols = OWN-8 (+ EARN) ; GBME+dtw cols = base + real DTW cols ; placebo cols = base + placebo
     DTW cols. Predictions floored to `FL` and clipped to `[FL, PRED_CAP]`.
6. Pool folds -> `_pool_doc`: 5-metric train/val/test for all arms, `fit_diagnostics`
   (`overfit_check.classify_fit`), date-clustered DM (`stats.date_clustered_dm` via `_safe_dm`),
   per-fold QLIKE, spike robustness. Atomic checkpoint per fold.

## DTW implementation (own, vectorized — `dtaidistance`/`tslearn` not installed)
- `banded_dtw_batch(Q[B,W], template[W], band)`: Sakoe-Chiba banded DP, squared-diff local cost,
  vectorized over the batch axis `B`; returns `sqrt(D[B,W,W])`. Identical window vs itself -> 0.
- `build_templates(train_windows_std, bands)`: sort train windows by mean level, split into the
  configured quantile bands, per-band centroid = mean window -> one template per band.
- Standardization: per-ticker train log-variance mean/std (train-only) applied to the whole series so
  query and template share one causal scale.

## Causality guarantees
- Templates + (mu, sd) scaling use train-region rows only (`date < ts - embargo`).
- Every query/placebo window uses indices `<=` the row's own position (front-clamped, never future).
- No cross-day, no cross-ticker leakage: each ticker's features come only from its own series.
- Test: perturbing FUTURE values of a series leaves an earlier row's DTW feature unchanged.

## Gates
- **Simplicity Gate**: reuses `full_matrix` champion GBM + `stats`/`metrics`/`overfit_check`; the only
  new code is the DTW feature + a thin walk-forward driver copied from the sibling `run_glm_anchor`.
- **Anti-Abstraction Gate**: no DTW library wrapper — one small numpy banded-DP function.
- **Performance/Batching Gate**: DTW is vectorized over the batch of query windows (no per-row Python
  DTW loop); `W=22`, band radius 4 -> the DP is ~22x9 vectorized steps per template. HGBR handles NaN,
  so no dense fill pass.

## Config (all tunables in `dtw_config.py`, inline-commented)
`DTW_WINDOW`, `DTW_BAND`, `DTW_BANDS` (template count = len), `DTW_PLACEBO_SHIFT`,
`DTW_MIN_TRAIN_WINDOWS`, `VALID_LEN`, `PRED_CAP`, `HORIZONS(_SMOKE)`, `MIN_ROWS`, `GAIN_MIN`,
`DM_ALPHA`, `KILL_MIN_HORIZONS`, `SPIKE_WINDOWS`.

## Model naming
`GBME`, `GBME+dtw`, `GBME+dtw_placebo` — no neural tokens; the over/under-fit gate treats the file as a
non-learned (deterministic-boosting) result, and the full fit evidence is carried regardless.
