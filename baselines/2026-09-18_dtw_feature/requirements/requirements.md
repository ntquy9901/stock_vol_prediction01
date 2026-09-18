# DTW self-similarity feature -> GBME (HOSE)

## Provenance / germ
Transferable germ of Nakagawa & Yoshida (2022) "Time-series gradient boosting tree" — the DTW
(dynamic-time-warping) time-series *shape* matching idea. The paper builds DTW into a bespoke tree
SPLIT criterion; the transferable germ under this project's variance-QLIKE thesis is instead a
**strictly causal DTW feature** appended to the champion own-history gamma-GBM (GBME). We do NOT build a
new split criterion — we test whether "is my recent volatility path shaped like my own past templates"
carries QLIKE-relevant information beyond the HAR/own-history feature set.

## Mechanism (what the feature is)
For each stock-day `t`:
1. Take the trailing window of length `W` of the stock's own log-variance `log(parkinson_variance)`,
   ending at day `t` (past-only, causal).
2. Standardize it with that stock's TRAIN-region log-variance mean/std (train-only scaling).
3. Compute a Sakoe-Chiba **banded DTW distance** between that window and each of a small set of
   **reference templates** built causally from the stock's OWN further-past history — the per-band
   centroids of its trailing windows over the fold's train period (low / mid / high vol-level bands).
4. The DTW distances (one per template) are the extra causal features fed, alongside OWN-8 (+ earnings),
   into the champion gamma-GBM.

## Arms
- **GBME** — champion own-history gamma-GBM (OWN-8 + earnings), the base to beat.
- **GBME+dtw** — GBME + the DTW distance features (real templates, real trailing window).
- **GBME+dtw_placebo** — GBME + DTW distances computed against the SAME templates but from a
  **date-shifted trailing window** (window ended `DTW_PLACEBO_SHIFT` business days earlier, still
  causal). Mandatory per the 2026-09-10 seasonal-artifact lesson: a shifted/wrong query must NOT
  reproduce a real gain; if the placebo reproduces the gain, the "recent shape" story is just
  vol-level autocorrelation and the feature is spurious.

## Inputs / outputs
- Input: `data/processed_enriched/hose/*.csv` (via `full_matrix.load`), real crawled VN earnings
  `results/gamma_gbm/hose_earnings_combined.parquet`.
- Output: `results/gamma_gbm/dtw_hose_h<h>.json` for h in {1,5,10,22}, atomically checkpointed per fold,
  each carrying `metrics` / `train_metrics` / `val_metrics` / `fit_diagnostics` (for GBME, GBME+dtw,
  placebo), date-clustered DM, per-fold QLIKE, and COVID/2022/Apr-2025 spike robustness.

## Success criteria (pre-registered kill)
GO iff **GBME+dtw beats GBME (DM-significant, QLIKE) at >= `KILL_MIN_HORIZONS` (=2) horizons**, the win
is **spike-robust** (survives excluding the regime-spike windows) at those horizons, **AND the placebo
does NOT beat GBME** at those horizons. Otherwise NO-GO.

## Prior belief
Expect NO-GO: DTW captures *shape/pattern* similarity, but QLIKE penalizes *magnitude* error, and vol
level is already carried by HAR/own-history features. Run honestly for a concrete verdict; do not
fabricate.

## Go/No-go verification (Definition of Done)
- 5 subfolders present (requirements/design/code/code_review/test).
- `pytest` green under `.venv_gpu_encode`; diff-cover C0=100% / C1>=95% on changed lines.
- Real HOSE run complete, all 4 horizons, **`n_folds == 8` verified in each JSON**.
- Code review (adversarial) done; findings triaged.
- Summary report written. Coordinator (not this baseline) pushes.
