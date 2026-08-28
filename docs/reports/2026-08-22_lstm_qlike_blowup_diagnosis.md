# LSTM QLIKE blow-up on VN100 h1 — diagnosis

Date: 2026-08-22
Scope: `baselines/2026-08-21_har_anchored_residual/code/{masked_rich.py,run_masked_rich.py}`,
result `results/masked_rich/vn100_h1/result.json`. Analysis on CPU (GPU busy); one instrumented
re-run (VN100 h1, 3 LSTM seeds + 1 GAT seed, batch 64, full 20-epoch config with early stopping).

## Summary (6 lines)
1. The no-graph LSTM's QLIKE = 1.02 (vs HAR 0.50, LSTM+wGAT 0.54) is **not a code bug and not a
   seed artifact**: all three LSTM seeds individually blow up (1.10 / 1.24 / 1.33), and the metric,
   scalers and inputs are correct (MSE/MAE are at parity with HAR).
2. It is a **tail pathology of the deep model interacting with the aggressive per-node positivity
   floor**: the LSTM occasionally collapses a prediction to the floor `1e-3*t_mean` (~500x below the
   typical target), and QLIKE (`y/yhat`) turns each such observation into a per-obs loss of ~400–1000.
3. **49 observations out of 46,308 (0.1%)** — where the ensemble LSTM sits within 2x of the node
   floor — account for **52.8% of the LSTM total QLIKE mass and ~99% of its excess over HAR**.
4. HAR **never** approaches the floor (min prediction 1.0e-4, floor ~4e-7), so its linear structure
   cannot produce these collapses; that is why only the deep model is affected.
5. The GAT branch **genuinely reduces** the pathology (floor hits 49→21; on the same 49 collapse-obs
   GAT's QLIKE sum is 12,384 vs the LSTM's 27,276) via neighbour smoothing — a real, quantifiable
   "prevents near-zero predictions" effect — but it only halves the damage (GAT 0.77 > HAR 0.50 in
   this single-seed check).
6. The entire QLIKE gap is **floor-sensitive**: raising the prediction floor from `1e-3*t_mean` to
   `1e-2*t_mean` drops LSTM QLIKE 1.12→0.56 and GAT 0.77→0.54; at `1e-1*t_mean` all three are at
   parity (~0.50). The point-forecast metrics (MSE/MAE/R²) are already at parity.

## Verdict
**Genuine deep-model QLIKE instability that the graph partially regularizes — amplified by an
aggressive positivity floor. Not a bug, not a data problem, not a seed artifact.**

- (a) Code/data bug: **ruled out.** QLIKE metric is standard and shared-floored identically across
  models; MSE = 2.37e-7 and MAE ≈ 2.9e-4 are identical for HAR/LSTM/GAT, so point predictions and
  the train/val/test pipeline are sound. The 5 per-node feature scalers are applied to all 5 columns
  (`masked_rich.py:189-190,217`), inputs to LSTM and GAT are the same `D.X_te` (GAT only adds the
  day-t slice + adjacency), and `market_pk`/`volume_zscore` are train-fit and reused for test.
- (b) Seed artifact: **ruled out.** Each seed blows up on its own (1.10 / 1.24 / 1.33); the ensemble
  (1.12) does not — a single bad seed is not dragging a healthy ensemble.
- (c) Genuine instability + graph regularization + floor amplification: **confirmed** (evidence below).

## Decisive evidence (instrumented VN100 h1 re-run)

Per-seed QLIKE (LSTM, no graph):

| seed | QLIKE | MSE | MAE |
|---|---|---|---|
| 42 | 1.0976 | 2.368e-7 | 2.91e-4 |
| 123 | 1.2367 | 2.371e-7 | 2.90e-4 |
| 2026 | 1.3322 | 2.365e-7 | 2.84e-4 |
| ens(3) | 1.1157 | 2.366e-7 | 2.88e-4 |
| HAR | 0.5004 | 2.370e-7 | 2.93e-4 |
| GAT vol2pk (seed 42) | 0.7726 | 2.391e-7 | 2.97e-4 |

Floor-hit fractions (prediction within 2x of the per-node floor `1e-3*t_mean`):

| model | within-2x-floor | n | min pred |
|---|---|---|---|
| HAR | 0.000% | 0 | 1.00e-4 |
| LSTM ens(3) | 0.106% | 49 | 4.17e-7 |
| LSTM s42 / s123 / s2026 | 0.11% / 0.14% / 0.21% | 51 / 66 / 95 | ~4e-7 |
| GAT vol2pk s42 | 0.045% | 21 | 4.17e-7 |

Mass concentration (LSTM ensemble):
- Top-1% obs (n=463) hold **62.3%** of the LSTM total QLIKE mass; mean under-prediction there
  `y/yhat` = 68.6x.
- The **49** obs within 2x of the floor contribute **27,276** QLIKE = **52.8%** of the LSTM mass.
  On those **same 49 obs**, HAR's QLIKE sum is **32** (~0.65 each, normal) and GAT's is **12,384**.
- Ranked by per-obs (LSTM − HAR) QLIKE difference: top-10 obs hold 66.5% and top-49 obs hold
  **99.3%** of the entire LSTM-over-HAR excess. The gap is ~50 observations, not a broad shift.
- Worst-obs tickers are concentrated: HHV (65 of the top-463), KOS (22), HVN/VIB (13), VHC (11).

Floor sensitivity (re-clamp existing predictions upward; QLIKE floor 1e-8 unchanged):

| prediction floor | HAR | LSTM ens(3) | GAT vol2pk |
|---|---|---|---|
| 1e-3 * t_mean (current) | 0.5004 | 1.1157 | 0.7726 |
| 1e-2 * t_mean | 0.5004 | 0.5629 | 0.5376 |
| 5e-2 * t_mean | 0.5004 | 0.5058 | 0.5120 |
| 1e-1 * t_mean | 0.5004 | 0.4978 | 0.5075 |

HAR is invariant (it never predicts near the floor); the deep models converge to HAR-parity as the
floor is lifted, confirming the gap lives entirely in a handful of near-floor collapses.

## Mechanism
On ~0.1% of test observations the LSTM's normalized output maps to a value at or below the node
floor (raw output ≤ `1e-3*t_mean`, i.e. ≈0 or negative on the normalized scale), so `infer()` clamps
the prediction to ~4e-7 while the true target is ~2e-4 (ratio ~500). QLIKE `= y/yhat − log(y/yhat) − 1`
grows without bound as `yhat→0`, so each collapse adds ~400–1000 to the sum; 49 such obs move the
mean by ~0.6. MSE/MAE barely move because the absolute error on a ~2e-4 target is tiny. HAR's linear
map cannot emit near-zero values (min 1e-4), so it is immune. The GAT branch concatenates a
neighbour-aggregated signal into the head, which pulls collapsing nodes back toward peer levels and
prevents roughly half of the collapses (49→21 floor hits; ~55% less tail mass on the shared obs) —
a genuine but partial stabilization.

## Implication for the paper
- The "graph stabilizes QLIKE by preventing near-zero predictions" claim is **real and quantifiable**,
  but it is a **tail effect on ~0.1% of observations**, not a broad accuracy gain: MSE/MAE/R² are at
  parity across HAR/LSTM/GAT, and ~99% of the LSTM-vs-HAR QLIKE gap is ~50 collapsed obs.
- The magnitude is **fragile to the positivity floor**: a gentler floor (`1e-2*t_mean`) or prediction
  winsorization collapses the LSTM/GAT/HAR QLIKE differences to near-parity. Any headline QLIKE
  comparison should either (i) report the floor sensitivity and frame the graph's role as tail
  damage-control on prediction collapse, or (ii) adopt a less aggressive floor / winsorize deep
  predictions, in which case the three models tie on QLIKE as they already do on point accuracy.
- Recommended framing: on VN100 h1 the deep models match HAR on point-forecast accuracy; the raw
  QLIKE ranking is dominated by ~0.1% deep-model prediction collapses under the `1e-3*t_mean` floor,
  which the GAT branch partially prevents — a stabilization finding that should be reported with its
  tail/floor caveat rather than as a general QLIKE improvement.

## Artifacts
- Instrumented run script and saved predictions: `_tmp_qlike_diag/` (build_inspect.py, diagnose.py,
  preds.npz). These are scratch/analysis files, not committed.
