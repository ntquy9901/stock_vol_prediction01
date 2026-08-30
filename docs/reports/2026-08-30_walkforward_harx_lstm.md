# Walk-forward (periodic retrain) vs fixed split: HAR-X vs no-graph LSTM — VN100 h1

## Question
Does periodic retraining (expanding-window walk-forward) change the fixed-split verdict that the
no-graph LSTM is significantly WORSE than HAR-X on QLIKE?

## Headline answer
**Yes — walk-forward changes the verdict.** The fixed split's significant HAR-X advantage disappears
under periodic retraining: pooled over the same 454-date OOS region, the retrained LSTM ensemble is
statistically indistinguishable from HAR-X on QLIKE (date-clustered DM p = 0.37), with the point
estimate marginally favouring the LSTM. The fixed-split HAR-X advantage was largely a single-training
-window artifact, not a structural property of the models.

## Side-by-side
| Setup | LSTM QLIKE | HAR-X QLIKE | DM LSTM-vs-HARX (QLIKE) | Verdict |
|---|---|---|---|---|
| Fixed split (delivered `results/masked_rich_floor1e2/vn100_h1`) | 0.5784 | 0.5115 | p = **1.14e-3**, favours HAR-X (mean_diff +0.0668) | LSTM significantly WORSE |
| **Walk-forward** (7 folds, K=66, expanding, 5-seed ensemble) | **0.4965** | 0.5074 | p = **0.372**, favours LSTM (mean_diff −0.0109) | **no significant difference** |

Same 102-node universe, same 454 OOS dates (46,308 obs), identical QLIKE floor across models. The
walk-forward LSTM ensemble also wins on MSE (2.349e-7 vs 2.364e-7), MAE (2.83e-4 vs 2.87e-4) and R²
(0.229 vs 0.225); HAR (3-feature) pooled QLIKE 0.4983.

Per-seed LSTM QLIKE mean 0.5153 (std 0.0148; seeds 0.498/0.534/0.512/0.531/0.501) — the 5-seed
ensemble (0.4965) improves on the per-seed mean, and the per-seed mean is itself roughly tied with
HAR-X (0.5074). So the honest reading is: **the gap closes to statistical insignificance** (point
estimate flips to the LSTM), not "the LSTM now clearly wins."

## Per-fold forecast QLIKE (seed-ensemble)
| fold | n_forecast | HAR-X | LSTM | HAR | n_train obs | LSTM fit |
|---|---|---|---|---|---|---|
| 0 | 6732 | 0.4906 | 0.5337 | 0.5024 | 297,916 | ok |
| 1 | 6732 | 0.7750 | 0.6955 | 0.7154 | 304,648 | overfit |
| 2 | 6732 | 0.4664 | 0.4515 | 0.4691 | 311,380 | ok |
| 3 | 6732 | 0.4230 | 0.4260 | 0.4249 | 318,112 | ok |
| 4 | 6732 | 0.4470 | 0.4348 | 0.4398 | 324,844 | ok |
| 5 | 6732 | 0.4519 | 0.4479 | 0.4523 | 331,576 | ok |
| 6 | 5916 | 0.4965 | 0.4846 | 0.4825 | 338,308 | ok |

The LSTM matches or beats HAR-X in 5 of 7 folds, and does best on the hard fold 1 (all models spike to
QLIKE ~0.7; the LSTM 0.6955 vs HAR-X 0.7750). Fold 1 is flagged "overfit" for ALL THREE models (HAR,
HAR-X, LSTM) — a fold-region effect (an early-OOS volatility regime the training window fits worse),
not an LSTM-capacity problem; the other 6/7 folds are "ok" for every model.

## Interpretation
- **Periodic retraining/adaptation is a real improvement for the LSTM.** Refitting the LSTM every 66
  days on an expanding window removes the significant deficit it showed when trained once on the fixed
  80/10/10 split. The deep model benefits from seeing more recent data before each forecast block; HAR-X
  (a 5-feature OLS) is already near its low-variance ceiling and gains little from retraining.
- **HAR-X's fixed-split advantage was NOT structural.** Under a fair, repeatedly-retrained protocol the
  two models are within noise on QLIKE (and the LSTM edges ahead on MSE/MAE/R²). The strong fixed-split
  DM (p=1.1e-3) overstated a durable HAR-X superiority that does not survive periodic retraining.
- This does not make the LSTM a clear winner: the DM is not significant (p=0.37) and the per-seed mean
  is tied. The defensible claim is *equivalence under periodic retraining*, reversing the fixed-split
  *significant-loss* claim.

## Method / leakage / caveats
- Expanding-window walk-forward over the delivered fixed-split TEST region (454 dates). Each fold r
  (r = test_start, +66, …): train = `[0, r-h-val)`, val tail = 66 (LSTM early-stop), purge gap h between
  val and forecast, forecast = next 66 days (1-step-ahead, features ≤ t, frozen model). h=1, val=66,
  epochs=16, patience 5, 5 seeds, lookback=10, batch=32.
- **No leakage:** per-ticker feature + target scalers refit TRAIN-ONLY each fold (forecast/val never
  enter the fit — asserted by test); the pooled OOS forecast dates equal the tiled fold blocks; every
  train/val target date strictly precedes every forecast target date (`assert_no_leakage`). HAR-X and
  LSTM use the IDENTICAL QLIKE floor and the same per-node positivity floor `1e-2·mean` within each
  fold (prior H2 floor-mismatch bug avoided).
- **DM caveat:** the date-clustered Diebold-Mariano aggregates each loss to one value per unique OOS
  date (cross-sectional mean) then runs HLN DM at h=1. This handles the cross-sectional dependence and
  the block/temporal structure of the tiled folds, but it is **not a fully HAC-corrected DM** across the
  fold boundaries; treat the p-value as approximate.
- Single market (VN100), single horizon (h1), Parkinson VARIANCE target. Runtime 1243 s on one RTX 4060.

## Artifacts
- `results/walkforward_harx_lstm/walkforward_vn100_h1.json` — pooled + per-seed + per-fold metrics,
  date-clustered DM, per-fold over/under-fit evidence (train/val/test + verdict + learning curves).
- Code / tests / review: `baselines/2026-08-30_walkforward_harx_lstm/`.
