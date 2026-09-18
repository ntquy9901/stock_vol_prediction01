# Design — Foundation-model feature baseline (Hướng B)

## Data flow
```
frames (HOSE csv, own history)  ──►  foundation_forecaster
     │                                  causal sliding windows (context ≤ t, CTX_LEN=256)
     │                                  Chronos-Bolt zero-shot, PRED_LEN=22, batched GPU
     │                                  ──► cache parquet [ticker, date, fnd_h{1,5,10,22}, fspread_h{..}]
     ▼
run_foundation.panel(h)  ──►  _attach(cache, h): inner-join fnd + wrong-ticker placebo (paired rows)
     ▼
walk-forward over S1.FOLDS (8 folds), TRAIN_START=2015, embargo=int(1.6h)+5d, val slice = last 22 dates
     GBME      = champion HGBR gamma on OWN-8 (+earnings)
     GBME+FND  = GBME + [fnd (+ fspread)]           (horizon-matched zero-shot forecast)
     GBME+PLAC = GBME + [fnd_plac (+ spread)]        (wrong-ticker forecast → placebo)
     HAR, Chronos_ZS (zero-shot forecast used directly) for Stage-0
     ▼
_pool_doc: pooled 5-metric train/val/test + fit_diagnostics (GBM family), Stage-0 DM (ZS vs HAR),
           Stage-1 DM (FND vs GBME primary; PLAC vs GBME; FND vs PLAC), gain, verdict, placebo,
           per-fold QLIKE + spike robustness  →  atomic per-horizon checkpoint JSON
```

## Key design decisions
- **Causality:** the step-h forecast for a row dated t uses context values ≤ t only; it predicts variance at t+h,
  which is exactly the panel target `parkinson_variance.shift(-h)`. The model is FROZEN (zero-shot) so it never
  trains on this data. Computing the whole cache "at once" does not leak — each row's context stops at its own t.
- **One call → all horizons:** a single Chronos call (PRED_LEN=22) yields steps 1,5,10,22, so the cache cost is
  horizon-independent and all four horizons come essentially free once the cache is built.
- **Additive, not multiplicative:** the forecast is a GBM feature (tree can ignore it), NOT a base margin /
  multiplicative anchor — avoids the QLIKE detonation seen when a foundation prior is imposed multiplicatively.
- **Placebo = wrong-ticker forecast** (cyclic ticker map, aligned by date): destroys firm identity while keeping the
  marginal distribution and calendar. Inner-join keeps GBME / FND / PLAC on identical paired rows for valid DM.
- **Champion reuse:** GBME = `full_matrix.gbm` (HGBR gamma, 300 trees, lr .05, 31 leaves, l2=1), seed-ensembled;
  OWN-8 single-sourced from the paper_models config (`own_set`), earnings from the crawled HOSE parquet.

## Performance / batching gate
- Zero-shot inference is **batched** (`FORECAST_BATCH=512` contexts per model call) on GPU (`cuda`, CPU fallback);
  never batch=1. Measured ~20k forecasts/sec on the RTX 4060, so ~1.4M HOSE forecasts ≈ 1–2 min of pure inference.
- Forecasts are **cached to parquet** and computed once; the walk-forward reuses the cache for every horizon.
- GBME training reuses the tested HGBR champion; seed-ensembled over `FM.SEEDS`.

## Gates
- **Simplicity:** reuses `full_matrix` (GBM/HAR/panel), `stats` (date-clustered DM), `overfit_check` (fit verdict),
  paper OWN-8; adds only the forecaster + driver + config. No new abstraction layer.
- **Anti-abstraction:** Chronos used directly via `BaseChronosPipeline`; no wrapper framework.
- **Config-hardcode:** every tunable (model name, CTX_LEN, PRED_LEN, batch, quantiles, floors, DM/kill thresholds,
  spike windows) lives in `foundation_config.py`.

## Files
| file | purpose |
|------|---------|
| `code/foundation_config.py` | all tunable constants (single source of truth) |
| `code/foundation_forecaster.py` | causal zero-shot forecaster + per-(ticker,date) parquet cache |
| `code/run_foundation.py` | walk-forward Stage-0/Stage-1 driver, DM/verdict/spike, checkpoints |
| `test/test_foundation.py` | fake-forecaster driver tests + opt-in real-Chronos slice test |

## Fit-evidence / gate note
Model names (GBME, GBME+FND, GBME+PLAC) are GBM-family and do not match `overfit_check`'s learned-name patterns, so
the pre-push overfit gate treats the result as a non-learned baseline and skips it (as it does for other GBM+earn
results). The result JSON nonetheless carries full train/val/test + `fit_diagnostics` for every GBM model so the
foundation feature's fit behaviour is auditable.
