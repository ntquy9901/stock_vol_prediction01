# Design — regime / change-point features

## Regime features (`regime_features.py`, causal)
From the market-level daily volatility series (mean of the panel's `market_pk` column per day), compute three
causal features over a trailing window (default `pc.HAR_MONTHLY_WINDOW`=22):
- `vol_z`: trailing z-score of market vol (how extreme today is).
- `regime`: 1 if the trailing-mean vol exceeds its EXPANDING median (only past values), else 0.
- `dsc`: days since the regime last flipped.
No PELT/HMM dependency: a rolling threshold is reproducible and auditable, and the research's "PELT accurately
detects change points" claim did not survive adversarial verification. Look-ahead-freeness is unit-tested
(prefix-invariance: features for days 0..k are unchanged when future days are appended).

## Experiment (`run_regime_har.py`)
Probe the hypothesis on the QLIKE champion (HAR-X) without deep-model surgery. Per walk-forward fold (reusing
`pack_fold` read-only): fit an OLS on the 5 HAR-X features and on HAR-X + the 3 regime features (broadcast the
market-wide regime to every node), score both on the pooled test set, and compare with a date-clustered
Diebold-Mariano test. QLIKE floor identical to the main experiment.

### Gates
- Simplicity: OLS + three rolling features; no new heavy dependency. Passes.
- Anti-abstraction: reuse delivered panel/fold/metrics directly. Passes.
- Performance/batching: OLS on flattened arrays (vectorised lstsq); no per-item Python loop over rows. Passes.

## Integration plan (only if GO)
If regime features help HAR-X, add them as node features to the LSTM/VolGA input (in_dim 5→8): extend the panel
feature stack and the model's first layer, then retrain. This changes the model architecture and the enriched
panel, so it is gated behind a positive HAR-X result rather than done speculatively.

## Files
- `code/regime_features.py` — causal feature computation (+ tests).
- `code/run_regime_har.py` — HAR-X vs HAR-X+regime walk-forward OLS comparison (+ tests for OLS/design).
- Results: `results/regime_features/regime_<market>_h<h>.json`.
