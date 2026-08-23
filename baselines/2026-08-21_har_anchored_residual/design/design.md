# Design / Plan — HAR-Anchored LSTM–GAT Residual Study

> **SUPERSEDED (2026-08-23).** Describes an earlier residual-experts / blend / gate / cross-fitting design.
> The delivered critical-path code is the masked-rich HAR / HAR-X / LSTM / weighted-GAT comparison
> (`code/run_masked_rich.py`, `MaskedRichNet`); see `docs/paper/soict_harlstmgat*.tex`. Kept for lineage only.


Plan for `requirements/requirements.md`. Passes Simplicity + Anti-Abstraction + Performance/Batching gates.

## Architecture / data flow
```
processed CSV (date, parkinson_volatility)
   -> har_features (reuse data_utils)            # [N,3] daily/weekly/monthly, cutoff at t
   -> make_windows (reuse data_utils)            # valid anchors t
   -> purged_split (NEW, folds.py)               # 80/10/10 + purge h anchors at each boundary
   -> per-ticker TickerScaler (reuse)            # fit TRAIN only
   -> pooled tensors {train,val,test} + raw HAR features + per-ticker series
Experts:
   HAR (reuse baselines.har_fit/predict)         # frozen anchor, per-horizon pooled OLS on TRAIN
   cross-fitted HAR preds on TRAIN (NEW)         # expanding-window, for residual targets
   LSTM (reuse run_lstm.LSTM)                     # ticker-local temporal, E1/E5
   GAT branch (reuse model.HARLSTMGAT use_graph) # cross-sectional, E2/E6/E7
Hybrids:
   E3/E4 convex blend: freeze HAR+NN, fit alpha on VAL (closed-form MSE / grid QLIKE)
   E5/E6/E7 additive: y_hat = har + lambda * f_NN(residual); zero-init residual head
   E8 multiplicative: y_hat = (har+eps) * exp(lambda * delta_NN); positive by construction
   E9 static gate: lambda_h = sigmoid(b_h)
   E10 dynamic gate: lambda = sigmoid(g_h(z_{i,t})); z = observable state at t; HAR-biased init
Eval: metrics (reuse) + DM (reuse) + block-bootstrap CI + MCS + date-clustered (NEW, stats.py)
Export: row-aligned predictions parquet/csv per fold/model/seed (NEW, io_preds.py)
```

## File list (all under `code/`)
- `config.py` — central `ExpConfig` (dates via fracs, horizons, seeds, graph, loss, hyperparams, eps, paths).
- `folds.py` — `purged_split(anchors, horizon, train_frac, val_frac)`; reuses data_utils primitives.
- `har_cv.py` — `crossfit_har(feats, pk, anchors_train, horizon, n_folds)` -> OOS HAR preds on train rows.
- `experts.py` — thin wrappers: build pooled tensors (via folds), LSTM expert, GAT expert (import model.py).
- `residual.py` — additive (E5-E7) + multiplicative (E8) heads with zero-init; HAR-fallback invariant.
- `blend.py` — E3/E4 frozen-expert alpha fit (closed-form MSE, QLIKE grid) on validation predictions.
- `gate.py` — E9 static lambda_h, E10 dynamic soft gate (Linear / MLP-8), observable-state feature builder.
- `stats.py` — block-bootstrap loss-diff CI, MCS, date-aggregated DM.
- `io_preds.py` — row-aligned prediction export (parquet) + feature-availability manifest CSV.
- `run_experiment.py` — CLI `--experiment E{0..10} --horizon {1,5,10,22} --seed S --dataset vn30 [--smoke]`.

## Reuse (read-only imports; hard isolation per §3.F)
From `submission/soict_lstm_gat/`: `data_utils` (har_features, make_windows, TickerScaler),
`baselines` (har_fit, har_predict, garch_forecast), `edges` (glasso_adjacency), `metrics` (all + DM),
`model` (HARLSTMGAT / GATLayer), `run_lstm.LSTM`. Bootstrap sys.path to that dir (folder name has dashes).

## Key design decisions
1. **Purge = h anchors** at train/val and val/test. Property: `max(train_anchor)+h < min(val_anchor)` and
   likewise val/test. Fixes leakage F1.
2. **Cross-fitted residuals** via expanding-window inner folds on the TRAIN anchors only: split train into
   K time-ordered blocks; for block k, fit HAR on blocks <k, predict block k. Residual = y − OOS_HAR.
   Deployment-realistic; never uses in-sample HAR residuals.
3. **Zero-init residual head** so at init the hybrid == HAR exactly (unit-tested invariant). E8 uses
   `exp(0)=1` multiplicative fallback; E5-E7 additive fallback `+0`.
4. **E3/E4 frozen experts**: train HAR + NN independently, freeze, then fit α ONLY on validation preds.
   MSE closed-form `alpha* = clip_{[0,1]}( sum (y-nn)(har-nn) / sum (har-nn)^2 )`; QLIKE dense grid over [0,1].
5. **E10 gate inputs observable at t only**: current market PK, vol-of-vol, cross-sectional dispersion,
   |market return proxy|, rolling avg correlation/graph density, HAR level, |HAR−NN| disagreement. No future
   target/regime. Soft routing first; HAR-biased bias init (lambda≈0.05–0.15).
6. **Deep branch reuses model.py**; the GAT-hurts finding is a hypothesis to re-test under residual framing
   (E6 vs E5), not a foregone conclusion — but graph counts as useful only if it beats E5 (no-graph residual).

## Performance / batching gate
Pooled batched tensors `[B, seq, 3]` (LSTM) and `[B, N, seq, 3]` snapshots (GAT), GPU, `batch_size` from
config, `non_blocking` H2D, val batched. No per-item main-thread loop in hot path. Cross-fitting is CPU OLS
(cheap). Matches the existing run_lstm/run_all performance profile.

## Task list (verifiable)
1. `folds.purged_split` + `test_folds` (purge property) — verify: pytest pass.
2. `har_cv.crossfit_har` + test (OOS shape, no-leak: fold k HAR unaffected by block k) — verify: pytest.
3. `io_preds` export + manifest + test (round-trip parquet) — verify: pytest.
4. `blend` E3/E4 + test (alpha in [0,1]; MSE closed-form matches grid within tol) — verify: pytest.
5. `residual` E5-E8 + test (HAR-fallback-at-init invariant; E8 positivity) — verify: pytest.
6. `gate` E9/E10 + test (positivity; gate uses only provided state; HAR-biased init small lambda) — verify.
7. `stats` block-bootstrap + MCS + test (CI covers known diff; MCS keeps best) — verify: pytest.
8. `run_experiment` CLI + smoke test (E0 + E1 on tiny synthetic pass) — verify: pytest -m smoke.
9. Staged full runs + `reports/experiment_results.md` + H1–H6 decision — verify: decision table complete.

## Complexity tracking
No gate broken. One new folder, thin wrappers over existing tested code; no re-implementation of HAR/LSTM/GAT.
