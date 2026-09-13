# Design — Hybrid feature-concat GBM (own+earn ⊕ graph-spillover)

Date: 2026-09-13. Plan phase of CLAUDE.md §5 SDD. Spec: `../requirements/requirements.md`.

## 1. Architecture (single gamma-GBM, feature concatenation)
The project's tabular paradigm: one `HistGradientBoostingRegressor(loss="gamma")` per model, seed-averaged
over `FM.SEEDS`, fed a concatenated feature vector. No neural branches; the "hybrid" is a concatenation of
two feature blocks into one GBM:

```
feature vector = [ own8 block ] ++ [ FM.EARN block ] ++ [ graph-spillover block ]
                  own dynamics    earnings proximity   cross-stock propagation
```

Models share the same walk-forward panel and floor, so their pooled per-obs QLIKE errors are aligned and
DM is fair.

## 2. Data flow
```
FM.load(hose)                       -> per-ticker frames (own8, parkinson_variance, daily_return,
                                       volume_zscore_22, market_pk, logpk, mr_*)  [read-only reuse]
inject hose_earnings_combined.parquet -> edates {ticker: sorted announce dates}
for h in HORIZONS:
  a = FM.panel(frames, edates, h)    -> pooled panel with y=variance.shift(-h) + FM.EARN columns
  for fold k in S1.FOLDS:
    tr / te split, embargo int(h*1.6)+5 d, gate len(tr) >= min_rows
    Wc = S1.build_graph(tr, tickers, rng(S1.RNG_SEED+k))   # TRAIN-ONLY correlation top-k graph
    fold = a[TRAIN_START .. tend]
    fold = spillover_features.add_spillover_features(fold, tickers, Wc)   # + g_corr + 7 S1.GRAPH feats
    trf/tef = fold split
    preds[model] = mean_s FM.gbm(trf, tef, cols[model], s)   # seed ensemble
    preds[graph_only] = mean_s FM.gbm(trf, tef, SPILL_COLS, s)
  pool y, dates, preds -> per-obs QLIKE -> mean QLIKE, DM, residual err_corr, fit diagnostics
write results/gamma_gbm/gbm_earn_graph_<market>.json
```

## 3. Spillover-feature construction (`code/spillover_features.py`)
Thin, read-only wrapper over shared machinery (anti-abstraction gate: reuse, do not reimplement):
- `SPILL_COLS = list(S1.GRAPH)` — the 7 neighbour-aggregation features.
- `add_spillover_features(fold, tickers, Wc)`:
  1. `fold["g_corr"] = FM.nb(fold, tickers, Wc)` — single-feature neighbour-mean volatility.
  2. `gf = S1.graph_feats(fold, tickers, Wc, "")` — the 7-feature block (index-aligned to `fold`).
  3. Attach the 7 columns; `fillna(0.0)` on the 7 + `g_corr` (isolated tickers / empty neighbourhoods).
  4. Return the augmented copy (never mutate the caller's frame in place → no hidden state).

### Which spillover features, and why they are richer than `g_corr`
`g_corr` is a single neighbour-mean of `parkinson_variance`. The 7-feature `S1.GRAPH` block additionally
carries: neighbour shock (`mr_change`), neighbour max / dispersion (tail + heterogeneity of the
neighbourhood), node-minus-neighbour (own vs neighbourhood level), neighbour return, neighbour
volume-shock. This is a genuinely richer cross-sectional description, so if spillover ever helps beyond a
single mean, this block is where it would show.

## 4. Causality / leakage argument (proven by tests)
1. **Graph from train only.** `Wc = S1.build_graph(tr, ...)` where `tr` = rows with `date < ts - embargo`.
   The adjacency never sees test-window correlations.
2. **Aggregation uses only the date-t cross-section.** For row (date=t, ticker=i), every S1.GRAPH feature
   is `sum_j W[i,j] * value_j(t)` where `value_j(t)` is a **contemporaneous** (day-t) quantity
   (parkinson_variance, mr_change, daily_return, volume_zscore_22). No future rows enter. The target
   `y = parkinson_variance.shift(-h)` (day t+h) is separate. Identical causal footing to the existing
   `g_corr` / `FM.nb`, which the paper already uses.
3. **Future-invariance.** Perturbing rows with `date > t` does not change any feature at date `t`
   (aggregation is per-date). Enforced by `test_spillover_causal_future_invariant`.
4. **Embargo.** `int(h*1.6)+5` trading days between train end and test start removes the h-overlap of the
   forward-shifted target.

## 5. Error-correlation diagnostic
`err_corr` = Pearson correlation of pooled residuals `(pred - y)` between `GBM+earn` and `graph_only`.
Rationale (blend-diversity): if two forecasters' residuals correlate ~1, a blend adds nothing; the prior
`GNN ⊕ GBM+earn` failure had ~0.98. `graph_only` (GBM on the 7 spillover features alone, no own-AR) is the
cleanest standalone "graph branch". Reported per horizon, no spin.

## 6. Gates (CLAUDE.md §5)
- **Simplicity Gate:** one module (`spillover_features.py`) + one driver (`run_hybrid.py`) + `config.py`.
  No new abstractions; reuses `FM`/`S1` verbatim.
- **Anti-Abstraction Gate:** calls `S1.build_graph`/`S1.graph_feats`/`FM.nb`/`FM.gbm`/`FM.panel` directly.
- **Performance/Batching Gate:** GBM is vectorised (sklearn, all rows per fit); neighbour aggregation is a
  single `V @ W.T` matmul per fold (already batched in `S1.graph_feats`). No per-item Python hot loop over
  observations. Seed loop = 3 fits (ensemble, unavoidable) — acceptable. CPU-bound sklearn GBM; no GPU path
  in the shared `FM.gbm` (out of scope to change a shared module).
- **No silent degradation:** `run_hybrid` raises if no fold is ever scored (empty result) rather than
  writing an empty JSON; `fillna(0.0)` only on graph columns for genuinely isolated tickers (bounded,
  documented) — matching the existing harness behaviour.
- **Config single-source:** all tunables (`HORIZONS`, `MIN_ROWS`, success thresholds, model/feature names)
  live in `code/config.py`; no magic numbers in the driver.

## 7. Files
- `code/config.py` — tunable constants (single source of truth).
- `code/spillover_features.py` — causal spillover-block builder (read-only reuse of `FM`/`S1`).
- `code/run_hybrid.py` — walk-forward driver; `run_hybrid(market, load_fn=None)` returns the result dict;
  `main()` writes JSON.
- `code/__init__.py` — package marker.
- `test/conftest.py` — sys.path bootstrap (dashes in folder name block `python -m`).
- `test/__init__.py`
- `test/test_spillover_features.py` — causality (future-invariance), formula/aggregation, fillna, no-mutation.
- `test/test_run_hybrid.py` — end-to-end smoke on stub loader + patched folds, empty-fold guard,
  default-loader branch, `_success` verdict, err_corr computation.

## 8. Complexity tracking
No gate broken.
