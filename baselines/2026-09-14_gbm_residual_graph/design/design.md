# Design — GBM residual-graph refiner

## Architecture (data flow)
```
frames ─► FM.panel(h) ─► per fold k:
    train[<k] ──► FM.gbm(own-8, seed-avg) ──► gbm_pred(k)        (OOS level, log-variance residual base)
    train[<k] ──► S1.build_graph (top-10 corr) ──► Wc(k)
    fold rows ──► S1.graph_feats(Wc) ──► g_nb_*(k)               (causal neighbour features)
    resid(k) = log(y) − log(gbm_pred)                            (target, known only in hindsight)
expanding residual stack (causal):
    ridge_k = Ridge(alpha).fit( standardise(g_nb_* of folds<k), resid of folds<k )
    resid_hat(k) = ridge_k.predict( standardise(g_nb_*(k)) )     (fold 0: resid_hat = 0)
    final(k) = gbm_pred(k) · exp(resid_hat(k))                   (positive by construction)
pool folds ─► QLIKE(gbm_pred), QLIKE(final) ─► DM(final vs gbm_pred) ─► verdict
```

## Files
- `code/config.py` — single-source tunables: `RIDGE_ALPHA`, `NB_FEATS` (neighbour feature subset of
  `S1.GRAPH`), `MIN_STACK_ROWS` (min earlier-fold rows before the refiner activates). Reuses
  `paper_models.config` HORIZONS/MIN_ROWS + `S1.FOLDS`/`FM.SEEDS`/`FM.FL` (no duplication).
- `code/residual_graph.py` — pure functions + CLI: `fold_stream()` (per-fold gbm_pred + features + resid),
  `refine()` (expanding causal ridge stack → **two** finals: `raw` unclipped + `clip` bounded), `run()` (loop
  horizons, per-horizon checkpoint), `_verdict()`, `main()` (`python residual_graph.py [hose|sp500]`).
- `test/` — TDD: positivity (both variants), causality (no future-fold leakage), signal-recovery,
  noise-neutrality, run structure, verdict logic, checkpoint.

## Key design decisions
1. **Residual on log-variance** (not raw variance): additive-in-log reconstruction keeps `final > 0` without a
   clamp hack, and matches the reference's positivity concern (variance-scale QLIKE is rank-sensitive).
2. **Expanding OOS residual stack, NOT in-sample GBM residuals.** GBM overfits train → in-sample residual ≈ 0
   and would not generalise. Training the ridge on EARLIER folds' OOS residuals is both honest (no leakage) and
   representative of test-time residuals. Fold 0 has no history → refiner is identity.
3. **Linear refiner first (Ridge).** Minimises overfit surface; the prior graph failure blew up at h10 with a
   tree on 7 collinear features. If ridge is DM-positive, escalate to a 1-layer GNN (separate baseline).
4. **Two variants reported (`raw` + `clip`).** The raw (unclipped) log-variance residual reconstruction
   `gbm·exp(resid_hat)` amplifies HOSE floor/limit-lock residuals through `exp(.)` and is QLIKE-catastrophic —
   an instance of the reference's warning that QLIKE ranking is scale-sensitive (log-MSE fit ≠ variance-QLIKE).
   The `clip` variant winsorises the residual target and clips the prediction to ±`RESID_CLIP`, bounding the
   multiplicative adjustment to [0.61×, 1.65×]; it is the FAIR best case. The GO verdict is judged on `clip`
   (if even the bounded refiner cannot beat GBM under DM, graph carries no orthogonal signal).
4. **Reuse shared infra** (`FM.gbm`, `S1.build_graph`, `S1.graph_feats`, `M.per_obs_qlike`,
   `ST.date_clustered_dm`) — anti-abstraction gate: no reimplementation.

## Gates
- **Simplicity Gate:** one module of pure functions + a thin driver; no new abstraction. Pass.
- **Anti-Abstraction Gate:** uses sklearn Ridge + existing project helpers directly. Pass.
- **Performance/Batching Gate:** GBM is the existing seed-averaged batched fit; the ridge is a single vectorised
  `.fit`/`.predict` over pooled arrays (no per-item Python loop in the hot path). Folds loop is inherent to
  walk-forward (sequential by time, not a batchable axis). Pass.

## Leakage controls
- GBM, graph adjacency, feature standardiser, and ridge all fit on train/earlier-fold rows only.
- Neighbour features are day-t cross-sections (no h-ahead target). Residual target is used only as a training
  label for folds strictly earlier than the predicted fold. A causality test asserts fold-k output is invariant
  to permuting later folds' data.
- Shared positivity floor `FM.FL` identical to the GBM baseline for QLIKE.

## Overfit evidence (gate-required)
`fit_diagnostics` per horizon: ridge train-resid MSE vs test-resid MSE + verdict; the JSON records
`train`/`test` QLIKE for both gbm and final so the pre-push overfit-evidence gate can classify the refiner.
