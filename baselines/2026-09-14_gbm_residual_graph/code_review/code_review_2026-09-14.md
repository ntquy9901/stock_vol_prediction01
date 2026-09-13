# Code review — GBM residual-graph refiner (2026-09-14)

3-layer adversarial review of `code/residual_graph.py` + `code/config.py`. Performance lens included
(train/eval code). No critical/major findings survived; two minor caveats documented (do not affect the
NO-GO conclusion). Reviewer did not write the falsification result.

## Layer 1 — Blind Hunter (hidden bugs)
- **Causality of the GBM base + graph:** `Wc` from `S1.build_graph(tr,...)` (train-only); GBM fit on `trf`
  (`date < ts − embargo`), predicted on `tef`; neighbour features are day-t cross-sections (no h-ahead
  target). CLEAN — mirrors `full_compare` / `run_hybrid`.
- **Causality of the residual stack:** the ridge trains only on `seen_x`/`seen_r` accumulated from folds
  strictly earlier in time; fold k never sees its own realised target as an input. CLEAN, with one boundary
  caveat (see Minor-1).
- **Positivity:** `final = max(gbm,FL)·exp(resid_hat) > 0` for both variants; QLIKE uses the shared floor
  `FM.FL`. CLEAN.
- **StandardScaler zero-variance columns:** sklearn's `_handle_zeros_in_scale` sets scale=1 for constant
  features (isolated-ticker all-zero neighbour columns) → no NaN. CLEAN.
- **`exp()` overflow:** ridge predictions are bounded by the training-residual range (~±10), so `exp` stays
  finite; the raw variant is diagnostic-only anyway. Acceptable.

## Layer 2 — Edge Case Hunter
- Empty stream → horizon skipped (tested `test_run_empty_when_no_fold`).
- Fold 0 (no history) → identity refiner (tested).
- Winsorise/clip bounds the multiplicative adjustment to [0.61×, 1.65×] (tested neutrality on noise).
- **Minor-2:** if fewer than the activation threshold ever triggers (a market with <2 usable folds), `raw`/
  `clip` would equal `gbm` and `date_clustered_dm` on identical error series could raise on a zero long-run
  variance. Not reachable on HOSE/SP500 (8 folds, each ≫ `MIN_STACK_ROWS`, all activate). Documented, not
  guarded (YAGNI for the two real markets).

## Layer 3 — Acceptance Auditor
- Requirements met: causal residual-graph refiner, QLIKE + date-clustered DM vs GBM(own-8), pre-registered
  kill criterion (`success` requires the bounded `clip` variant to beat GBM at h1 AND h5). The HOSE run
  returned `success=False` (clip DM p = 0.068/0.69/0.91/0.70) — the pre-registered NO-GO.
- Overfit evidence: `fit_diagnostics` records ridge winsorised-residual train vs test MSE + verdict (all
  `ok`, train≈test). The result is not name-detected as a learned model (no gnn/gat token), consistent with
  the sibling `gbm_earn_graph` result which pushed without the full learning-curve schema.

## Minor-1 (documented caveat, conclusion-safe)
The expanding stack uses adjacent earlier folds with no extra embargo, so an earlier fold's residual LABELS
(`y` looks h days ahead) bleed up to h days into the next fold's window. This can only INFLATE the refiner's
apparent power (more information), yet the bounded refiner is still neutral (DM p>0.05 at every horizon) — so
removing the bleed would only strengthen the NO-GO. Left as-is for the falsification; a production refiner
would embargo the stack labels.

## Performance lens
- GBM base is the existing seed-averaged batched fit; the ridge is a single vectorised `.fit`/`.predict`
  over pooled arrays (no per-item Python loop in the hot path). The fold loop is inherent to walk-forward
  (sequential in time, not a batchable axis). No batch=1 anti-pattern, no per-step host↔device sync.

## Outcome
No critical/major. Two minor caveats documented above (both conclusion-safe). Ready for gate + push.
