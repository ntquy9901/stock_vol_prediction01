# Code review — GBME+DAE baseline (2026-09-17)

Adversarial review (Blind Hunter + Edge Case Hunter + Acceptance Auditor lenses), run against the sibling
template `2026-09-14_gbm_gnn_embed` and the reused shared code (`full_matrix`, `stats`, `overfit_check`). One
MAJOR finding (fixed), no CRITICAL, several MINOR (dispositioned).

## MAJOR

### M-1 (FIXED) — learned-model name not recognised by the pre-push overfit gate → result JSONs silently skipped
`run_dae.LEARNED = "GBME+DAE"` was NOT matched by `overfit_check.looks_learned` (its `_LEARNED_PATTERNS`
had no token for autoencoders). Consequently `check_overfit_evidence._is_masked_rich_result` returned False
for the DAE result JSONs and the pre-push over/under-fit gate **skipped** them entirely — a real learned model
could ship without the mandated fit-evidence check (CLAUDE.md: "Gate chặn (pre-push) — MỌI training result
JSON … overfit/underfit → BLOCK"). The in-baseline tests hid this by passing `learned=(R.LEARNED,)`
explicitly, never exercising the auto-detection path the real gate uses.

**Fix (mandate overrides §3.F isolation — CLAUDE.md constitution wins on conflict):** registered the new
learned-model family in the shared enforcement registry — added `"dae"`, `"autoencoder"` to
`scripts/quality_gate/overfit_check.py::_LEARNED_PATTERNS` (additive, analogous to how xgb/catboost/lightgbm
were added; verified no existing result JSON has a model key containing `dae`/`autoencoder`, so nothing else
is newly flagged). Added a shared-module test (`test_overfit_check.py`: `GBME+DAE`, `StackAutoencoder`
detected) and a baseline test asserting the AUTO-DETECTION path (`OF.looks_learned(R.LEARNED)` and
`check_result_evidence(doc)` with NO explicit `learned=`). The four delivered horizons all classify
`fit=ok`, so the now-active gate PASSES them (validated, not skipped).

## MINOR (dispositioned as accepted / follow-up)

- **m-1 — outer-fold causality test uses a position-based fake embedder.** `test_outer_fold_causality`
  injects a fake trainer whose `z` depends only on row position, so it verifies GBM fold-slicing but cannot
  itself catch DAE-mediated future leakage. Mitigated: the leakage-relevant piece (burn-in-only standardiser)
  is separately unit-tested (`test_standardize_burnin_stats`), the frozen single-basis causality is asserted
  in `test_frozen_z_single_basis_covers_all_rows` (DAE trains on burn-in dates only), and the REAL torch DAE
  path is exercised by `test_run_real_dae_smoke`. No live bug; accepted.
- **m-2 — `test_z_plumbing_noise_neutrality` bound (×1.10) is loose.** Matches the accepted sibling template
  threshold; tightening risks flakiness. Accepted as-is.
- **m-3 — embargo `int(h*1.6)+5` literals live in the driver, not `dae_config`.** Copied verbatim from
  `full_matrix.main` and the sibling driver (single-sourced there by convention, consistent across the
  comparison table); WARN-only under the config-hardcode scanner. Left to match the sibling.

## Verified-correct (scrutiny items that check out)
- **Causality / no leakage:** DAE is unsupervised (reconstructs features, never sees `y`); standardiser +
  weights fit on burn-in rows only (first eligible fold's causal `trf`, strictly before every test window);
  frozen extractor embedding later rows is legitimately out-of-sample. GBM fits on `trf_e` (train minus the
  last `VALID_LEN` dates), so `val_metrics` is a TRUE hold-out.
- **Frozen-basis zmap alignment:** `FM.panel` ends with `reset_index(drop=True)` → unique RangeIndex; `a[keep]`
  preserves it; `frozen_z` keys the map by that index; `trf.index`/`tef.index` are subsets — no collision. One
  DAE per horizon, embeds all rows, reused across all 8 folds (single shared basis, no per-fold refit).
- **Numerical guards:** `swap_noise` `torch.gather(x, 0, src)` is same-column (dim-0) with fraction ≈ rate;
  zero-variance columns guarded (std→1); `_gbm_clip` applies identical `[FL, PRED_CAP]` to BOTH models so the
  guard cannot bias DM; `_safe_dm` guards identical losses (`np.allclose`) and `h ≥ n_dates` (DM raises →
  caught).
- **GPU batching:** DAE trains in `C.BATCH=4096` minibatches over a GPU-resident tensor with fully-vectorised
  swap noise; only per-epoch `.item()` syncs; full-batch embed. No batch=1, no per-step host↔device sync in
  the hot loop.
- **Config hygiene:** all DAE tunables live in `dae_config.py` with inline comments; GBM hyper-params + FL
  reused from `full_matrix` (single source, not duplicated).

## Verdict
Pipeline causally sound; MAJOR M-1 fixed with tests; minors accepted/deferred. Baseline is delivery-ready.
