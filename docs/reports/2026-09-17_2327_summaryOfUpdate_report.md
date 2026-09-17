# Summary of update — GBME+DAE (Denoising-Autoencoder representation → GBDT) falsification on HOSE

Date: 2026-09-17. Baseline: `baselines/2026-09-17_dae_gbdt/`. Market: HOSE, all 4 horizons.

## What changed / was built
A new falsification baseline implementing **Hướng 2 — the Kaggle-Grandmaster / Jahrer DAE-representation
paradigm**: an unsupervised denoising autoencoder (swap noise) over the champion OWN-8+EARN features, its
bottleneck `Z ∈ R^16` concatenated to the raw features and fed to the champion gamma-GBM. Compared against
the delivered GBME (GBM+earn) champion over the shared VN walk-forward with date-clustered Diebold-Mariano
and regime-spike robustness. **Frozen-basis** design (one DAE trained on the burn-in window, frozen, embeds
every row) to avoid the per-fold neural-basis drift that detonated the sibling GNN-embed baseline.

## Files (path → purpose)
- `baselines/2026-09-17_dae_gbdt/requirements/requirements.md` → spec, success/kill criteria.
- `baselines/2026-09-17_dae_gbdt/design/design.md` → architecture, data flow, gates, frozen-basis rationale.
- `baselines/2026-09-17_dae_gbdt/code/dae_config.py` → single-source tunable constants (swap rate, K, hidden,
  epochs/patience/lr/wd/batch, val len, seed, horizons, MIN_ROWS, PRED_CAP, GAIN_MIN, DM_ALPHA,
  KILL_HORIZONS, SPIKE_WINDOWS).
- `baselines/2026-09-17_dae_gbdt/code/dae.py` → `DAE` MLP autoencoder, `swap_noise`, `standardize`,
  `val_len`/`val_split`, `train_dae` (batched GPU training + learning curves + early stop), `default_trainer`,
  `frozen_z` (single-basis embedding of every row).
- `baselines/2026-09-17_dae_gbdt/code/run_dae.py` → walk-forward driver GBME vs GBME+DAE, per-horizon JSON,
  DM, per-fold, spike robustness, verdict; reuses `full_matrix`/`gnnhar_sp500._metrics5`/`stats`/
  `overfit_check`.
- `baselines/2026-09-17_dae_gbdt/code_review/code_review_2026-09-17.md` → adversarial review + M-1 fix.
- `baselines/2026-09-17_dae_gbdt/test/{conftest.py,test_dae.py}` → 24 tests (fake-trainer fast driver +
  real-torch smoke).
- `scripts/quality_gate/overfit_check.py` → **shared-module edit (M-1 fix):** added `dae`/`autoencoder` to
  `_LEARNED_PATTERNS` so the pre-push overfit gate recognises DAE results instead of skipping them.
- `scripts/quality_gate/test_overfit_check.py` → asserts the new patterns are detected.
- `results/gamma_gbm/dae_hose_h{1,5,10,22}.json` → results (fit evidence + learning curves + DM + spike).

## Final HOSE result table (QLIKE; 8 folds/horizon; n≈390–399k obs)
| h  | GBME | GBME+DAE | gain% | DM p | beats | fit | gain% ex-spike | DM p ex-spike |
|----|------|----------|-------|------|-------|-----|----------------|---------------|
| 1  | 1.5719 | 1.5802 | −0.53 | 0.0000 | False | ok | −0.67 | 0.0000 |
| 5  | 1.6479 | 1.6508 | −0.18 | 0.0331 | False | ok | −0.12 | 0.0430 |
| 10 | 1.6856 | 1.6888 | −0.19 | 0.3450 | False | ok | +0.03 | 0.6935 |
| 22 | 1.7270 | 1.7356 | −0.49 | 0.2316 | False | ok | −0.57 | 0.2412 |

**Verdict: NO-GO (as expected).** The DAE bottleneck HURTS or ties at every horizon (negative gain), and is
significantly worse at h1 (p<0.001) and h5 (p=0.033); at h10/h22 the degradation is not statistically
significant but still negative. The sign survives spike-window exclusion. Pre-registered success (beat at
BOTH h1 and h5) = **False**.

Notably the DAE genuinely converged (h1 validation reconstruction MSE 0.089 → 0.0093 over 87 epochs,
early-stopped) and every horizon classifies `fit=ok` — so this is a real, non-degenerate representation that
simply adds no incremental OOS forecasting value. This strengthens the paper's thesis: a learned
representation of an 8-feature own-history + earnings set, fed back into the same gamma-GBM, is redundant on
this QLIKE target (consistent with the GNN-embed falsification).

## Tests + coverage
- `python -m pytest baselines/2026-09-17_dae_gbdt/test/ scripts/quality_gate/test_overfit_check.py` →
  **41 passed** (24 baseline + 17 shared).
- Diff-coverage-equivalent (all lines new): **C0 line = 100%, C1 branch = 100%** on `dae.py` (91 stmts /
  14 branch), `run_dae.py` (172 / 30), `dae_config.py` (19) — exceeds the C0=100% / C1≥95% gate.
- Real-data smoke (1 fold h1) and full 4-horizon run both completed; per-fold atomic checkpointing verified.

## Code review
Adversarial review (subagent, 3 lenses). One MAJOR **M-1** (fixed): `GBME+DAE` was not recognised by
`overfit_check.looks_learned`, so the pre-push overfit gate skipped the result JSONs — defeating the "ALL
training result JSON overfit-gated" mandate. Fixed by registering `dae`/`autoencoder` in the shared
`_LEARNED_PATTERNS` (mandate overrides §3.F isolation per CLAUDE.md constitution) + tests for the
auto-detection path. Three MINOR findings dispositioned (accepted/deferred): position-based fake embedder in
the causality test (real path covered elsewhere), loose noise-neutrality bound (matches sibling), embargo
literals in the driver (copied verbatim from the sibling, WARN-only).

## Performance
DAE trained on GPU-resident tensors in 4096-row minibatches with fully-vectorised swap noise (no batch=1, no
per-step host↔device sync in the hot loop); GBM is native-multithreaded `HistGradientBoostingRegressor`. Full
HOSE run (4 horizons × 8 folds, ~1.6M panel rows/horizon) completed in ~17 min on the RTX 4060.

## Data-quality gate
N/A (no data change) — this baseline reads existing enriched HOSE panel + the already-delivered
`hose_earnings_combined.parquet`; it does not crawl, load, or modify any raw/processed data.

## Risks / follow-ups
- Gate: the shared `overfit_check._LEARNED_PATTERNS` edit is additive and verified collision-free; the four
  delivered result JSONs are now gate-CHECKED (not skipped) and all pass (`fit=ok`).
- No git operations performed (coordinator handles git). Smoke output files were removed; only the four
  `dae_hose_h*.json` results remain.

## DoD checklist
- [x] 5 baseline subfolders present (requirements/design/code/code_review/test)
- [x] Tests pass (41) with pytest; C0=100%/C1=100% on changed lines
- [x] Adversarial code review run; MAJOR fixed, minors dispositioned
- [x] Real run completed (all 4 horizons, 8 folds each), fit evidence + learning curves present
- [x] Performance (GPU batching) satisfied; summary recorded
- [x] No git commit/push (per instruction)
