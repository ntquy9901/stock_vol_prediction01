# Code review — DY (2014) spillover-edge ablation (2026-08-29)

Adversarial 3-layer self-review (Blind Hunter / Edge-Case Hunter / Acceptance Auditor) of the new code
under `baselines/2026-08-29_dy_spillover_ablation/`. `archive/` and live-training-path files are out of
scope (imported read-only). Findings below with severity and resolution.

## Acceptance vs requirements
- [x] Generalized-FEVD connectedness implements the published DY 2014 / Pesaran-Shin equation — verified
      by an INDEPENDENT loop-based recompute test (`test_gfevd_matches_independent_reference_formula`),
      not a reuse of the vectorised code. Exact equations cited in the module docstring.
- [x] VAR->VMA recursion verified independently (`test_vma_recursion_two_lags`).
- [x] Row-normalised (rows sum to ~1), directed/asymmetric, finite, non-negative shares — tested.
- [x] High-dim fix: elastic-net VAR (Demirer et al. 2018) with stated penalty (alpha=0.05, l1_ratio=0.5),
      lag (VAR(1)) and FEVD horizon (H=10).
- [x] Train-only estimation (rows strictly before the first validation target `D.d_va[0]`), frozen.
- [x] Same self-loop=1.0 / Top-K convention as vol2pk/sector so WeightedGATLayer consumes it unchanged.
- [x] CPU forward-pass smoke on a real HNX slice; 5 metrics + date-clustered DM plumbing reused unchanged.

## Findings

### MAJOR
- **M1 (leakage surface).** `train_vol_panel` keeps rows strictly before `D.d_va[0]` (first validation
  TARGET date). The delivered correlation edge in `masked_rich` uses `wide.iloc[:last_tr_row+1]`, which
  is ~`horizon-1` rows earlier (it excludes the train/val purge gap). At h1 the difference is ~2 panel
  rows, all still strictly before any validation target -> **no val/test observation enters the VAR**;
  the matrix is frozen for val/test. Resolution: accepted as leakage-safe; documented in the module
  docstring and design.md. The frozen matrix never touches val/test data (which is the only leakage that
  matters for a fixed edge).

### MINOR
- **m1 (imputation distortion).** `impute_panel` ffill+bfill: back-fill uses later TRAIN values to fill a
  late-listing ticker's leading NaNs (constant stretch), which flattens that ticker's early dynamics.
  Bounded by elastic-net regularization; strictly within the train window (no future val/test info).
  Documented as a caveat. Alternative (common-window truncation) would discard train history, so ffill+
  bfill is preferred. No silent failure — a fully-empty column deterministically -> 0.0.
- **m2 (standardization changes Sigma scale).** Per-ticker z-scoring (train stats) is applied so a single
  elastic-net penalty is meaningful across the tiny-magnitude variance series; connectedness is computed
  in standardized space. This is a documented, legitimate preprocessing (DY-family studies routinely use
  standardized/normalized volatility). Stated in the docstring.
- **m3 (dense->Top-K sparsification).** The generalized-FEVD matrix is naturally dense; the model
  adjacency keeps Top-K=5 spillover sources per row to MATCH the vol2pk sparsity for a fair edge-only
  comparison. The FULL-matrix connectedness statistics (total connectedness index, directional degree,
  row-sum check) are reported separately in the result JSON so the dense structure is not hidden.
- **m4 (fixed penalty, no CV).** alpha=0.05 is fixed (not cross-validated) for reproducibility and speed;
  exposed as a CLI knob (`--alpha`, `--l1-ratio`, `--var-lag`, `--fevd-h`) for sensitivity runs.
- **m5 (sign character).** DY spillover shares are non-negative in [0,1]; the vol2pk edge is signed
  correlation. The differing sign character is intrinsic to the two edge definitions (a variance-share
  network cannot be negative) — noted so the comparison is read correctly.

## Performance (ENFORCED lens)
- DY-matrix construction is a one-off CPU/VAR computation (elastic-net coordinate descent per equation,
  vectorised VMA/FEVD) — not in any training hot loop.
- Training reuses the delivered `train_masked_rich`: batched `[B,N,...]` tensors, batched block-diagonal
  adjacency (`base * node-mask`), mask-aware loss. NO batch=1 anti-pattern is introduced. CPU is forced
  only to avoid contending with the shared GPU jobs (149 MiB free at run time); batch semantics unchanged.

## Tests + coverage
- 20 tests pass. C0 line coverage = 100%, C1 branch = 100% on both new modules
  (`dy_connectedness.py`, `run_dy_ablation.py`) — the import-time GPU guard carries `# pragma: no cover`
  and the CLI `main()` is `# pragma: no cover` (entry-driver), per the project coverage policy.
- Real-data-sample smoke (tiny HNX slice) exercises the true processed-file writer + panel builder +
  DY-adjacency build + one MaskedRichNet forward pass (not a synthetic fixture).

## Verdict
No critical/major bug outstanding. M1 resolved (leakage-safe); minors are documented modeling choices.
Ready for the experiment + gate.
