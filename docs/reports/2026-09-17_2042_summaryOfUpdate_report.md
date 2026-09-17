# Summary of update — frozen-basis GNN embedding variant (GBME+GNN-embed) + Hướng B research

Date: 2026-09-17. Two items: (1) implemented + ran the fixed-basis ("frozen") variant of the
`2026-09-14_gbm_gnn_embed` baseline to close its design §9 basis-drift caveat; (2) an adversarial research
agent assessed the "Foundation Model prior + GBME residual" direction (Hướng B).

## 1. Frozen-basis GNN embedding → GBME

### Motivation
The delivered OOF variant showed apparent long-horizon losses (HOSE h10 −36%, h22 −18%) that were root-caused
as an **embedding basis-drift artifact**: OOF cross-fitting stitches `z` from `INNER_K` separately-fitted GNNs
plus a test GNN, each in a different (rotation/permutation-ambiguous) latent basis. design §9 flagged that a
NO-GO under OOF is only a *lower bound* on graph value. The frozen variant removes the ambiguity: train ONE GNN
on the burn-in window (first eligible fold's causal train window, before every test fold), freeze it, and reuse
its single-basis embeddings for every fold.

### Files (path → change)
- `baselines/2026-09-14_gbm_gnn_embed/code/embed.py` → new `frozen_z(...)`: trains one GNN on burn-in via the
  existing `_group_z`, embeds every panel row in one basis, returns `{row_index → z}` + learning curves.
- `baselines/2026-09-14_gbm_gnn_embed/code/run_gnn_embed.py` → `run(..., frozen=False)` param + `--frozen` CLI;
  frozen branch builds the panel-wide embedding once at the first eligible fold and slices z_train/z_test per
  fold (no per-fold GNN refit). Output tag `_frozen`. OOF path unchanged (backward-compatible).
- `baselines/2026-09-14_gbm_gnn_embed/test/test_gnn_embed.py` → 2 tests: `frozen_z` single-basis coverage +
  causality (one `_group_z` call, train=burn-in, emb=all dates, every row covered); `run(frozen=True)`
  structure + shared-basis learning curve + gate evidence.

### Result (HOSE, gamma-GBM, 8 folds × 3 GBM seeds, DM date-clustered, spike-robust)
| h | GBME QLIKE | +frozen QLIKE | gain | DM p | ex-spike gain | fit |
|---|---|---|---|---|---|---|
| h1 | 1.5719 | 1.6004 | −1.82% | 0.0000 | −1.83% (p=0.0000) | ok |
| h5 | 1.6479 | 1.8683 | −13.37% | 0.29 | −15.75% | ok |
| h10 | 1.7574 | 1.7932 | −2.04% | 0.097 | −2.67% | ok |
| h22 | (base) | | −4.58% | 0.19 | −5.51% | ok |

All horizons `beats=False`. Two findings:
1. **Frozen basis removed the OOF detonation.** `fit=ok` at every horizon; h10 is now −2% not −36%. This
   **confirms the earlier root-cause**: the OOF long-horizon losses were basis-drift artifacts, not graph
   evidence. Design §9 caveat is closed — with a fixed basis there is no drift.
2. **Even artifact-free, the embedding is NO-GO.** h1 is DM-**significantly worse** (−1.82%, p=0.0000,
   spike-robust, clean across all 8 folds) → the graph embedding genuinely dilutes the own-AR signal that
   dominates per-stock QLIKE. h5/h22 pooled negatives (−13% / −4.6%) are each driven by a single unstable fold
   (h5 fold6 −116%, h22 fold4 −29%); the other 7 folds are −0.0…−1.1%, and date-clustered DM correctly reports
   not-significant (isolated thin-market storm folds, not a systematic effect).

**Verdict: clean NO-GO.** Stronger evidence than the OOF run: correcting the basis-identifiability defect does
not rescue graph-as-feature for per-stock QLIKE — it remains non-positive at all horizons and DM-sig worse at
h1. The residual single-fold blow-ups show a fixed basis still cannot fully stabilise a graph feature on thin
HOSE storm folds.

### Tests / gate
- New frozen tests pass (2 passed, 49s). Real end-to-end frozen smoke on HOSE h1 ran clean (no detonation).
- Full baseline test suite re-run (see gate log at push time). Overfit-evidence: each `_frozen` result JSON
  carries train/val/test 5-metric blocks + fit_diagnostics + learning_curves (fit=ok).

## 2. Hướng B research — Foundation Model prior + GBME residual → NO-GO (high confidence)

Adversarial agent, repo-grounded + external literature. Verdict NO-GO for the multiplicative composition
`ŷ = ŷ_foundation · exp(r̂_GBME)`; only a cheap zero-shot-vs-HAR additive-feature gate is defensible (low prior).
Decisive points:
- **TimesFM/Chronos measured to lose to Log-HAR under QLIKE OOS** — Brini 2026 (arXiv:2607.05291), exact
  QLIKE+DM+MCS protocol: TimesFM 2.5 loss-ratio vs HAR 1.086/1.201/1.331 (h1/5/22); only small TTM marginally
  wins, best as a HAR blend.
- **Foundation stage redundant with HAR** the champion GBM already ingests (har_daily/weekly/monthly) —
  collinearity like the +graph concat (err_corr 0.98).
- **Multiplicative form detonates QLIKE** — repo already hit this: `har_anchored_residual` E8
  `multiplicative_pred` produced −310%/−223% fold blow-ups.
- **TimesFM already built + archived here** (`archive/src_legacy/timesfm_baseline/`, commit a0b18c37), never
  beat HAR.

## Follow-ups
- Frozen-basis result closes the GNN-embed §9 caveat; graph-as-feature for per-stock QLIKE is now a clean
  falsification across OOF + frozen. Recommend consolidating into the paper's negative-result section.
- Hướng B: do not build the multiplicative anchor. The per-stock QLIKE frontier stands.

## Code review
Pre-push 3-tier gate (ruff-F, config-hardcode, diff-cover C0=100%/C1≥95%, overfit-evidence, checklist) runs on
push. Changes are a backward-compatible variant mode on an existing baseline + tests + result JSONs; no
production-path logic altered outside the added frozen branch.
