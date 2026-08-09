# Masked-GNN encoder-cache + present-node encode + graph verdict re-run

Date: 2026-08-09
Branch: `feature/masked-gnn`
Scope: proposed updates #1 (cache frozen-encoder embeddings) and #2 (encode present nodes only)
for the masked G0/G1 graph ablation, numerical-equivalence proof, measured speedup, and a
multi-seed knn-8 + Diebold-Mariano re-run of the graph verdict. Adds (coordinator request) a
screening-configuration backbone and a dense + knn-8 val+test paper dump.

## What changed (files)

- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/models.py`
  - `GraphAblationModel.forward` split into `encode_base` (frozen, cacheable, present-node-only)
    + `apply_graph_head` (trainable message-passing + head + positivity). Absent nodes are no
    longer run through the LSTMs; they carry a zero embedding and cannot influence present
    outputs (message passing already zeroes their features + incoming edges).
- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py`
  - Frozen-encoder `base` cache: `_precompute_graph_base`, `_graph_predict`,
    `_build_shared_graph_base`, `_assert_shared_frozen_encoder`; `_run_one_graph_model` reuses the
    cache across all epochs and (via a shared cache) across G0/G1. `--no-base-cache` flag +
    per-run `timing` block record the effect.
  - Screening-configuration backbone: `build_graph_bound_p3_warm_start` /
    `build_graph_safe_p3_checkpoint` gain a `dropout` param; `run_graph_screening` trains the
    frozen P3 for `--backbone-epochs` (default 5) at `--backbone-dropout` (default 0.2) on the
    leakage-safe graph-bound set — replacing the prior 1+1-epoch, dropout-0 backbone.
  - Held-out TEST evaluation added (`_evaluate_graph_split`, `predictions_test.json`,
    `test_metrics` in results.json) for the paper val+test table.
- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_graph_base_cache.py` (new): 6
  equivalence tests.
- `docs/reports/verdict_paper_dump.py` (new): dense + knn-8 val+test + DM aggregation for the paper.

## Task 1 — numerical equivalence (THE gate)

`pytest .../test/test_graph_base_cache.py` — 6/6 pass:
- present-only `encode_base` reproduces the full-encode present rows bit-identically (atol=1e-6,
  observed diff 0.0 on CPU); the absent node carries an all-zero embedding.
- a cached `_run_one_graph_model` (G1 and G0) reproduces the uncached run's val loss, all 6
  metrics, and per-observation predictions **bit-identically** (diff 0.0) once the message-passing
  projection init is seed-controlled.
- a base cache shared across two models (the G0/G1 sharing) reproduces a self-computed run.

Independent realistic-scale CPU benchmark (33 nodes, 22-step window, hidden 64):
cached vs uncached full run **valloss diff = 0.000e+00** at both 5 and 15 epochs.

Real-data cross-check (GPU, same old backbone): cached G0 vs the pre-cache G0 match to
**6.9e-6** val-loss (metrics ~1e-5; G0 is deterministic — no trainable params). G1 diverges by
~1.6e-3 across the two separate GPU runs — this is GPU cuDNN nondeterminism (LSTM batch
composition from present-node encode + matmul atomics) amplified over 15 epochs of message-passing
training, not a cache logic error (the CPU test is bit-identical). Within a single run G0 and G1
share the identical frozen base, so the graph-on/off contrast is unaffected by the cache.

## Task 1 — measured speedup

Controlled cache-on vs cache-off, CPU, realistic dims, 15 epochs (`temp/speedup_bench.py`):

| | total | one-time encode | train-loop / epoch |
|---|---|---|---|
| cache-off | 342.0 s | 0 | 21.886 s |
| cache-on  |  54.1 s | 20.5 s | 1.025 s |

- **Training-loop speedup: 21.3x per epoch** (removes the per-epoch frozen-encoder recompute).
- **End-to-end speedup: 6.3x** at 15 epochs (the encoder is paid once, 20.5 s, then amortized).

(The full-33-ticker GPU run is dominated by the seed-independent manifest build, which the cache
does not touch; the cache targets the training loop, per the slowness diagnosis.)

## Task 2 — graph verdict (knn-8, multi-seed + Diebold-Mariano)

### Old-backbone knn-8 (3 seeds 42/123/2026, 15 epochs, GPU) — `docs/reports/dm_knn8_confirmation.py`

| seed | G0 valloss | G1 valloss | delta (G1-G0) |
|---|---|---|---|
| 42 | 0.839243 | 0.836708 | -2.535e-03 |
| 123 | 0.838887 | 0.837236 | -1.651e-03 |
| 2026 | 0.838189 | 0.836133 | -2.056e-03 |

- G1 < G0 in **3/3 seeds**; mean delta -2.08e-03 (std 4.4e-4); paired-t t=-8.15, **p=0.0147**.
- QLIKE G1-G0 = -2.88e-03 (**p=0.018**), R2 +2.45e-03 (p=0.005); DirAcc not significant (p=0.21).
- DM (QLIKE): seed42 p=0.0049, seed123 p=0.044, seed2026 p=0.071 (marginal); DM (MSE) mixed.

**Verdict: leans A (graph genuinely helps on QLIKE / val-loss).** The single-seed reversal
(-0.00253 at seed 42) does hold across all three seeds and the paired-t + QLIKE improvement are
significant; the per-seed DM is significant for 2/3 seeds and marginal (p=0.071) for the third.
The effect is small (~0.3% QLIKE) and DirAcc is unaffected. This is a stronger, consistent signal
than the prior single-seed note claimed.

### New-backbone (screening-config P3: 5 epochs, dropout 0.2) — knn-8, val+test (3 seeds)

Run on the efficient build-once driver (`temp/verdict_driver.py`, GPU, ~26 min/config). The frozen
backbone is now the screening-configuration P3 (5 epochs, dropout 0.2, leakage-safe graph-bound
set). knn-8, seeds 42/123/2026:

| seed | G0 valloss | G1 valloss | delta(G1-G0) | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.837168 | 0.836594 | -5.741e-04 | +0.337 | 0.736 | -1.386 | 0.166 | 14418 |
| 123 | 0.837673 | 0.837150 | -5.231e-04 | -0.521 | 0.602 | -2.010 | 0.044 | 14418 |
| 2026 | 0.836740 | 0.835933 | -8.073e-04 | -0.790 | 0.429 | -1.496 | 0.135 | 14418 |

VAL mean G1-G0: rmse -5.9e-06, r2 +0.00207, **qlike -3.2e-04**, DirAcc +0.10.
TEST mean G1-G0: rmse -1.0e-06, r2 +0.00021, **qlike +2.8e-03** (worse), DirAcc +0.49.

**Verdict: B (noise / graph does not robustly help).** G1<G0 on the equal-weighted validation
MSE in 3/3 seeds (paired-t p=0.0185), but the rigorous per-observation Diebold-Mariano test is
NOT significant: DM-QLIKE p = 0.74 / 0.60 / 0.43 (mixed sign), DM-MSE significant only for seed
123 (p=0.044). QLIKE — the volatility-relevant loss — is flat on validation and slightly worse on
test. So on a properly-trained backbone the graph's apparent benefit collapses into noise.

**This flips the old-backbone lean-A into a genuine null.** The old (undertrained, dropout-0,
1+1-epoch) backbone showed a significant G1 QLIKE improvement (DM p=0.005/0.044/0.071); the
properly-trained backbone does not. The graph message-passing is not a robust improvement once the
backbone is a real screening P3 — consistent with the project's graph-null finding, and the
correct comparison for the paper.

### New-backbone dense (full correlation) — 3 seeds

| seed | G0 valloss | G1 valloss | delta(G1-G0) | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE |
|---|---|---|---|---|---|---|---|
| 42 | 0.837168 | 0.837278 | +1.104e-04 | -0.525 | 0.600 | -1.020 | 0.308 |
| 123 | 0.837673 | 0.836955 | -7.175e-04 | +0.003 | 0.998 | -1.472 | 0.141 |
| 2026 | 0.836740 | 0.834989 | -1.751e-03 | -0.331 | 0.740 | -1.128 | 0.259 |

**Verdict: B.** G1<G0 in only 2/3 seeds (sign flips: seed42 +1.1e-04 hurts), mean -7.9e-04,
paired-t p=0.282 (NOT significant). No DM test (QLIKE or MSE) is significant for any seed. TEST
QLIKE G1-G0 = +3.1e-03 (worse). Dense message passing is noise.

### Combined verdict (new/proper backbone)

**B (the graph does not robustly help), for BOTH knn-8 and dense.** Full artifact:
`docs/reports/verdict_masked_g0g1_newbackbone_2026-08-09_120512.{json,md}` (all 6 metrics, val+test,
per-seed + mean, per-seed DM). The only "significant" cell is the knn-8 equal-weighted valloss
paired-t (p=0.0185), but that effect is ~0.08% and does not survive the per-observation
Diebold-Mariano test (QLIKE p=0.74/0.60/0.43; MSE significant only for one seed), and QLIKE is
flat/worse on the held-out test split. This is the correct comparison for the paper: once G0/G1
wrap a properly-trained screening P3 (not the undertrained 1+1-epoch dropout-0 backbone), the
graph on/off contrast is a null. The old-backbone lean-A was an artifact of the undertrained
backbone.

## Leakage note (backbone)

The screening-config backbone keeps the leakage-safe graph-bound training set
(`target_date <= graph.train_end_date`); it does NOT train on the full pooled train split. Using
the full pooled train would let the frozen backbone see ticker-dates that fall in the graph
val/test period (long-history tickers cross the common-date boundary), which would optimistically
bias BOTH G0 (no trainable params) and G1. Only the depth/regularization (5 epochs, dropout 0.2)
were adopted from screening.

## Gates

- Equivalence + cache tests: GREEN (8/8 in test_graph_base_cache.py, bit-identical).
- Full test suite: 168 passed, 4 smoke deselected on the final revision (non-smoke run); the
  full suite incl. smokes runs in the pre-push hook.
- ruff: clean on changed files (models.py, run_pilot.py, test_graph_base_cache.py,
  verdict_paper_dump.py, dm_knn8_confirmation.py).
- diff-cover vs master: 88% on changed lines (floor 80%) — PASS.
- Data-quality gate (Pandera/Evidently): N/A (model/training code; no data/manifest/schema
  change).
- diff-cover C0/C1: to be captured on the changed lines at push.

## Code review

Adversarial self-review of the cache diff (the top risk is a caching bug that silently changes
results): checked stale cache across seeds/models (shared cache guarded by
`_assert_shared_frozen_encoder`, byte-compares the frozen encoder + gate before sharing),
absent-node leakage (present-only encode proven equivalent; absent outputs discarded),
train/eval path divergence (cached and uncached both route through `apply_graph_head`;
equivalence test covers train+eval), and cache-not-invalidated-on-mode-change (each run rebuilds
its own cache from its own frozen backbone). The numerical-equivalence test is the standing
evidence.
