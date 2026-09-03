# Design — Pooled/transfer ablation for VN30

Full design spec: `docs/superpowers/specs/2026-09-04-pooled-transfer-vn30-design.md`.
Implementation plan: `docs/superpowers/plans/2026-09-04-pooled-transfer-vn30.md`.

## Data flow (implemented realisation — single-panel mask)

Refinement over the spec's two-panel wording: use ONE VN100 panel + ONE fold set, and differ arms
by a **training-node mask**, so the OOS grid is byte-identical across arms (stronger alignment than
date-mapped two panels). Isolation (Arm 0 = genuine 31-node system) is test-gated
(`test_isolation.py`): Arm 0 VN30 predictions must be invariant to non-VN30 node feature values,
which holds because the vol→PK adjacency is restricted to VN30 and the LSTM branch is per-node. If
that test fails, fall back to a separate VN30 panel with date-aligned folds.

```
frozen_universe(vn100) -> 102-node panel ---> make_folds (shared)
frozen_universe(vn30)  -> vn30_index within the 102-node panel
per fold:
  D = pack_fold(panel, fold)                     # train-only scalers + vol->PK graph
  Arm1: run trainer on D (all 102), score vn30   # train_idx = arange(102)
  Arm0: run trainer on restrict_fold(D, vn30)    # train_idx = vn30_index; graph+loss VN30-only
  both: score_mask -> _pred_dict on vn30 only
paired DM (Arm1 vs Arm0) for LSTM & VolGA on shared VN30 keys, 3 bases; diff-in-diff vs HAR.
```

## Design gates (SDD §5)
- Simplicity: reuses delivered panel/trainer/DM helpers read-only; new code = 3 small modules.
- Anti-abstraction: no wrappers beyond `restrict_fold`/`score_mask`/`run_arm`.
- Performance/batching: reuses the already-batched `train_masked_rich` ([B,N,seq,5] on GPU); no
  batch=1 loop introduced. Arm 0 processes the full 102-node tensor (loss/graph masked) — same
  batched cost as Arm 1; accepted for exact alignment.
