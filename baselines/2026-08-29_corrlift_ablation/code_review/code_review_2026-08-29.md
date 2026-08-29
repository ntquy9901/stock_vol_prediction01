# Code review — corr+lift edge probe (3-lens adversarial)

**Date:** 2026-08-29
**Scope:** `code/corrlift_edge.py`, `code/run_corrlift_ablation.py` (+ requirements/design/tests).
**Method:** 3-lens adversarial review (Blind Hunter / Edge Case Hunter / Acceptance Auditor), run as an
independent subagent against the two new files plus the delivered pipeline they import read-only and the
sibling `2026-08-29_sector_gat_ablation` runner.

## Outcome
No CRITICAL findings. One MAJOR (documentation accuracy of the leakage boundary) and three MINOR. All
resolved (MAJOR + terminology MINORs by precise wording; the two dead-safe redundancy MINORs accepted as
defensive). The edge math (Pearson, lift, combine, symmetry, self-loop, overlap guards, NaN/singleton
handling) and the runner wiring (leave-one-out variants on the same D/folds/seeds/floor; over/under-fit
evidence fed from the correct splits) were verified correct.

## Findings + resolution

### MAJOR — M1: leakage boundary looser than the delivered edges; design wording overstated it
The frozen graph cut is `D.d_va[0]` (first VALIDATION target date). The delivered `adj_corr` / `adj_vol2pk`
cut at `last_tr_row` (last TRAIN target), which — because `build_masked_rich` purges `horizon` anchors
between train and val — is a few rows earlier. So `d_va[0]` additionally ingests the purge-gap rows.
- **No evaluation leakage:** every included close row has `date < d_va[0] <=` every val/test target date, so
  the DM/test conclusion is uncontaminated. The only effect is a negligible train-relative optimism and an
  inconsistency vs the delivered edges' exact train universe.
- **Resolution:** `last_tr_row` is not exposed on `D` and could not be reproduced without editing the
  delivered `masked_rich.py` (out of scope — read-only). Per the reviewer's recommended option, the boundary
  is kept at `D.d_va[0]` (what the requirements literally specify and a genuinely non-leaking cut) and the
  overstated "strictly in the train region" wording was corrected in `corrlift_edge.py` docstring,
  `run_corrlift_ablation.py::corrlift_adj_for`, `design/design.md`, and the report — each now states the
  boundary is strictly before all val/test targets but marginally looser than the delivered cut, with nil
  effect here (the graph is near-empty: 15 edges / 142 singletons).

### MINOR — m1: "min-normalised" was a misnomer (code max-scales the lift excess)
`lift_str = lift_excess / max_excess` is max-scaling anchored at 0, not min/min-max normalisation. Numerically
correct (result in (0,1]). **Resolution:** reworded to "MAX-scaled by the largest fired excess" in the module
docstring, `build_corrlift_adjacency` docstring, the inline comment, and the report formula map.

### MINOR — m2: lift supports are over the co-observed transaction set, not global days
`support_i/j/ij` are all computed on `valid_i & valid_j` (days both stocks traded). Internally consistent and
the fair transaction universe for HNX's heterogeneous listing dates. **Resolution:** documented explicitly in
the module docstring and the report formula map so it is not mistaken for a global-day support.

### MINOR — m3: dead-safe redundancies (`np.nan_to_num(lift)`, `np.errstate`)
`np.nan_to_num(lift)` is redundant (masked by `lift_fires`); the `errstate` guard can't trigger (divisor
`np.where(n_fired>0, n_fired, 1) >= 1`). **Resolution:** kept as harmless defensive guards (no numeric or
readability cost that warrants a logic edit under the surgical-change rule); recorded here as accepted.

## Verified correct (no action)
- Pearson `rho` = standard mean-centred formula on co-finite days; `min_overlap=100` and constant-series both
  return NaN (no spurious edge). Independent-recompute test present.
- Lift = `support(i,j)/(support(i)support(j))`; `min_pairs=30`; zero marginal support -> NaN. Independent-
  recompute test present.
- Notable-move item = `|return| >` stock's TRAIN-median `|return|`, strict, on train-only returns.
- Combine: edge iff `|rho|>0.7` OR `lift>1.7`; weight in [0,1], finite; symmetric; self-loop forced to 1.0;
  `np.isfinite(adj).all()` asserted on real HNX.
- Alignment to `D.tickers`; missing price file -> all-NaN column -> singleton (no crash, no spurious edge);
  `adj.shape == (D.N, D.N)` always.
- Runner: three variants on the SAME `D`, `cfg.seeds`, `cfg.qlike_floor`; adjacency built once and frozen
  across seeds; DM includes the two required comparisons + `stat_vs_no_graph`.
- Over/under-fit evidence: `train_metrics`/`val_metrics` fed from the correct split arrays; `classify_fit`
  called in the right positional order; per-seed learning curves stored. A test pins the split ordering.
- Guards (empty val split, empty test split, non-finite forward output, `<2` processed files) all raise.
- Device label uses `torch.cuda.is_available()` (single source of truth; correct under `CORRLIFT_FORCE_CPU`).

## Tests / gate
- `test/test_corrlift_edge.py` (14) + `test/test_corrlift_runner.py` (12) = 26 pass under the GPU venv.
- Coverage on changed code: C0 line 100%, C1 branch 100% (both new modules).
- `ruff check --select F` clean (the 5 `E702` semicolon findings are the repo's warn-only house style).
