# Code review — GBME + 2-layer GNN node-embedding (2026-09-14)

Adversarial 3-layer review (Blind Hunter + Edge Case Hunter + Acceptance Auditor) plus a dedicated
**leakage lens** and **performance lens**, on `code/gnn_embed_config.py`, `code/embed.py`,
`code/run_gnn_embed.py`, `test/test_gnn_embed.py`. Shared deps (`gnnhar_sp500`, `full_matrix`,
`vn_gbm_graph_stage1`, `metrics`, `stats`, `overfit_check`) reviewed only for correct usage. `archive/`
out of scope.

## Leakage verdict: NO out-of-sample leakage found
- No test-window rows enter any GNN or GBM training; outer target embargo `date < ts - embargo` applied to
  train (`run_gnn_embed.py`), test window `[ts, tend)`.
- Inner-fold graph built on inner-train rows only (`embed._group_z`: `S1.build_graph(df[df.date.isin(train_dates)]...)`).
- Feature z-score stats + target scale train/inner-train only (`G.build_fold_tensors` train-cell mask).
- OOF assembly maps each embedded row back to its `trf` position, blocks partition all train dates, fail-loud
  coverage guard (`embed.oof_train_z`); asserted by `test_oof_coverage_and_causality`.
- Outer walk-forward causality now asserted by `test_outer_fold_causality` (mutating genuine-future rows
  leaves the fold result byte-identical).

## Performance verdict: PASS
GNN batched over dates on GPU (`G.BS`/step), no batch=1, one `.item()`/epoch for early stop, GPU-resident
tensors, CUDA `empty_cache`. GBM seed-ensemble refits are required (distinct `random_state`); train/val/test
predicted in a single `predict` over the concatenated frame (no 3× refit). Added OOF cost
`(INNER_K) + 1` GNN trainings/fold/horizon (single embedding seed) is documented.

## Findings and resolutions

| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| MAJOR-1 | Major | Embedding basis non-identifiability: OOF stitches `z` from `INNER_K` separate GNNs + a test GNN (different latent bases); seed-averaging embeddings shrinks them in an ill-defined basis. Biases the falsification toward NO-GO. | **Fixed (partial) + documented.** Dropped embedding seed-averaging — a single embedding seed (`emb_seeds = (G.SEEDS[0],)`); GBM still seed-ensembled. Inner-vs-test basis drift is inherent to leakage-safe OOF cross-fitting of neural features → documented as a validity caveat in `design.md` §9: a NO-GO is a **lower bound** on graph value, not proof of none. |
| MAJOR-2 | Major | Learned model `val_metrics` was in-sample (GBM fit on `trf_z` ⊇ `vaf`) → optimistic val → could false-flag `overfit` and BLOCK push. | **Fixed.** GBM now fits on `trf_e` (train MINUS the val slice); `val_metrics` is a true hold-out. GBM has no early stop, so dropping ~`VALID_LEN` dates is negligible on a multi-year window. |
| MAJOR-3 | Major | Missing tests for (a) outer walk-forward causality and (b) a real-embedder (`trainer=None`) end-to-end run. | **Fixed.** Added `test_outer_fold_causality` and `@pytest.mark.smoke test_run_real_embedder_smoke` (real torch forward + `embed()` gather + OOF/test wiring on a tiny panel). |
| MINOR-4 | Minor | Spike-excluded DM could crash at long horizons (`h >= n_dates`). | **Fixed** via `_safe_dm` try/except (covers spike + main). |
| MINOR-5 | Minor | `_safe_dm` guarded only exact-identical losses, not near-constant/zero-variance differentials. | **Fixed** — try/except around the reused DM maps its `ValueError` to p=1.0. `test_safe_dm_identical_losses` covers both the identical and the raise path. |
| MINOR-6 | Minor | Result provenance omitted GBM seeds. | **Fixed** — doc records `emb_seeds` + `gbm_seeds`. |
| MINOR-7 | Minor | `learning_curves` are first-seed only vs "per seed" wording. | **Reconciled** — single embedding seed now, so first-seed == the seed; requirements/design say representative. |
| MINOR-10 | Minor | Degenerate tiny inner-train (1 date) silently under-trained. | **Fixed** — `_group_z` raises loud when inner-train `< 2` dates (`test_group_z_raises_on_single_train_date`). |
| MINOR-8 | Minor | Import-time side effect `OWN = _own8()` reads a file. | **Accepted** — verified on a fresh git-archive tree (`run_gnn_embed.py --help` imports clean); by-path load avoids the bare-`config` module collision. |
| MINOR-9 | Minor | Redundant final checkpoint write. | **Accepted** — harmless; ensures a final durable write + the verbose print after a `fold_cap` break. |

## Tests / coverage
22 tests pass; diff-cover **C0=100% / C1=100%** on changed lines (`--cov-branch`). ruff-F clean;
config-hardcode clean. No critical/major finding left open.
