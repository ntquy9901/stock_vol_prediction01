# Pooled LSTM, News, and GNN Ablation Pilot — Current Context

**Updated:** 2026-08-08
**Status:** Pilot implementation complete through GPU and batching optimization; full GNN 5-epoch result is partial (G0 complete, G1 safety-blocked).

## Research decision

- Price-only LSTM baselines must pool all eligible observations from every ticker; they do not require the 33-ticker common-date intersection.
- The 1,296 common trading days starting 2021-03-24 are retained only where synchronized cross-sectional tensors are structurally required: graph batches and the existing aligned news/gate experiment.
- News is a parallel per-stock branch fused with the temporal price representation. The common-date restriction is not assumed to be intrinsically required by news embeddings; it originated from batch/tensor alignment and gate-logit integration in the prior implementation.
- GNN remains a research component for ablation, not the mandatory core predictor.
- Planned ablations use a common pooled sample manifest for P0-P3 and a separate global-date graph manifest for G0/G1.
- Graph experiments must use a graph-safe P3 checkpoint trained only through the graph-train boundary. An unrestricted pooled P3 checkpoint may have observed dates belonging to graph validation/test sets.

## Data and leakage invariants

- Forecast horizon: 5 trading days; sequence length: 22.
- Split each ticker chronologically 70/15/15 before HAR construction, outlier handling, scaling, or window generation.
- Random split is forbidden. All data loaders use `shuffle=False` as the safest reproducible policy.
- Winsor bounds, feature scalers, and target scalers are fitted from training data only and stored per ticker.
- Validation/test extremes must not affect fitted preprocessing state.
- Preserve an untouched raw target for metrics separately from the clipped/normalized training target.
- Inverse target transforms dispatch by explicit `ticker_id`.
- QLIKE alone applies an epsilon floor; MSE, RMSE, MAE, R², and directional accuracy use unfloored inverse-transformed predictions.
- Report nonpositive prediction rate; a rate above 1% is a go/no-go failure.
- Directional accuracy is computed from changes, not signs of positive volatility levels.

## Experiment policy

- Pilot runs use seed 42 and 5 epochs first; a maximum of 10 epochs is allowed under the current approval.
- Training beyond 10 epochs requires a new explicit approval based on 5/10-epoch results.
- Report all six metrics: MSE, RMSE, MAE, R², QLIKE, and directional accuracy.
- Produce learning curves and progress metrics every 5 epochs during pilot training.

## Source-of-truth artifacts

- Requirements: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/requirements/requirements.md`
- Design: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/design/design.md`
- Implementation plan: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/design/implementation_plan.md`
- SDD ledger in worktree: `.superpowers/sdd/implementation_plan/progress.md`

## Git/worktree state

- Main repository: `C:\luanvan\stock_vol_prediction01`
- Isolated worktree: `C:\luanvan\stock_vol_prediction01\.worktrees\pooled-news-gnn-pilot`
- Branch: `feature/pooled-news-gnn-pilot`
- Setup commit on main: `f180496` (`Ignore local worktree directory`)
- Specification commit: `0edc36f`
- Design/plan commit: `87ce198`

## Implementation progress

1. Task 1 — deterministic ticker vocabulary and per-ticker raw chronological splits:
   - Commit: `e4be94ed870419698473a8ab8361d270a07bfe18`
   - Verification: 5 focused tests passed; Ruff passed.
   - Independent review: PASS for spec compliance and task quality; no findings.
2. Task 2 — train-only preprocessing, scaler invariants, HAR/window manifest:
   - Commits: `1e3174c`, `42b550f`, `c621221`, `2a540fa`.
   - Verification: Task 1-2 suite 24 passed; focused Ruff and diff checks passed.
   - Independent review: PASS for spec compliance and task quality after three correction rounds.
   - Corrections include full-window split-local HAR, persisted/hash-validated immutable manifests,
     exact feature-order enforcement, JSON round trips, finite/shape validation, read-only copied
     sample tensors, typed missing mappings, and strict integer ticker IDs.
3. Tasks 3-7 — causal news alignment, datasets/loaders, pooled baselines, news branch, runner and graph ablation: completed in the pilot worktree.
4. GPU path commit `387262e`: graph CLI supports `--device auto|cpu|cuda`, validates provenance before device transfer, records CUDA metadata, and preserves snapshot order. Independent review PASS; 83 tests and Ruff passed.
5. Pooled batching commits (worktree equivalents `24d5f68`, `c92020c`): default batch 256, cached normalized targets, pinned/non-blocking CUDA transfers, and sample-weighted loss aggregation. Independent review PASS.
6. Graph validation batching commits (worktree equivalents `3297004`, `417d556`): validation-only batching, per-snapshot weighting corrected for short final batches; training remains one-update-per-snapshot. Independent review PASS.
7. Final pilot worktree verification: 87 tests passed, Ruff passed, `git diff --check` passed. Summary: `docs/reports/2026-08-08_1320_summaryOfUpdate_report.md`.

## Verified pilot results

- Full 33-ticker GNN CUDA 5-epoch run: approximately 18m44s on RTX 4060 (`torch 2.6.0+cu124`). G0: RMSE `0.0026032073`, QLIKE `0.82523848`, DirAcc `49.3902%`, R² `0.70135174`.
- G1 was blocked by the existing safety guard because nonpositive predictions were `1.78%` (threshold `1%`). Checkpoints were preserved; do not treat G1 as a completed result.
- Three-ticker batching benchmarks did not show end-to-end speedup because preprocessing dominates. Graph batched validation metrics matched serial validation; larger pools may benefit.
- Earlier full 33-ticker references: P0 HAR RMSE `0.0014845167`, P1 pooled LSTM RMSE `0.0014670183`; do not claim G0 currently beats these baselines.

## Current continuation point

- Active implementation branch: `feature/pooled-news-gnn-pilot` in `C:\luanvan\stock_vol_prediction01\.worktrees\pooled-news-gnn-pilot`.
- Latest optimization commits on that branch: `24d5f68`, `c92020c`, `3297004`, `417d556`, report `298464f`.
- Untracked `temp/` and task report files are user-owned experiment artifacts; preserve them.
- Next research action: investigate positivity parameterization/safety handling for G1, then rerun G1 for 5 epochs. Do not increase beyond 10 epochs without explicit approval. Consider preprocessing/index caching before claiming batching speedup.

## Working rules and preservation notes

- Implement through the subagent-driven development workflow selected by the user: fresh implementer per task, independent spec/quality review, correction loop before advancing, then broad final review.
- Use test-first development and baseline-local hard isolation. Shared `src/` modules are read-only.
- Preserve unrelated modified/untracked files in the main worktree. The main worktree currently contains user-owned horizon scripts and experiment results that must not be cleaned or reset.
- Project `AGENTS.md` is the constitution and overrides conflicting plan details.

## Immediate continuation

Complete Task 3 with RED/GREEN evidence and independent spec/quality review. Advance to Task 4 only
after causal effective-date alignment, real-panel smoke coverage, and deterministic content hashes
pass review.
