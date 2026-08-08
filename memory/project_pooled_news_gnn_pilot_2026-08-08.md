# Pooled LSTM, News, and GNN Ablation Pilot — Current Context

**Updated:** 2026-08-08  
**Status:** Implementation in progress in an isolated worktree

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
   - Commit: `1e3174cdd15d1015d385acfde2d9126625d0683f`
   - Verification reported by implementer: Task 1-2 suite 12 passed; Ruff and diff checks passed.
   - Independent review: CHANGES_REQUIRED (round 1).
   - Blocking corrections in progress: use full-window HAR and drop split-local warm-up rows;
     persist and validate the shared P0-P3 manifest; select only the persisted feature order and
     reject stale/non-finite columns. Review also requested preprocessing JSON round-trip coverage
     and typed missing-ticker errors.
3. Tasks 3-10 remain pending: datasets/loaders, pooled baselines, news branch, runner, graph-safe checkpoint, GNN ablations, pilot execution, and final validation/reporting.

## Working rules and preservation notes

- Implement through the subagent-driven development workflow selected by the user: fresh implementer per task, independent spec/quality review, correction loop before advancing, then broad final review.
- Use test-first development and baseline-local hard isolation. Shared `src/` modules are read-only.
- Preserve unrelated modified/untracked files in the main worktree. The main worktree currently contains user-owned horizon scripts and experiment results that must not be cleaned or reset.
- Project `AGENTS.md` is the constitution and overrides conflicting plan details.

## Immediate continuation

Wait for the Task 2 correction commit, generate a delta review package, and repeat independent
review. Advance to Task 3 only after both spec-compliance and task-quality verdicts pass.
