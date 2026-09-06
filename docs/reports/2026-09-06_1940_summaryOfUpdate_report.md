# Summary of update — XGBoost residual-ratio volatility study (2026-09-06)

## What changed

New isolated baseline `baselines/2026-09-06_xgboost_residual/` implementing the study specified in
`docs/experement_guide/AI_MASTER_INSTRUCTION_XGBOOST_EXPERIMENTS.md`: does HAR-X + XGBoost residual-ratio
(and a direct XGBoost) beat HAR-X on out-of-sample Parkinson-**variance** QLIKE for VN30/VN100 at
h∈{1,5,10,22}? Answer: **NO-GO** (clean negative result).

## Files (path → purpose)

- `baselines/2026-09-06_xgboost_residual/requirements/requirements.md` — spec + go/no-go.
- `.../design/design.md` — data flow + SDD gates.
- `.../code/xgb_config.py` — all tunables (windows, floors, XGB grid, alpha/clip grid, OOF params, seed, n_jobs).
- `.../code/xgb_features.py` — causal stock/market/graph feature builders (+ aligned column reader).
- `.../code/xgb_oof.py` — chronological OOF HAR-X residual target + reconstruct.
- `.../code/xgb_model.py` — XGBoost fit + val-only grid/guardrail selection (CPU, deterministic).
- `.../code/xgb_eval.py` — pooled diagnostics (lock split, top-1% exclusion, win rates, counts, shock trace).
- `.../code/run_xgboost_residual.py` — walk-forward fold loop + model ladder + metrics + DM + JSON.
- `.../code/summarize_xgb.py` — summary/ablation/lock tables + permutation-importance figure.
- `.../test/test_xgbres_*.py` — 42 tests (causality/OOF-cutoff/graph-cutoff/split-isolation/count/
  reproducibility/HAR-X parity/lock mask/real-data smoke).
- `.../code_review/code_review_2026-09-06.md` — adversarial self-review (leakage/fairness/edge/perf).
- `results/xgboost_residual/xgb_{vn30,vn100}_h{1,5,10,22}.json` — 8 result artifacts (metrics, DM, fit
  evidence, selections, shock diagnostics, reference VolGA/LSTM, git commit, versions).
- `results/xgboost_residual/importance_vn30_h1.png` — validation permutation-importance figure.
- `docs/reports/2026-09-06_xgboost_residual_report.md` — method/results/limitations/GO-NO-GO report.
- `scripts/quality_gate/overfit_check.py` — **modified**: added `xgb`/`catboost`/`lightgbm` to the
  learned-model pattern list so the pre-push overfit gate enforces fit evidence for tree learners (the one
  deviation from strict isolation; flagged for coordinator sign-off; regression test added).

## Tests + coverage

- `python -m pytest baselines/2026-09-06_xgboost_residual/test/ scripts/quality_gate/ -k "xgbres or overfit"`
  → **66 passed** (42 baseline + 24 gate regression).
- Diff coverage: **C0 = 100%** on all new non-pragma code lines (`--cov` term-missing = 0 misses across the
  6 code modules); only argparse `main()` / environment probes / matplotlib I/O carry `# pragma: no cover`.

## Code review

Adversarial self-review across leakage / fairness / edge-case / performance lenses (see code_review doc). No
critical/major defects. Leakage guarded by tests (backward-only features, OOF cutoff, train-only graph
adjacency, HAR-X parity with the delivered split). Coordinator to run the full 3-layer `/code-review`.

## Data-quality gate

N/A for schema/drift — this study reads the already-frozen enriched panels read-only and creates no new
raw/processed data. Fairness/parity checks (shared floors, identical test cells, HAR-X == delivered
edge_hmatched HAR-X) stand in for the model-comparison integrity.

## Performance

CPU-only (`tree_method='hist'`, `n_jobs=4`, one deterministic seed) — GPU left untouched for the concurrent
transformer agent. XGBoost fits are batched matrix ops; grid is a curated 5-candidate list, not a sweep.

## Result (headline)

NO-GO. XGB residual lower QLIKE than HAR-X in only 1/8 cells (vn30 h1, not significant, p=0.791); worse in
7/8; ticker/date win-rate < 0.5 everywhere; market/graph add nothing (guardrail often picks alpha=0); direct
XGBoost far worse and over/underfits (gate flags 4 JSONs on XGB_direct). Improvement absence survives
non-lock and top-1%-date exclusion. HAR-X remains the parsimonious tabular champion; VolGA the reference
deep model. Transformer escalation not warranted by this outcome.

## Risks / follow-ups

- 4 result JSONs (vn100 h1/h10/h22, vn30 h22) are blocked by the overfit gate **on XGB_direct** (a reported
  negative control that genuinely over/underfits). This is a true-positive gate hit, not a data problem —
  pushing requires coordinator judgment (QG_SKIP with this documented justification, or excluding
  XGB_direct from those artifacts). The residual-ratio primary models pass the gate everywhere.
- Not committed — left staged-clean for the coordinator per task instruction.

## DoD checklist

- [x] Code satisfies request; surgical; hard-isolated (one justified gate edit).
- [x] Tests written + run (66 pass), C0=100% on changed lines.
- [x] Smoke passes (real-data `test_smoke_run_vn30_h1`).
- [x] Adversarial code-review artifact produced.
- [x] Summary report + study report produced.
- [ ] Push — deliberately NOT done (coordinator commits after review + gate).
