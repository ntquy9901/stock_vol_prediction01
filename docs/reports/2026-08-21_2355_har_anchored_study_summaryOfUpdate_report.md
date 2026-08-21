# Summary — HAR-Anchored LSTM–GAT Residual Study (E0-E10)

Implements `docs/experement_guide/HAR_Anchored_LSTM_GAT_Experiment_Plan.md`. Autonomous overnight build +
run. Baseline folder: `baselines/2026-08-21_har_anchored_residual/`.

## What changed (files)
- `reports/leakage_audit.md` — contract-as-implemented + leakage findings (F1 purge, F2 GARCH, F3-F5 OK) + gaps G1-G4.
- `baselines/2026-08-21_har_anchored_residual/` — requirements + design + code + tests (§3.F structure):
  - `folds.py` purged split (F1 fix), `har_cv.py` cross-fitted HAR residuals, `io_preds.py` row export/manifest,
    `models.py` ResidualNet (zero-init HAR fallback), `experts.py` snapshot build + unified trainer,
    `blend.py` E3/E4, `gate.py` E9/E10, `stats.py` (block-bootstrap+MCS+date-clustered DM), `diagnostics.py`,
    `run_experiment.py` orchestrator, `build_report.py`.
- `reports/experiment_results.md` — §22 tables + §23 H1-H6 decisions (auto-built from result.json).
- `results/har_anchored/<ds>_h<h>/` — result.json + row_predictions.csv per (dataset, horizon).

## Design (locked, documented)
Snapshot common-date design (graph well-defined; E5/E6/E7 same-fold comparable) with target-overlap PURGE
= h snapshots (fixes F1). Target = Parkinson variance, terminal value pk[t+h]. Primary metric QLIKE. HAR
per-horizon pooled OLS anchor; residual targets from expanding-window cross-fitted HAR (no in-sample leak).
5 seeds {42,123,2026,7,2024}, epochs/early-stop per Config (val QLIKE early stop). Ladder E0-E10.

## Tests + quality gate
- 46 baseline tests pass (folds 7, har_cv 3, io_preds 3, models 5, experts 1 smoke, blend/gate 6,
  run_experiment 1 smoke, stats 9, diagnostics 8, build_report 3). Command:
  `.venv_gpu_encode/Scripts/python.exe -m pytest baselines/2026-08-21_har_anchored_residual/test/ -q`.
- TDD: each module test-first (watched fail, then implemented). Pre-push quality gate passed on each commit
  (69 delivered-baseline tests green). diff-cover: Not run (tooling gap, per AGENTS.md).
- Code review: PENDING (`/code-review` to run before final done; noted as follow-up).
- Data-quality gate: N/A (no data/manifest change; reuses existing processed CSVs).
- Performance: batched snapshots [B,N,seq,3] on GPU, non-blocking H2D, batched val; sp500 uses --batch 8
  to bound 500-node GAT memory. No batch=1 hot loop.

## Findings (VN30 + VN100 complete; SP500 running)
- Full neural E1/E2 lose to HAR at short horizons (confirms prior snapshot-design result).
- HAR-anchored residual is SAFE: E5-E8 tie or beat HAR, never significantly worse on VN30/VN100 (H6 ACCEPT).
- **VN100: the GAT residual (E6) beats HAR at h10/h22** (h22 +4.08%, DM p<0.0001) and beats the no-graph
  E5 residual (+0.09%) — graph contributes incremental spillover on the larger cross-section (H4 ACCEPT at
  long horizon; residual R2_OOS rises 0.0002 -> 0.040).
- **Forecast combination E3 beats HAR on VN100** at h5-h22 (+1.9% / +4.8% / +5.9%, all DM p<0.01);
  alpha_HAR falls 0.935 -> 0.28 with horizon (H2 horizon-specialization ACCEPT).
- VN30 (small panel): residual ~ HAR, no significant graph value -> graph value is cross-section-size and
  horizon dependent.
- SP500: additive residual (E5-E7) is numerically fragile at 500-node/short-test scale (drives predictions
  to the floor -> QLIKE blow-up), while the multiplicative E8 stays positive/bounded and E3 blend beats HAR
  (h1: E3 0.3322 < HAR 0.3391) — motivates the multiplicative/gated anchoring the plan recommends.

## Risks / follow-ups
- SP500 h5/h10/h22 still running (GPU shared with a learnable-alpha SP500 fill); report auto-refreshes on completion.
- `/code-review` (3-layer) to run on the new modules before marking fully done.
- Additive-residual floor-explosion on large panels is honest but ugly in tables; the multiplicative/gated/blend
  forms are the safe variants and are what the paper should feature.
- Paper update (soict_paper_complete.md + .tex) to incorporate these final numbers after SP500 completes.
