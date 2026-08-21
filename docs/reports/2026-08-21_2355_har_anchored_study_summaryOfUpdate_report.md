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

## Findings (VN30 + VN100 complete; SP500 partial) — inference-corrected after code review C-1

Statistics use the **date-clustered** Diebold–Mariano (one loss value per date), the panel-correct test:
all ~33/104 tickers share each target date, so naive per-observation DM treats n = N x T_dates and
over-states significance by ~sqrt(N) (~6-8x). The initial read used the row-level DM and reported spurious
significance; the code review (C-1) caught it and the report/decisions now read the date-clustered p-value.

- **No model significantly beats HAR** on VN30 or VN100 at any horizon under date-clustered DM:
  E3 combination vs HAR p in [0.14, 0.51] (VN100), graph residual E6 vs HAR p in [0.56, 0.93]. Point-estimate
  QLIKE gaps favor the hybrids at longer horizons (e.g. VN100 h22 E6 -4.1%) but are **within noise** given
  the short common-date test window (~49-130 dates).
- **Graph adds no incremental value under proper inference:** paired date-clustered E6-vs-E5 (graph vs
  no-graph residual) p in [0.76, 0.995] on VN100, direction often favoring the no-graph E5. H4 REJECT.
- HAR-anchored residual is SAFE as a point estimate (E5-E8 ~ HAR, never far worse) but full-neural E1
  occasionally edges the residual at long horizons on VN100 (H6 not universally held).
- Full neural E1/E2 lose to HAR at short horizons (confirms prior snapshot-design result).
- alpha_HAR falls 0.935 -> 0.28 and lambda rises 0 -> 2 with horizon (point-estimate horizon specialization,
  H2), consistent with more neural weight being optimal at longer horizons — but the resulting blend still
  does not significantly beat HAR.
- SP500: additive residual (E5-E7) is numerically fragile at 500-node/35-test-date scale (drives
  predictions to the floor -> QLIKE blow-up, E7=16.4); multiplicative E8 stays positive/bounded. SP500 h1
  shows date-clustered significance (E3 vs HAR p=0.0002) but rests on only 35 test dates and a degenerate
  additive branch — treated as unreliable, not a beat-HAR claim.

**Honest verdict:** consistent with the volatility literature (Branco et al., "Does anything beat linear
models?"), HAR is not beaten out-of-sample on these panels under rigorous panel inference. The main
methodological contribution is that apparent neural/graph wins are artifacts of ignoring cross-sectional
dependence in the significance test. HAR-anchoring makes deep models safe (tie HAR) where the full-target
deep models lose.

## Risks / follow-ups
- SP500 h5/h10/h22 still running (GPU shared with a learnable-alpha SP500 fill); report auto-refreshes on completion.
- `/code-review` (3-layer) to run on the new modules before marking fully done.
- Additive-residual floor-explosion on large panels is honest but ugly in tables; the multiplicative/gated/blend
  forms are the safe variants and are what the paper should feature.
- Paper update (soict_paper_complete.md + .tex) to incorporate these final numbers after SP500 completes.
