# Requirements — HAR-Anchored LSTM–GAT Residual Study (E0–E10)

Spec (source of truth) for the experiment ladder in
`docs/experement_guide/HAR_Anchored_LSTM_GAT_Experiment_Plan.md`. Governs by AGENTS.md/CLAUDE.md;
conflicts resolve in favor of AGENTS.md.

## Goal
Determine whether HAR-anchored residual learning, forecast combination, or regime-aware gating can beat
a HAR baseline **out of sample** for daily Parkinson-variance forecasting on VN30, under a leakage-safe
pipeline and defensible statistics. Reuse the tested `submission/soict_lstm_gat/` components read-only;
do not rewrite the project.

## Prediction contract (frozen for this study)
- Target: Parkinson **variance** (σ²), column `parkinson_volatility`. One scale throughout; never renamed
  "realized volatility".
- Target definition per horizon h: **terminal value** `pk[t+h]` (single day). Not an average/sum.
- Cutoff: close of day t. Features known at/after close of t only.
- Horizons: primary {1, 5, 10}; secondary {22}.
- Universe: 33 VN30 constituent files (survivorship snapshot — documented limitation). Secondary: VN100, SP500.
- Split: per-ticker chronological 80/10/10, fixed single split, **with target-overlap purge = h anchors**.
- Primary metric: **QLIKE** on variance. Also report MAE, RMSE, R²_OOS-vs-HAR, Pearson corr, mean bias,
  high-vol underprediction rate.
- Seeds: {42, 123, 2026, 7, 2024} (report mean, std, per-seed; never best-seed only).

## Experiment ladder
E0 HAR (locked benchmark) · E1 LSTM-HAR3 · E2 LSTM–GAT-HAR3 · E3 static convex α (frozen experts, val-fit)
· E4 horizon-specific α_h · E5 HAR + LSTM residual · E6 HAR + GAT residual · E7 HAR + LSTM–GAT residual
· E8 multiplicative/log HAR-anchored residual (recommended primary hybrid) · E9 static gated residual λ_h
· E10 dynamic regime-aware soft gate λ(z_{i,t}).

## Acceptance criteria (per plan §23 decision rules)
A hybrid is called "beats HAR" only when ALL hold:
1. Improves primary metric (QLIKE) on **validation AND locked test** vs HAR.
2. Loss differential statistically defensible (DM + block-bootstrap CI; MCS when many models; date-clustered).
3. Improvement holds across ≥3 seeds, not driven by one ticker or a few dates.
4. If graph value is claimed: beats the same-capacity **no-graph** residual model (E5), not just E2.
5. No critical temporal leakage / target-overlap remains (purge applied; audit clean).
6. Gate does not collapse to HAR everywhere unless the conclusion states neural correction adds no value.

## Leakage requirements (plan §19 — hard gate)
- Chronological split only; purge = exact target interval (h anchors) at each boundary.
- Scalers, HAR coefficients, glasso thresholds, graph edges, gate/regime thresholds fit on TRAIN only.
- Residual training targets built from cross-fitted (expanding-window / inner-fold) HAR predictions.
- Val used for early stop / α / λ / graph window / thresholds; test used once.

## Deliverables (plan §26)
1. `reports/leakage_audit.md` (done). 2. Leakage-safe fold generator + purge tests. 3. HAR + row-aligned
export. 4. E0–E10 implementations (staged). 5. Unit tests: HAR fallback, positive outputs, graph cutoff,
residual cross-fitting, tensor dims. 6. Prediction CSV/Parquet per fold/model/seed. 7. Metrics + stats
scripts (DM, block-bootstrap, MCS). 8. Graph/gate diagnostics. 9. `reports/experiment_results.md` with all
required tables + H1–H6 accept/reject. 10. `run_experiment.py` CLI.

## Go / No-go
- GO to residual gating (E9/E10) only if ≥1 residual expert (E5/E6/E7) shows incremental value on validation.
- If α→1 or residual R²_OOS ≤ 0: report HAR retained; neural correction adds no value (not a failure).

## Non-goals
- No walk-forward retraining (fixed single split, matching repo — documented).
- No new data crawling; use existing processed CSVs.
- No modification of the frozen `submission/soict_lstm_gat/` code or its `results/soict*`.
