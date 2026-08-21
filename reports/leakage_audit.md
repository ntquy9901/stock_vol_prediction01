# Leakage Audit — HAR-Anchored LSTM–GAT Study

Scope: audit of the current volatility-forecasting pipeline against the prediction contract and
temporal-leakage controls in `docs/experement_guide/HAR_Anchored_LSTM_GAT_Experiment_Plan.md`
(§2 contract, §19 leakage controls, §26 deliverables). Audited before any performance claim.

Audited code (current headline pipeline, reused read-only by the new study):
- `submission/soict_lstm_gat/data_utils.py` — HAR features, windowing, per-stock split, per-ticker scaler.
- `submission/soict_lstm_gat/run_lstm.py` — main per-observation LSTM vs HAR + GARCH.
- `submission/soict_lstm_gat/run_all.py` / `snapshots.py` / `model.py` — common-date snapshot graph-check.
- `submission/soict_lstm_gat/baselines.py`, `edges.py`, `metrics.py`.

Data audited: `submission/soict_lstm_gat/data/vn30/*_processed.csv` (33 files),
`.../vn100/`, `data/processed/sp500/`. Schema per file: `date, parkinson_volatility`.

---

## 1. Prediction contract (as implemented)

| Contract item (§2) | Value found in code | Source |
|---|---|---|
| Target type | Parkinson **variance** (σ²), despite the column name `parkinson_volatility` | `data_utils.py:8-11`; memory `parkinson-target-is-variance` |
| Target scale | Single scale throughout a run (raw variance); StandardScaler only for the neural branch, inverse-transformed before metrics | `run_lstm.py:144-146` |
| Target definition per horizon h | **Terminal value**: `pk[t+h]` — the single-day Parkinson variance on day t+h. NOT an average or sum over [t+1..t+h] | `data_utils.py:161,168`; `run_lstm.py:78-83` |
| Horizons | {1, 5, 10, 22} runnable; plan primary set {1, 5, 10} | `run_lstm.py` CLI |
| Prediction cutoff | Close of day t. Window = days [t−lookback+1 .. t]; HAR daily=pk(t), weekly=mean(pk[t−4..t]), monthly=mean(pk[t−21..t]) — all known at/after close of t | `data_utils.py:32-57` |
| Entity key | (ticker_id, target_date) where target_date = calendar date of day t+h | `run_lstm.py:82,147` |
| Universe policy | Current-membership snapshot of 33 VN30 constituents; **no delisted/replaced tickers** (survivorship snapshot) | data dir; memory `vn30-ticker-universe-mismatch` |
| Train/val/test | Per-ticker chronological contiguous **80/10/10** on valid anchors | `data_utils.py:60-68` |
| Retraining schedule | **Fixed single split** (not expanding/sliding/walk-forward) | `data_utils.py:60-68` |
| Scaler fit | Per-ticker StandardScaler, fit on **TRAIN anchors only** (features and target) | `data_utils.py:152-155`; `run_lstm.py:69` |

Naming note: the plan (§1–2) forbids renaming the target `realized volatility`. The column
`parkinson_volatility` is a **variance** computed from daily High/Low; QLIKE (`y/ŷ + log ŷ`) is applied
to that variance, which is internally consistent. Reports must call it Parkinson variance, not realized
volatility.

## 2. Leakage findings

### F1 — [MEDIUM] No target-overlap purge at split boundaries
`per_stock_split` (`data_utils.py:60-68`) cuts contiguous anchor ranges with **zero gap**. Anchors are
consecutive trading days, so the last (h−1) train anchors have target dates `t+h` that fall inside the
validation date range; the same holds at the val→test boundary. Plan §19 mandates purging training rows
whose target interval overlaps validation, and validation rows whose target interval overlaps test, with
the purge derived from the exact target interval.

- Impact scales with horizon: h=1 → 0 rows; h=5 → up to 4 rows/ticker/boundary; h=10 → 9; h=22 → 21.
- Severity MEDIUM: it is target-date overlap near the boundary (model-selection peek), not feature
  leakage; magnitude is small relative to thousands of pooled rows, but it is a mandated control and the
  effect grows with horizon.
- Fix (in the new study harness): insert a purge gap of `horizon` anchors between train/val and val/test.
  The existing frozen submission is NOT modified; the new leakage-safe fold generator adds the purge and
  a unit test asserting no target-date of an earlier split lands in a later split.

### F2 — [LOW] GARCH baseline fits on train+val
`run_lstm.py:73` and `evaluate.py:50` build the GARCH pre-test series from **train + val** targets so the
one-step-repeated forecast aligns to the test boundary. This is not test leakage (val is legitimately
pre-test), but GARCH sees more history than the LSTM's train-only fit. Acceptable for a persistence
baseline; recorded here for fair-comparison transparency. The new study fits HAR and neural experts on
train only and keeps GARCH's train+val series, documented.

### F3 — [OK] Graph edges are leakage-safe
`edges.glasso_adjacency` is fit on `snap.adj_pk_train` (train-only Parkinson panel) and frozen across
val/test (`run_all.py:51`). No contemporaneous/future information enters the graph. The graph is
**static** (single split), which is leakage-safe but does not exercise per-fold re-estimation; the new
study keeps train-only edge estimation and adds the identity/no-graph and shuffled-edge placebo ablations
required by plan §15.

### F4 — [OK] Scalers fit train-only
Per-ticker feature and target scalers are fit on train anchors only (`data_utils.py:152-155`) and applied
unchanged to val/test. No scaler leakage.

### F5 — [OK] Test set not used for selection
Early stopping and best-checkpoint selection use validation MSE (`run_lstm.py:122-126`); HAR and GARCH are
unselected; the test set is evaluated once. No hyperparameter/epoch/model choice reads the test set.

## 3. Methodology gaps vs the plan (not leakage, but blocking correct attribution)

### G1 — E3/E4 not yet the plan's frozen-expert form
The just-completed `run_alpha.py` **jointly co-trains** α with the LSTM. The plan's E3/E4 require the
experts to be **frozen** and α fit on **validation predictions only** (closed-form for MSE, grid/optim for
QLIKE). The joint form is a different estimator; E3/E4 as specified must be implemented separately.

### G2 — Cross-fitted residual targets do not exist
E5–E8 require residual targets built from **out-of-sample-style** (expanding-window or inner-fold) HAR
predictions inside the training set (plan §10, §19). No such cross-fitting exists yet; it must be built so
residual targets reflect deployment residuals, not optimistic in-sample HAR residuals.

### G3 — Statistics beyond single DM
The pipeline runs Diebold–Mariano (HLN small-sample correction, HAC lag h−1). The plan §18 additionally
requires block-bootstrap confidence intervals for loss differentials, a Model Confidence Set when many
models are compared, and date-aggregated (not naive row-level) standard errors for the dependent panel.
These are to be added in the new study's evaluation module.

### G4 — Row-aligned prediction export + feature-availability manifest
Plan §21/§26 require row-aligned predictions for every fold/model/seed (CSV/Parquet) and a
feature-availability manifest. The current runners persist only aggregate `result.json`; the new study
adds per-row prediction export and the manifest.

## 4. Actions

1. Build a leakage-safe fold generator **with target-overlap purge** (fixes F1) + purge unit test — before any E-experiment runs.
2. Keep F2 (GARCH train+val) as-is, documented.
3. Reuse F3/F4/F5 (already leakage-safe) read-only.
4. Implement E3/E4 in the plan's frozen-expert / validation-fit form (fixes G1).
5. Build cross-fitted HAR residual targets (fixes G2) with a unit test.
6. Add block-bootstrap CI + MCS + date-clustered inference (fixes G3).
7. Export row-aligned predictions + write the feature-availability manifest (fixes G4).

No critical leakage that invalidates existing results was found; F1 is a mandated control whose effect is
horizon-proportional and is fixed in the new study before any beat-HAR claim.
