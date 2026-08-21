# HAR-LSTM-GAT — SOICT submission (self-contained)

Proposed model **HAR-LSTM-GAT** for daily Parkinson-volatility forecasting, compared against **HAR**
and **GARCH** baselines, with a leave-one-out ablation (**LSTM w/o GAT**) and variation studies
(lookback 22, VN100). One pooled model, 80/10/10 chronological splits, 5 seeds, MSE training loss,
evaluation by MSE/RMSE/MAE/QLIKE/R² + Diebold–Mariano.

## Contents
- `config.py` — hyperparameters/seeds. `metrics.py`, `data_utils.py`, `edges.py` (graphical-lasso),
  `model.py` (HAR-LSTM-GAT), `snapshots.py`, `baselines.py` (HAR+GARCH), `train.py`, `evaluate.py`,
  `run_all.py`. `tests/` — 34 unit/smoke tests.
- `data/vn30/`, `data/vn100/` — SHIPPED processed data (`date, parkinson_volatility`; derived, not raw
  OHLCV). `data/sp500/` is not shipped (Yahoo-derived; regenerate — see REPRODUCE.md).

## Run
```
pip install -r requirements.txt
python run_lstm.py vn30 10 1 --data-root data     # MAIN model: per-observation LSTM vs HAR + GARCH
python run_all.py  vn30 10 1 --data-root data     # graph-check ablation: HAR-LSTM-GAT vs LSTM(w/o GAT)
```
The headline result is the per-observation **LSTM** (`run_lstm.py`); `run_all.py` is the leave-one-out
GAT graph-check ablation. Both write `result.json` + learning-curve PNGs (`run_lstm.py` →
`results/soict_perobs/`, `run_all.py` → `results/soict/`). See `REPRODUCE.md` for the full suite and `TASKBOARD.md`
for build/quality-gate status. Provenance: VN100 = vnstock; S&P500 = derived from Yahoo Finance
(non-commercial research use); raw OHLCV is not redistributed.

## Result (honest)
HAR is the strongest baseline; HAR-LSTM-GAT does not beat it and the GAT graph consistently hurts (the
ablation favours LSTM w/o GAT); all learned models beat GARCH. See
`docs/reports/2026-08-21_0141_soict_results_report.md` and `docs/paper/soict_harlstmgat_draft.md`.
