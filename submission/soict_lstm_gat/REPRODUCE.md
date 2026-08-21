# Reproduce (for reviewers)

## Environment
Python 3.10/3.11 + GPU (CUDA) recommended. `pip install -r requirements.txt`
(torch, numpy, pandas, scikit-learn, scipy, arch, matplotlib, pytest).

## Tests
```
python -m pytest tests -q          # 36 tests (metrics, data_utils, edges, baselines, model, snapshots, run_lstm)
```

## Main model — per-observation LSTM vs HAR + GARCH (headline result)
The headline model is the price-only **LSTM** trained on per-observation pooled windows (every
ticker × window), per-stock chronological 80/10/10, MSE loss, early-stop on val MSE. This is the
data design under which the LSTM is competitive with / beats HAR at short horizons.
From this folder (`submission/soict_lstm_gat/`); writes `results/soict_perobs/<name>/result.json`:
```
# dataset lookback horizon
python run_lstm.py vn30  10 1 --data-root data
python run_lstm.py vn30  10 5 --data-root data
python run_lstm.py vn100 10 1 --data-root data
python run_lstm.py vn100 10 5 --data-root data
```
Each run trains the LSTM over 5 seeds (20 epochs, early-stop), computes HAR + GARCH baselines,
all five metrics (MSE/RMSE/MAE/QLIKE/R²) and Diebold–Mariano (LSTM vs HAR / GARCH) on QLIKE + SE.
Learning-curve PNGs are written every 5 epochs. Add `--smoke` for a fast 2-epoch / 1-seed check.

## Graph ablation — HAR-LSTM-GAT vs LSTM (w/o GAT)
This is a leave-one-out **graph-check** ablation on common-date snapshots (a GAT needs per-date
snapshots), not the headline design. Writes `results/soict/<name>/result.json`:
```
python run_all.py vn30  10 1 --data-root data
python run_all.py vn30  10 5 --data-root data
python run_all.py vn30  22 1 --data-root data      # variation: lookback 22
python run_all.py vn30  22 5 --data-root data
python run_all.py vn100 10 1 --data-root data      # variation: VN100
python run_all.py vn100 10 5 --data-root data
```
Each run trains HAR-LSTM-GAT + LSTM (w/o GAT) over 5 seeds and runs DM (Ours vs HAR / GARCH / w/o-GAT).

Or run everything (tests + both suites) with `./reproduce.sh`.

## S&P500 (not shipped; Yahoo-derived)
S&P500 processed data is not redistributed. To regenerate from Yahoo Finance (yfinance) then run
both models:
```
python -m src.data.download_sp500 data/raw/prices/sp500        # from the main repo root
python -m src.common.process_parkinson_pipeline --raw data/raw/prices/sp500 --out submission/soict_lstm_gat/data/sp500
python run_lstm.py sp500 10 1 --data-root data                 # main model (no batch limit needed)
python run_all.py  sp500 10 1 --data-root data --batch 16      # graph model: small batch, GAT attention is O(N^2)
```

## Notes
- QLIKE positivity floor 1e-8 is identical across all compared models. Graphical-lasso edges are
  estimated on TRAINING rows only and frozen. Per-ticker scalers are fit on TRAINING rows only.
- Seeds are fixed (config.py `seeds=(42,123,2026,7,2024)`; `torch.manual_seed`/`np.random.seed`
  set per seed). Decision metric = QLIKE + Diebold–Mariano. Training loss = MSE.
