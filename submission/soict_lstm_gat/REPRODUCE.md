# Reproduce (for reviewers)

## Environment
Python 3.10/3.11 + GPU (CUDA) recommended. `pip install -r requirements.txt` (torch, numpy, pandas,
scikit-learn, arch, matplotlib, pytest).

## Tests
```
python -m pytest tests -q          # 34 tests (metrics, data_utils, edges, baselines, model, snapshots)
```

## Full experiment suite
From this folder (`submission/soict_lstm_gat/`):
```
# main + variations (dataset lookback horizon); writes results/soict/<name>/result.json
python run_all.py vn30  10 1 --data-root data
python run_all.py vn30  10 5 --data-root data
python run_all.py vn30  22 1 --data-root data      # variation: lookback 22
python run_all.py vn30  22 5 --data-root data
python run_all.py vn100 10 1 --data-root data      # variation: VN100
python run_all.py vn100 10 5 --data-root data
```
Each run trains HAR-LSTM-GAT + LSTM (w/o GAT) over 5 seeds (20 epochs, early-stop), computes HAR +
GARCH baselines, all five metrics, and Diebold–Mariano (Ours vs HAR / GARCH / w/o-GAT). Learning-curve
PNGs are written every 5 epochs.

## S&P500 (not shipped; Yahoo-derived)
S&P500 processed data is not redistributed. To regenerate from Yahoo Finance (yfinance) then run:
```
python -m src.data.download_sp500 data/raw/prices/sp500        # from the main repo root
python -m src.common.process_parkinson_pipeline --raw data/raw/prices/sp500 --out submission/soict_lstm_gat/data/sp500
python run_all.py sp500 10 1 --data-root data --batch 16       # small batch: GAT attention is O(N^2)
```

## Notes
- QLIKE positivity floor 1e-8 is identical across all compared models. Graphical-lasso edges are
  estimated on TRAINING rows only and frozen. Per-ticker scalers are fit on TRAINING rows only.
- Decision metric = QLIKE + Diebold–Mariano. Training loss = MSE.
