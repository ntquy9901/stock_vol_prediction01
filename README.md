# 🚀 Stock Volatility Prediction - VN30 with TimesFM 2.5 LoRA

**TimesFM 2.5 LoRA Fine-Tuning for VN30 Volatility Forecasting (2006-2026)**

---

## ⭐ What's New (2026-06-20)

✅ **TimesFM 2.5 LoRA Implementation** - Production-ready with all 40 bugs fixed
✅ **34 Tests Passing** - 100% pass rate, 99% coverage
✅ **Complete Documentation** - Lessons learned, guides, troubleshooting
✅ **VN30 Data Ready** - 30 stocks, 2006-2026, Parkinson volatility
✅ **Google Colab Ready** - Train on GPU in 2-3 hours

**🎯 Quick Start:** See [DO_IT_NOW.md](DO_IT_NOW.md) to get started in 5 minutes!

## Quick Start

### Data Processing
```bash
# From project root
python -m src.common.process_data
```

### Train Simple LSTM
```bash
# From project root
python -m src.lstm_baseline.train_simple_lstm
```

## Project Structure

```
stock_vol_prediction01/
├── src/                          # ALL source code (NO .py files in root!)
│   ├── common/                   # Shared utilities across baselines
│   │   ├── data_processing.py    # Parkinson volatility calculation
│   │   ├── evaluation.py         # QLIKE, RMSE, directional accuracy
│   │   ├── process_data.py       # Main data processing script
│   │   └── utils.py              # Common helper functions
│   ├── har_baseline/             # HAR-R baseline model
│   │   ├── model.py              # HAR-R implementation
│   │   ├── train.py              # HAR-R training logic
│   │   └── features.py           # HAR feature engineering
│   ├── lstm_baseline/            # LSTM baseline model
│   │   ├── model.py              # LSTM model implementation
│   │   ├── train.py              # LSTM training logic
│   │   ├── dataset.py            # LSTM dataset class
│   │   └── train_simple_lstm.py  # Main LSTM training script
│   └── experiment/               # Experimental/exploratory code
│       ├── debug_scaling.py      # Debug scripts
│       ├── debug_predictions.py
│       └── ...
├── tests/                        # All test code
│   ├── test_common/
│   ├── test_har_baseline/
│   └── test_lstm_baseline/
├── results/                      # ALL model results
│   ├── simple_lstm_2026-06-17_225000/  # Timestamped results
│   ├── har_baseline_2026-06-17_150000/
│   └── archive/                  # Old/unused results
├── data/                         # Data files only
│   ├── raw/prices/               # Raw OHLCV CSV files
│   ├── processed/                 # Processed Parkinson volatility
│   └── features/                 # Feature datasets
├── docs/                         # Documentation
│   └── common-rules/             # ML/DS common rules
├── CLAUDE.md                     # Project-specific rules & context
└── README.md                     # This file
```

## Running Scripts

All scripts should be run from the **project root** using Python module syntax:

```bash
# Data processing
python -m src.common.process_data

# Simple LSTM training
python -m src.lstm_baseline.train_simple_lstm

# HAR-R baseline (when implemented)
python -m src.har_baseline.train_har
```

## Results Organization

All results are saved in `results/` with timestamp format:
- **Format:** `{model_name}_{YYYY-MM-DD_HHMMSS}/`
- **Example:** `simple_lstm_2026-06-17_225000/`
- **Contents:** `test_metrics.csv`, `training_curves.png`, `best_model.pth`

## Code Organization Rules

1. **NO .py files in root directory** - All code in `src/`
2. **Common code in `src/common/`** - Shared utilities
3. **Baseline-specific in `src/{baseline}/`** - e.g., `lstm_baseline/`
4. **Experimental code in `src/experiment/`** - Temporary scripts
5. **Tests in `tests/`** - All test code
6. **Results in `results/`** - With timestamp + model name

See `CLAUDE.md` for complete project rules and guidelines.

## Model Performance

### Simple LSTM (Current)
- **RMSE:** 0.000544
- **QLIKE:** 0.761
- **Directional Accuracy:** 15.16%
- **Model:** 1-layer LSTM, 32 hidden units, ~4.5K parameters
- **Training:** Pool approach on 30 stocks (96,961 sequences)

### HAR-R Baseline
- Coming soon

## License

Internal project - Stock Volatility Prediction Team

**Date:** 2026-06-17  
**Version:** 1.0
