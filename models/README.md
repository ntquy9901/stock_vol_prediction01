# Models Directory

This directory contains trained models organized by baseline.

## Directory Structure

```
models/
├── har_baseline/           # HAR-R baseline models
│   ├── ACB_har_r_model.pkl
│   ├── BCM_har_r_model.pkl
│   └── ... (30 stocks total)
├── lstm_baseline/          # LSTM baseline models
│   ├── VCB_lstm_model.pth
│   ├── VHM_lstm_model.pth
│   ├── VIC_lstm_model.pth
│   └── archive/            # Old/experimental LSTM models
└── [other_baseline]/       # Future baselines
```

## Model Naming Convention

**Format:** `{TICKER}_{baseline}_model.{ext}`

Examples:
- ✅ `ACB_har_r_model.pkl` - HAR-R model for ACB
- ✅ `VCB_lstm_model.pth` - LSTM model for VCB
- ❌ `model.pkl` (too generic)

## Model Versions

For multiple versions of the same model:
```
lstm_baseline/
├── VCB_lstm_model_v1.pth
├── VCB_lstm_model_v2.pth
└── VCB_lstm_model_latest.pth -> VCB_lstm_model_v2.pth
```

## Best Models

Best performing models should be symlinked or copied with clear naming:
```
lstm_baseline/
├── best_simple_lstm.pth -> simple_lstm_2026-06-17.pth
└── simple_lstm_2026-06-17.pth
```

## Loading Models

```python
import joblib
import torch

# Load HAR-R model
model = joblib.load('models/har_baseline/ACB_har_r_model.pkl')

# Load LSTM model
model = torch.load('models/lstm_baseline/VCB_lstm_model.pth')
```

## Cleanup

- Archive old models to `{baseline}/archive/`
- Keep only best performing models in main folders
- Document model performance in README or metadata
