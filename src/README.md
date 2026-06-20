# Source Code Structure

This directory contains all source code for the volatility prediction project.

## Directory Organization

### `src/common/`
Shared utilities and functions used across multiple baselines:

- **data_processing.py**: Parkinson volatility calculation, OHLCV validation
- **evaluation.py**: QLIKE, RMSE, MAE, directional accuracy metrics
- **process_data.py**: Main data processing script

### `src/lstm_baseline/`
LSTM model implementation and training:

- **model.py**: SimpleVolatilityLSTM class (1-layer, 32 hidden units)
- **dataset.py**: PooledVolatilityDataset class (pool approach for 30 stocks)
- **train.py**: Core training logic with proper scaling
- **train_simple_lstm.py**: Main training entry point

### `src/har_baseline/`
HAR-R baseline model (to be implemented):

- **model.py**: HAR-R regression model
- **train.py**: HAR-R training logic
- **features.py**: HAR feature engineering

### `src/experiment/`
Experimental and exploratory scripts:

- **debug_*.py**: Debug scripts for development
- **test_*.py**: Temporary test scripts
- **analysis_*.py**: Data analysis scripts

## Import Conventions

When writing code in subdirectories:

```python
# Add project root to path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Import from other modules
from src.common.evaluation import evaluate_predictions
from src.lstm_baseline.model import SimpleVolatilityLSTM
```

## Running Scripts

From project root:
```bash
python -m src.common.process_data
python -m src.lstm_baseline.train_simple_lstm
```

Or from subdirectory:
```bash
cd src/lstm_baseline
python train_simple_lstm.py
```

## Code Style

- Follow ML/DS common rules (see `docs/common-rules/`)
- Update existing files, don't create new ones unless necessary
- Common code goes in `src/common/`
- Baseline-specific code stays in respective baseline directories
