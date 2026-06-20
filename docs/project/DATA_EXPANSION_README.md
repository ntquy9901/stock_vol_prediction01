# Data Expansion Guide - Vietnam Stock Market

**Date:** 2026-06-19
**Purpose:** Expand training data from 30 VN30 stocks to VN100, HNX, and combined datasets

---

## Overview

Currently the project has only **30 VN30 stocks**. This guide helps expand to:
- **VN100**: 100 stocks from Vietnam's 100 largest listed companies
- **HNX**: Stocks from Hanoi Stock Exchange
- **Combined datasets**: Multiple training scenarios

---

## New Directory Structure

```
data/raw/
├── vn30/                    # Existing: 30 VN30 stocks
│   ├── ACB_ohlcv.csv
│   └── stock_summary.csv
├── vn100/                   # NEW: 100 VN100 stocks
│   ├── [100 stocks]
│   └── stock_summary.csv
├── hnx/                     # NEW: HNX stocks
│   ├── [HNX stocks]
│   └── stock_summary.csv
├── combined/                # NEW: VN30 + VN100
│   └── stock_summary.csv
└── all/                     # NEW: VN30 + VN100 + HNX
    └── stock_summary.csv

configs/
└── training_config.yaml     # NEW: Training configuration
```

---

## Step 1: Crawl Additional Data

### Method 1: Using Yahoo Finance (Free, Limited Coverage)

```bash
# Install yfinance
pip install yfinance pyyaml

# Crawl all datasets
python src/data/crawl_vietnam_stocks.py
```

**Expected Output:**
- `data/raw/vn30/` - 30 stocks
- `data/raw/vn100/` - ~100 stocks (some may fail)
- `data/raw/hnx/` - ~HNX stocks (some may fail)

**Limitations:**
- Yahoo Finance doesn't cover all Vietnamese stocks
- Some symbols may return empty data
- For production, use Vietnamese data APIs

---

### Method 2: Using Vietnamese Data APIs (Production-Grade)

**Vietnamese Data Sources:**
1. **Fintext Securities** - https://fintext.vn
2. **Vietstock** - https://www.vietstock.vn
3. **Cafef** - https://cafef.vn
4. **SSI Data API** - Contact SSI Securities

**VN100 Stock List (2026):**
Reference: https://www.vietnamstock.com.vn/

```python
# Top VN100 stocks (examples)
VN100_STOCKS = [
    # Banks
    "VCB", "TCB", "MBB", "BID", "CTG", "ACB", "STB", "VPB", "HDB",

    # Real Estate
    "VHM", "VIC", "VRE", "DXG", "KDH", "NLG", "PHR", "HDG",

    # Technology
    "FPT", "CMG", "ELC", "FOX", "LCM",

    # Retail
    "MWG", "PC1", "DGW", "FRT", "PPG",

    # Manufacturing
    "HPG", "HSG", "NKG", "POM", "HCD",

    # Energy
    "GAS", "PLX", "PVD", "PVT", "POW",

    # Consumer Goods
    "VNM", "SAB", "VHC", "HNG", "KDC",

    # Healthcare
    "DHG", "PMC", "IMP", "TKV",

    # And 60+ more...
]
```

---

### Method 3: Using Existing vn30 data + Download More

```bash
# Combine existing vn30 with newly downloaded data
python src/data/combine_datasets.py \
    --sources vn30 vn100_new \
    --output combined
```

---

## Step 2: Combine Datasets

### Create Combined Datasets

```bash
# Combine VN30 + VN100
python src/data/combine_datasets.py \
    --sources vn30 vn100 \
    --output combined

# Combine ALL (VN30 + VN100 + HNX)
python src/data/combine_datasets.py \
    --sources vn30 vn100 hnx \
    --output all
```

**Output:**
- `data/raw/combined/` - Combined VN30+VN100
- `data/raw/all/` - All stocks combined

---

## Step 3: Update Training Configuration

Edit `configs/training_config.yaml`:

```yaml
data_scenarios:
  vn30_only:
    data_path: "data/raw/vn30"
    num_stocks: 30

  vn100_only:
    data_path: "data/raw/vn100"
    num_stocks: 100

  all_combined:
    data_path: "data/raw/all"
    combine_from:
      - "data/raw/vn30"
      - "data/raw/vn100"
      - "data/raw/hnx"
```

---

## Step 4: Train with New Data

### Option 1: Using Config Script (Recommended)

```bash
# Train on VN100 only
python src/train_with_config.py \
    --scenario vn100_only \
    --model enhanced_lstm_har

# Train on ALL combined data
python src/train_with_config.py \
    --scenario all_combined \
    --model enhanced_lstm_har

# Train LSTM-GAT on all data
python src/train_with_config.py \
    --scenario all_combined \
    --model lstm_gat
```

### Option 2: Using Existing Scripts (Modify Data Path)

```bash
# Modify src/lstm_har_enhanced/train_with_validation.py
# Change DATA_DIR = "data/raw/vn100"  # or "data/raw/all"

python src/lstm_har_enhanced/train_with_validation.py
```

---

## Training Scenarios

### Scenario 1: Baseline Comparison
Compare same model on different data sizes:

```bash
# VN30 only (30 stocks)
python src/train_with_config.py --scenario vn30_only --model enhanced_lstm_har

# VN100 only (100 stocks)
python src/train_with_config.py --scenario vn100_only --model enhanced_lstm_har

# Combined (VN30+VN100, ~100 stocks)
python src/train_with_config.py --scenario vn30_vn100_combined --model enhanced_lstm_har

# All (VN30+VN100+HNX, ~150+ stocks)
python src/train_with_config.py --scenario all_combined --model enhanced_lstm_har
```

### Scenario 2: Model Comparison
Compare different models on same data:

```bash
# Test all models on VN100
python -m src.experiment.model_comparison \
    --data_path data/raw/vn100 \
    --models har_linear simple_lstm lstm_har enhanced_lstm_har
```

### Scenario 3: LSTM-GAT (Requires Most Data)
LSTM-GAT needs maximum data for cross-stock learning:

```bash
# Only recommended with all_combined scenario
python src/train_with_config.py \
    --scenario all_combined \
    --model lstm_gat
```

---

## Expected Performance Improvements

### Current (VN30, 30 stocks)
- **RMSE:** 0.18
- **Directional Accuracy:** 67.90%
- **R²:** ~0.65

### Expected (VN100, 100 stocks)
- **RMSE:** 0.17 → 0.16 (5-10% improvement)
- **Directional Accuracy:** 67.90% → 70-72% (2-4% improvement)
- **R²:** ~0.65 → 0.70-0.72 (5-10% improvement)

### Expected (All Combined, 150+ stocks)
- **RMSE:** 0.18 → 0.15 (15-20% improvement)
- **Directional Accuracy:** 67.90% → 73-75% (5-7% improvement)
- **R²:** ~0.65 → 0.75-0.78 (15-20% improvement)

**LSTM-GAT with All Data (Target):**
- **RMSE:** < 0.15
- **Directional Accuracy:** > 75%
- **R²:** > 0.75

---

## Troubleshooting

### Issue: Some stocks fail to crawl

**Solution:** Yahoo Finance doesn't cover all Vietnamese stocks.

1. Check `data/raw/vn100/stock_summary.csv` to see which stocks succeeded
2. Use Vietnamese data APIs for missing stocks
3. Or proceed with available stocks (update `num_stocks` in config)

### Issue: Data format mismatch

**Solution:** Ensure all CSV files have same format:
```csv
date,open,high,low,close,volume
2006-11-21,3.86,4.83,3.83,4.34,56500
```

### Issue: Memory issues with large datasets

**Solution:**
1. Increase batch size in config: `batch_size: 64`
2. Use gradient accumulation
3. Train on GPU: `device: "cuda"`

### Issue: No improvement with more data

**Possible causes:**
1. **Data quality:** New stocks may have poor data quality
2. **Overfitting:** Model may need more regularization
3. **Data leakage:** Ensure temporal split is used (not random!)
4. **Model capacity:** Simple models may not benefit from more data

**Solutions:**
- Check data quality: `python src/experiment/check_data_quality.py`
- Increase dropout: `dropout: 0.3`
- Use LSTM-GAT for spatial relationships
- Ensure temporal split: `use_temporal_split: true`

---

## Next Steps

1. **Crawl data:**
   ```bash
   python src/data/crawl_vietnam_stocks.py
   ```

2. **Combine datasets:**
   ```bash
   python src/data/combine_datasets.py --sources vn30 vn100 hnx --output all
   ```

3. **Train baseline:**
   ```bash
   python src/train_with_config.py --scenario vn100_only --model enhanced_lstm_har
   ```

4. **Compare results:**
   ```bash
   python src/experiment/compare_results.py \
       --results_dir results/
   ```

5. **Train LSTM-GAT:**
   ```bash
   python src/train_with_config.py --scenario all_combined --model lstm_gat
   ```

---

## Additional Vietnamese Stock Indices

### HNX (Hanoi Stock Exchange)
- **HNX30:** Top 30 stocks on HNX
- **HNX Index:** Overall HNX market

### UPCoM (Unlisted Public Company Market)
- **UPoM:** Stocks not yet listed on HOSE/HNX
- Usually smaller companies

### Future Expansion
- **VNMidcap:** Mid-sized companies
- **VNSmallcap:** Small companies
- **Sector-specific:** Banking, Real Estate, Technology

---

## Production Data Sources

**For production/real-time trading, use:**

1. **Fintext Securities API**
   - Real-time data
   - Historical OHLCV
   - Corporate actions

2. **Vietstock**
   - Comprehensive data
   - News & events
   - Financial statements

3. **Cafef**
   - Real-time quotes
   - Market overview
   - Technical indicators

4. **Proprietary Data**
   - Contact brokerages (SSI, VPS, VCSC)
   - Direct exchange feeds

---

**Last Updated:** 2026-06-19
**Status:** Ready for data crawling
