"""
Train All Models on VN30 Stocks Only

This script trains and evaluates all models on 30 VN30 stocks only,
providing a clean, focused comparison for Vietnam's blue-chip index.

Models to train:
1. HAR-R Linear Baseline
2. Simple LSTM
3. LSTM-HAR
4. Enhanced LSTM-HAR

Output: Comprehensive VN30-only performance report
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def get_vn30_symbols():
    """Get list of 30 VN30 stock symbols"""
    vn30_summary = project_root / 'data' / 'raw' / 'vn30' / 'stock_summary.csv'

    if not vn30_summary.exists():
        print(f"[ERROR] VN30 stock summary not found: {vn30_summary}")
        return None

    df = pd.read_csv(vn30_summary)
    vn30_symbols = df['symbol'].tolist()

    print(f"Found {len(vn30_symbols)} VN30 stocks:")
    print(", ".join(vn30_symbols))

    return vn30_symbols


def prepare_vn30_data(vn30_symbols):
    """
    Prepare processed data for VN30 stocks only

    Creates a filtered dataset containing only VN30 stocks
    """
    processed_dir = project_root / 'data' / 'processed'
    vn30_dir = processed_dir / 'vn30_only'

    # Create output directory
    vn30_dir.mkdir(exist_ok=True)

    print(f"\nPreparing VN30-only dataset...")
    print(f"Source: {processed_dir}")
    print(f"Target: {vn30_dir}")

    vn30_files = []
    missing_files = []

    for symbol in vn30_symbols:
        source_file = processed_dir / f"{symbol}_processed.csv"
        target_file = vn30_dir / f"{symbol}_processed.csv"

        if source_file.exists():
            # Copy file to vn30_only directory (or create symlink)
            import shutil
            shutil.copy2(source_file, target_file)
            vn30_files.append(target_file)
            print(f"  [OK] {symbol}")
        else:
            missing_files.append(symbol)
            print(f"  [MISSING] {symbol}")

    print(f"\n[OK] Prepared {len(vn30_files)} VN30 stocks")
    if missing_files:
        print(f"[WARNING] Missing {len(missing_files)} stocks: {missing_files}")

    return vn30_dir, len(vn30_files)


def train_har_baseline_vn30(vn30_data_dir):
    """Train HAR-R baseline on VN30 data"""
    print("\n" + "="*80)
    print("1. TRAINING HAR-R LINEAR BASELINE (VN30)")
    print("="*80)

    from src.har_baseline.train import train_har_baseline

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = f'results/har_baseline_vn30_{timestamp}'

    model, metrics = train_har_baseline(vn30_data_dir, output_dir)

    return {
        'model': 'HAR-R Linear (VN30)',
        'output_dir': output_dir,
        'metrics': metrics
    }


def train_simple_lstm_vn30(vn30_data_dir):
    """Train Simple LSTM on VN30 data"""
    print("\n" + "="*80)
    print("2. TRAINING SIMPLE LSTM (VN30)")
    print("="*80)

    # Import Simple LSTM training
    from src.lstm_baseline.train_with_validation import main as train_simple_lstm

    # Need to modify training script to accept custom data directory
    # For now, use existing training with symlink
    print("  [TODO] Implement Simple LSTM training for VN30")

    return None


def train_lstm_har_vn30(vn30_data_dir):
    """Train LSTM-HAR on VN30 data"""
    print("\n" + "="*80)
    print("3. TRAINING LSTM-HAR (VN30)")
    print("="*80)

    # Import LSTM-HAR training
    from src.lstm_har_baseline.train_with_validation import main as train_lstm_har

    print("  [TODO] Implement LSTM-HAR training for VN30")

    return None


def train_enhanced_lstm_har_vn30(vn30_data_dir):
    """Train Enhanced LSTM-HAR on VN30 data"""
    print("\n" + "="*80)
    print("4. TRAINING ENHANCED LSTM-HAR (VN30)")
    print("="*80)

    # Import Enhanced LSTM-HAR training
    from src.lstm_har_enhanced.train_with_validation import main as train_enhanced

    print("  [TODO] Implement Enhanced LSTM-HAR training for VN30")

    return None


def create_vn30_report(results, vn30_stocks_count):
    """Create comprehensive VN30-only performance report"""
    print("\n" + "="*80)
    print("VN30-ONLY PERFORMANCE REPORT")
    print("="*80)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    report_file = f'results/VN30_Performance_Report_{timestamp}.md'

    report_content = f"""# VN30 Stocks Only - Performance Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Scope:** 30 VN30 Blue-Chip Stocks Only
**Purpose:** Clean comparison focused on Vietnam's premier index

---

## Dataset Information

**VN30 Stocks:** {vn30_stocks_count} stocks
**Data Source:** Yahoo Finance (processed with Parkinson volatility)
**Target:** 5-day ahead volatility forecasting

**VN30 Constituents:**
```json
{json.dumps(get_vn30_symbols() if callable(get_vn30_symbols) else [], indent=2)}
```

---

## Model Comparison (VN30 Only)

| Model | RMSE | MAE | R² | QLIKE | Dir Acc | Status |
|-------|------|-----|----|----|----------|---------|
"""

    # Add results from each model
    if results.get('har_baseline'):
        m = results['har_baseline']['metrics']
        report_content += f"| HAR-R Linear | {m['RMSE']:.6f} | {m['MAE']:.6f} | {m['R²']:.6f} | {m['QLIKE']:.6f} | {m['Directional_Acc']:.2f}% | Baseline |\n"

    # Add placeholders for other models
    report_content += """| Simple LSTM | TBD | TBD | TBD | TBD | TBD | ⚠️ Not trained |
| LSTM-HAR | TBD | TBD | TBD | TBD | TBD | ⚠️ Not trained |
| Enhanced LSTM-HAR | TBD | TBD | TBD | TBD | TBD | ⚠️ Not trained |

---

## Success Criteria (VN30 Focus)

| Criterion | Target | Current | Status | Gap |
|-----------|--------|---------|---------|-----|
| **RMSE** | < 0.20 | TBD | ⚠️ TBD | TBD |
| **Dir Acc** | > 55% | TBD | ⚠️ TBD | TBD |
| **R²** | > 0.50 | TBD | ⚠️ TBD | TBD |
| **QLIKE** | < 0.50 | TBD | ⚠️ TBD | TBD |

---

## Analysis & Insights

### VN30-Specific Characteristics

**Advantages of VN30-only training:**
- ✅ **Uniform quality** - All 30 stocks are blue-chip, liquid stocks
- ✅ **Clean comparison** - No noise from smaller, illiquid stocks
- ✅ **Market focus** - Direct relevance to Vietnam's key index
- ✅ **Stable data** - VN30 stocks have reliable, long histories

**Comparison with full dataset (208 stocks):**
- **Dataset size:** ~30 stocks vs 208 stocks
- **Data quality:** Higher quality (blue-chip only)
- **Market representation:** Core Vietnam market vs broad market
- **Expected performance:** Potentially better (less noise)

---

## Detailed Results

### HAR-R Linear Baseline (VN30)

**Output Directory:** `{results.get('har_baseline', {}).get('output_dir', 'N/A')}`

**Model Specification:**
```
target_5d = β₀ + β₁*har_daily_vol + β₂*har_weekly_vol + β₃*har_monthly_vol
```

**Performance Metrics:**
"""

    if results.get('har_baseline'):
        m = results['har_baseline']['metrics']
        report_content += f"""
```
MSE:  {m['MSE']:.6f}
RMSE: {m['RMSE']:.6f}
MAE:  {m['MAE']:.6f}
R²:   {m['R²']:.6f}
QLIKE: {m['QLIKE']:.6f}
Directional Accuracy: {m['Directional_Acc']:.2f}%
```

**Analysis:**
- RMSE: {"✅ Good (< 0.20)" if m['RMSE'] < 0.20 else "❌ Too high"}
- Directional Accuracy: {"✅ Good (> 55%)" if m['Directional_Acc'] > 55 else "❌ Below target"}
- R²: {"✅ Acceptable" if m['R²'] > 0.3 else "⚠️ Low explanatory power"}
"""

    report_content += """
---

## Recommendations

### For VN30-Specific Trading

**If using this model for VN30 trading:**
1. **Focus on best-performing models** - Use model with highest Dir Acc
2. **Monitor R² scores** - Ensure model explains variance
3. **Cross-validate** - Test on different time periods
4. **Risk management** - Use predictions as one signal, not sole decision

### Next Steps

**Immediate:**
- [ ] Train all models on VN30-only data
- [ ] Compare with full-dataset results
- [ ] Analyze performance differences
- [ ] Document VN30-specific insights

**Short-term:**
- [ ] Hyperparameter tuning for VN30
- [ ] Feature engineering for blue-chip stocks
- [ ] Ensemble methods combination
- [ ] Production deployment planning

---

## Files Generated

**Training Results:**
"""

    if results.get('har_baseline'):
        report_content += f"- `{results['har_baseline']['output_dir']}/` - HAR baseline results\n"

    report_content += f"""
**This Report:** `{report_file}`

---

**Report Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Next Review:** After all models trained
**Scope:** VN30 Blue-Chip Stocks Only
"""

    # Write report
    with open(report_file, 'w') as f:
        f.write(report_content)

    print(f"\n✓ VN30 Report saved to: {report_file}")

    return report_file


def main():
    """Main execution - Train all models on VN30 and create report"""
    print("\n" + "="*80)
    print("VN30-ONLY MODEL TRAINING & EVALUATION")
    print("="*80)

    # Step 1: Get VN30 symbols
    vn30_symbols = get_vn30_symbols()
    if vn30_symbols is None:
        print("[ERROR] Could not load VN30 symbols")
        return

    # Step 2: Prepare VN30-only data
    vn30_data_dir, vn30_count = prepare_vn30_data(vn30_symbols)

    if vn30_count < 25:
        print(f"[WARNING] Only {vn30_count} VN30 stocks available (expected 30)")
        print("Proceeding with available stocks...")

    # Step 3: Train models
    results = {}

    # HAR Baseline
    try:
        results['har_baseline'] = train_har_baseline_vn30(vn30_data_dir)
    except Exception as e:
        print(f"[ERROR] HAR baseline failed: {e}")

    # TODO: Add other models when ready
    # try:
    #     results['simple_lstm'] = train_simple_lstm_vn30(vn30_data_dir)
    # except Exception as e:
    #     print(f"[ERROR] Simple LSTM failed: {e}")

    # try:
    #     results['lstm_har'] = train_lstm_har_vn30(vn30_data_dir)
    # except Exception as e:
    #     print(f"[ERROR] LSTM-HAR failed: {e}")

    # try:
    #     results['enhanced_lstm_har'] = train_enhanced_lstm_har_vn30(vn30_data_dir)
    # except Exception as e:
    #     print(f"[ERROR] Enhanced LSTM-HAR failed: {e}")

    # Step 4: Create report
    if results:
        report_file = create_vn30_report(results, vn30_count)

        print("\n" + "="*80)
        print("VN30-ONLY TRAINING COMPLETE")
        print("="*80)
        print(f"\nResults:")
        for model_name, result in results.items():
            print(f"  ✓ {result['model']}")
            print(f"    Output: {result['output_dir']}")

        print(f"\n✓ Report: {report_file}")

    else:
        print("\n[ERROR] No models trained successfully")


if __name__ == "__main__":
    main()
