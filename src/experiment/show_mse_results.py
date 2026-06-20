"""
Display MSE metric demonstration.
"""

import pandas as pd
import sys
from pathlib import Path

# Try to load recent test results
test_results_path = Path('results_test_mse/summary_report.csv')
main_results_path = Path('results/summary_report.csv')

if test_results_path.exists():
    df = pd.read_csv(test_results_path)
    print("Using test results (3 stocks: VCB, VIC, VHM)\n")
elif main_results_path.exists():
    df = pd.read_csv(main_results_path)
    # Check if MSE column exists
    if 'test_mse' not in df.columns:
        print("MSE not in current results. Current results use older pipeline version.")
        print("\nCurrent columns:", df.columns.tolist())
        print("\nNOTE: MSE has been successfully added to the evaluation system!")
        print("Re-run the pipeline to get MSE values for all 30 stocks.")
        sys.exit(0)
    else:
        print("Using main results (30 stocks)\n")
else:
    print("No results found. Run pipeline first.")
    sys.exit(0)

# Select all metrics columns including MSE
metrics_cols = ['ticker', 'test_qlike_loss', 'test_directional_accuracy', 'test_theil_u', 'test_rmse', 'test_mse', 'test_r2']
metrics_df = df[metrics_cols].copy()

# Format for display
metrics_df['test_qlike_loss'] = metrics_df['test_qlike_loss'].apply(lambda x: f'{x:.6f}')
metrics_df['test_directional_accuracy'] = metrics_df['test_directional_accuracy'].apply(lambda x: f'{x*100:.1f}%')
metrics_df['test_theil_u'] = metrics_df['test_theil_u'].apply(lambda x: f'{x:.4f}')
metrics_df['test_rmse'] = metrics_df['test_rmse'].apply(lambda x: f'{x:.6f}')
metrics_df['test_mse'] = metrics_df['test_mse'].apply(lambda x: f'{x:.8e}')
metrics_df['test_r2'] = metrics_df['test_r2'].apply(lambda x: f'{x:.4f}')

# Rename columns for Vietnamese display
metrics_df.columns = ['Ma CP', 'QLIKE Loss', 'Do chinh xac huong', 'Theil U', 'RMSE', 'MSE', 'R2']

print('=' * 120)
print('CAC CHI SO HIEN TAI CUA HE THONG VOI MSE')
print('=' * 120)
print(metrics_df.to_string(index=False))
print('=' * 120)

# Show MSE = RMSE² relationship
print('\nKHOAINGH MSE = RMSE²:')
print('-' * 120)
for idx, row in metrics_df.iterrows():
    # Get original values from dataframe
    ticker = row['Ma CP']
    rmse = float(df.loc[df['ticker'] == ticker, 'test_rmse'].values[0])
    mse = float(df.loc[df['ticker'] == ticker, 'test_mse'].values[0])
    rmse_squared = rmse ** 2

    print(f"{ticker}: RMSE = {rmse:.6f}, RMSE² = {rmse_squared:.8e}, MSE = {mse:.8e}")
    print(f"         -> MSE = RMSE²: {'CHINH XAC' if abs(rmse_squared - mse) < 1e-10 else 'KHAC BANG'}")

print('\nTHONG TIN MSE:')
print('-' * 120)
print('- MSE (Mean Squared Error) la MSE of squared residuals')
print('- MSE = RMSE² (RMSE binh phuong)')
print('- MSE don vi te: lower is better')
print('- MSE co gia tri trong optimization (khong can ve scale)')

print('\nCAC METRICS TRONG HE THONG:')
print('-' * 120)
print('1. QLIKE Loss     - Primary metric (academic standard cho volatility)')
print('2. RMSE           - Accuracy metric (same units as target)')
print('3. MSE            - Squared error (good cho optimization)')
print('4. MAE            - Robust error metric')
print('5. R2             - Variance explained')
print('6. Directional Acc - Direction prediction accuracy')
print('7. Theil U         - Benchmark vs random walk')
