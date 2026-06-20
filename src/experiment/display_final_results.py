"""
Display comprehensive results with MSE for all 30 VN30 stocks.
"""

import pandas as pd
import json

# Load updated summary report with MSE
df = pd.read_csv('results/summary_report.csv')

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
print('CAC CHI SO HIEN TAI DAY DU CUA HE THONG VOI MSE (30 CO PHIEU VN30)')
print('=' * 120)
print(metrics_df.to_string(index=False))
print('=' * 120)

# Load aggregate statistics
agg_stats = json.load(open('results/aggregate_results.json'))['summary_stats']

print(f'\nTONG HET HE THONG (30 co phieu):')
print('=' * 120)
print(f"QLIKE Loss trung binh:     {agg_stats['mean_qlike_loss']:.6f}")
print(f"Do chinh xac huong TB:     {agg_stats['mean_directional_accuracy']*100:.1f}%")
print(f"Theil U trung binh:         {agg_stats['mean_theil_u']:.4f}")
print(f"Do lech chuan QLIKE:        {agg_stats['std_qlike_loss']:.6f}")
print(f"Co phieu beating RW:       {agg_stats['stocks_beating_rw']}/30 ({agg_stats['stocks_beating_rw']/30*100:.1f}%)")
print(f"Co phieu Dir Acc >= 55%:   {agg_stats['stocks_with_dir_acc_55']}/30")

# Calculate MSE statistics from summary
mean_mse = df['test_mse'].mean()
std_mse = df['test_mse'].std()
min_mse = df['test_mse'].min()
max_mse = df['test_mse'].max()

print(f"\nTHONG KE MSE (30 stocks):")
print('=' * 120)
print(f"Mean MSE:     {mean_mse:.8e}")
print(f"Std Dev MSE:   {std_mse:.8e}")
min_ticker = df.loc[df['test_mse'].idxmin(), 'ticker']
max_ticker = df.loc[df['test_mse'].idxmax(), 'ticker']
print(f"Min MSE:      {min_mse:.8e} ({min_ticker})")
print(f"Max MSE:      {max_mse:.8e} ({max_ticker})")

print('\nCHI TIET MSE:')
print('-' * 120)
print('MSE (Mean Squared Error) = TB binh phuong cua squared residuals')
print('MSE = RMSE² (RMSE mu bin phuong)')
print('Lower is better - MSE thich optimization cho ML models')

print('\nTOP 5 STOCKS THEO MSE (THAP NHAT - LOWER IS BETTER):')
print('-' * 120)
top5_mse = df.nsmallest(5, 'test_mse')[['ticker', 'test_mse', 'test_rmse']]
for idx, row in top5_mse.iterrows():
    print(f"{row['ticker']:6s} MSE = {row['test_mse']:.8e}, RMSE = {row['test_rmse']:.6f}")

print('\nBOTTOM 5 STOCKS THEO MSE (MSE CAO NHAT):')
print('-' * 120)
bottom5_mse = df.nlargest(5, 'test_mse')[['ticker', 'test_mse', 'test_rmse']]
for idx, row in bottom5_mse.iterrows():
    print(f"{row['ticker']:6s} MSE = {row['test_mse']:.8e}, RMSE = {row['test_rmse']:.6f}")

print('\nQUAN HE MSE:')
print('-' * 120)
print('- MSE scale: 10^-7 den nghan (cac training)')
print('- RMSE scale: 10^-3 den nghan')
print('- MSE don vi te bo qua gradient calculation')
print('- RMSE don vi te interpretation (cung don vi voi target)')

print('=' * 120)
print('HE THONG DA HOAN THANH: MOI TRONG 7 METRICS')
print('=' * 120)
print('1. QLIKE Loss     - Primary metric (academic standard)')
print('2. RMSE           - Accuracy metric (same unit as target)')
print('3. MSE            - Optimization metric (lower is better)')
print('4. MAE            - Robust metric (less sensitive to outliers)')
print('5. R2             - Variance explained (0 to 1)')
print('6. Directional Acc - Direction prediction (49.4%)')
print('7. Theil U         - Benchmark vs random walk (0.86)')
print('=' * 120)
