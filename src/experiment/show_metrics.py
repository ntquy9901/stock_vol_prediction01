"""
Display current metrics values for all 30 stocks.
"""

import pandas as pd
import json

# Load summary report
df = pd.read_csv('results/summary_report.csv')

# Select key metrics columns
metrics_cols = ['ticker', 'test_qlike_loss', 'test_directional_accuracy', 'test_theil_u', 'test_rmse', 'test_r2']
metrics_df = df[metrics_cols].copy()

# Format for display
metrics_df['test_qlike_loss'] = metrics_df['test_qlike_loss'].apply(lambda x: f'{x:.6f}')
metrics_df['test_directional_accuracy'] = metrics_df['test_directional_accuracy'].apply(lambda x: f'{x*100:.1f}%')
metrics_df['test_theil_u'] = metrics_df['test_theil_u'].apply(lambda x: f'{x:.4f}')
metrics_df['test_rmse'] = metrics_df['test_rmse'].apply(lambda x: f'{x:.6f}')
metrics_df['test_r2'] = metrics_df['test_r2'].apply(lambda x: f'{x:.4f}')

# Rename columns for Vietnamese display
metrics_df.columns = ['Ma CP', 'QLIKE Loss', 'Do chinh xac huong', 'Theil U', 'RMSE', 'R2']

print('=' * 100)
print('CAC CHI SO HIEN TAI CUA HE THONG DU BAO CHO 30 CO PHIEU VN30')
print('=' * 100)
print(metrics_df.to_string(index=False))
print('=' * 100)

# Show aggregate statistics
print('\nTONG HET HE THONG (30 co phieu):')
print('=' * 100)
agg_stats = json.load(open('results/aggregate_results.json'))['summary_stats']
print(f"QLIKE Loss trung binh:     {agg_stats['mean_qlike_loss']:.6f}")
print(f"Do chinh xac huong TB:     {agg_stats['mean_directional_accuracy']*100:.1f}%")
print(f"Theil U trung binh:         {agg_stats['mean_theil_u']:.4f}")
print(f"Do lech chuan QLIKE:        {agg_stats['std_qlike_loss']:.6f}")
print(f"Co phieu beating RW:       {agg_stats['stocks_beating_rw']}/30 ({agg_stats['stocks_beating_rw']/30*100:.1f}%)")
print('=' * 100)