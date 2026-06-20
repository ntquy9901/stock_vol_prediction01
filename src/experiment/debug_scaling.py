"""Debug scaling issue"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load sample data
df = pd.read_csv('data/processed/ACB_processed.csv')
volatility = df['parkinson_volatility'].values[:100]

print("Original volatility range:")
print(f"  Min: {volatility.min():.6f}")
print(f"  Max: {volatility.max():.6f}")
print(f"  Mean: {volatility.mean():.6f}")

# Create target scaler
target_scaler = StandardScaler()
volatility_reshaped = volatility.reshape(-1, 1)
target_scaler.fit(volatility_reshaped)

# Scale and inverse
volatility_scaled = target_scaler.transform(volatility_reshaped).flatten()
volatility_restored = target_scaler.inverse_transform(volatility_scaled.reshape(-1, 1)).flatten()

print("\nScaled volatility range:")
print(f"  Min: {volatility_scaled.min():.6f}")
print(f"  Max: {volatility_scaled.max():.6f}")
print(f"  Mean: {volatility_scaled.mean():.6f}")

print("\nRestored volatility range:")
print(f"  Min: {volatility_restored.min():.6f}")
print(f"  Max: {volatility_restored.max():.6f}")
print(f"  Mean: {volatility_restored.mean():.6f}")

print("\nDifference between original and restored:")
print(f"  Max error: {np.abs(volatility - volatility_restored).max():.10f}")
