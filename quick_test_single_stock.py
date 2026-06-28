"""
Quick test - Train LSTM-GAT with single stock first
"""
import torch
import sys
sys.path.append('.')

# Test with just ACB stock
from src.lstm_har_enhanced.train_enhanced import main as train_lstm_har

print("Quick test: Train LSTM-HAR Enhanced first (single stock)")
print("This will give us a baseline before attempting LSTM-GAT Hybrid")
print("\nRun with: python quick_test_single_stock.py")
print("Then we can optimize LSTM-GAT for multi-stock processing")

train_lstm_har()
