"""
Train Enhanced LSTM on S&P 500 with HAR + Market + Sentiment Features

Usage:
    python src/experiments/sp500/train_enhanced.py --tickers AAPL MSFT GOOGL
    python src/experiments/sp500/train_enhanced.py --feature_set full
    python src/experiments/sp500/train_enhanced.py --feature_set har  # baseline comparison
    python src/experiments/sp500/train_enhanced.py --forecast_horizon 10
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime

# Bootstrap path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from src.common.evaluation import evaluate_predictions
from src.common.multi_ticker_dataset import build_per_ticker_datasets


FEATURE_SETS = {
    "har": ["har_daily_vol", "har_weekly_vol", "har_monthly_vol"],
    "har_market": ["har_daily_vol", "har_weekly_vol", "har_monthly_vol", "vix", "treasury_10y", "sp500_index"],
    "full": ["har_daily_vol", "har_weekly_vol", "har_monthly_vol", "vix", "treasury_10y", "sp500_index",
             "sentiment_score", "sentiment_confidence", "news_count"],
}


class EnhancedLSTM(nn.Module):
    """LSTM with configurable input size."""

    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x).squeeze()
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_pooled(model, per_ticker, split, criterion, device, batch_size=64):
    """Evaluate a split across all tickers, inverse-transforming each ticker's
    predictions with ITS OWN target_scaler before pooling for metrics.

    Fixes the bug where a single (pooled) scaler was used to inverse-transform
    predictions from multiple tickers with different volatility scales.
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for ticker, splits in per_ticker.items():
            ds = splits[split]
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=batch_size)
            target_scaler = splits["target_scaler"]
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).squeeze(-1)
                loss = criterion(pred, y)
                total_loss += loss.item()
                n_batches += 1

                pred_np = target_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).flatten()
                true_np = target_scaler.inverse_transform(y.cpu().numpy().reshape(-1, 1)).flatten()
                all_true.extend(true_np)
                all_pred.extend(pred_np)

    avg_loss = total_loss / max(n_batches, 1)
    metrics = evaluate_predictions(np.array(all_true), np.array(all_pred))
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL"])
    parser.add_argument("--feature_set", default="full", choices=["har", "har_market", "full"])
    parser.add_argument("--forecast_horizon", type=int, default=5, choices=[1, 5, 10])
    parser.add_argument("--seq_length", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = FEATURE_SETS[args.feature_set]
    target_col = f"target_{args.forecast_horizon}d"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = os.path.join(project_root, "results", f"sp500_enhanced_h{args.forecast_horizon}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print(f"[INFO] Feature set: {args.feature_set} ({len(feature_cols)} features)")
    print(f"[INFO] Forecast horizon: {args.forecast_horizon} day(s) ahead (target_col={target_col})")
    print(f"[INFO] Tickers: {args.tickers}")
    print(f"[INFO] Device: {device}")

    # Load one DataFrame PER TICKER (kept separate — never concatenated across
    # tickers — so downstream windowing/scaling cannot cross ticker boundaries).
    ticker_dfs = {}
    for ticker in args.tickers:
        csv_path = os.path.join(project_root, "data", "processed_sp500_enhanced", f"{ticker}_enhanced.csv")
        if not os.path.isfile(csv_path):
            print(f"[WARN] {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        if target_col not in df.columns:
            print(f"[WARN] {ticker}: {target_col} not found (re-run feature_merger.py "
                  f"with --forecast_horizon {args.forecast_horizon}), skipping")
            continue
        df = df.dropna(subset=feature_cols + [target_col])
        ticker_dfs[ticker] = df

    if not ticker_dfs:
        print("[ERROR] No data found")
        return

    per_ticker = build_per_ticker_datasets(
        ticker_dfs, feature_cols, target_col, seq_length=args.seq_length,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, date_column="Date",
    )

    train_loader = DataLoader(
        ConcatDataset([per_ticker[t]["train"] for t in per_ticker]),
        batch_size=args.batch_size, shuffle=True,
    )

    # Model
    model = EnhancedLSTM(input_size=len(feature_cols), hidden_size=args.hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"[INFO] Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training loop
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate_pooled(model, per_ticker, "val", criterion, device, args.batch_size)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val DirAcc: {val_metrics['directional_accuracy']:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))

    # Test evaluation
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth")))
    test_loss, test_metrics = evaluate_pooled(model, per_ticker, "test", criterion, device, args.batch_size)

    print(f"\n[TEST] DirAcc: {test_metrics['directional_accuracy']:.2f}% - RMSE: {test_metrics['rmse']:.6f} - QLIKE: {test_metrics['qlike']:.6f}")

    # Save results
    results = {
        "feature_set": args.feature_set,
        "forecast_horizon": args.forecast_horizon,
        "n_features": len(feature_cols),
        "tickers": list(per_ticker.keys()),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Results saved to {results_dir}")


if __name__ == "__main__":
    main()
