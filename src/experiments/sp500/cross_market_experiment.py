"""
Cross-Market Experiment: Train on one market, test on another

Usage:
    python src/experiments/sp500/cross_market_experiment.py --train_market sp500 --test_market vn30
    python src/experiments/sp500/cross_market_experiment.py --train_market vn30 --test_market sp500
    python src/experiments/sp500/cross_market_experiment.py --forecast_horizon 10
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
from src.common.har_features import generate_har_features
from src.common.multi_ticker_dataset import build_per_ticker_datasets, build_full_series_datasets


MARKET_PROCESSED_DIRS = {
    "sp500": os.path.join(project_root, "data", "processed_sp500"),
    "vn30": os.path.join(project_root, "data", "processed"),
}

FEATURE_COLS = ["har_daily_vol", "har_weekly_vol", "har_monthly_vol"]


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
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


def load_market_data(market, tickers=None, horizon=5):
    """Load processed data for a market, kept as ONE DATAFRAME PER TICKER (never
    concatenated across tickers) so downstream windowing/scaling cannot cross
    ticker boundaries.

    Returns:
        dict mapping ticker -> DataFrame with FEATURE_COLS, f"target_{horizon}d", and "date".
    """
    processed_dir = MARKET_PROCESSED_DIRS[market]
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(f"Processed data not found for {market}: {processed_dir}")

    files = [f for f in os.listdir(processed_dir) if f.endswith('_processed.csv')]
    if tickers:
        files = [f for f in files if f.replace('_processed.csv', '') in tickers]

    target_col = f"target_{horizon}d"
    ticker_dfs = {}
    for f in files:
        ticker = f.replace('_processed.csv', '')
        csv_path = os.path.join(processed_dir, f)
        try:
            df = pd.read_csv(csv_path)
            if "date" not in df.columns:
                df["date"] = df.iloc[:, 0]

            if "har_daily_vol" not in df.columns:
                df = generate_har_features(df, volatility_col="parkinson_volatility")

            df[target_col] = df["parkinson_volatility"].shift(-horizon)
            df = df.dropna(subset=FEATURE_COLS + [target_col])
            if len(df) == 0:
                continue
            ticker_dfs[ticker] = df
        except Exception as e:
            print(f"[WARN] Error loading {f}: {e}")

    if not ticker_dfs:
        raise ValueError(f"No valid data found for {market}")

    total_rows = sum(len(df) for df in ticker_dfs.values())
    print(f"[INFO] {market}: {total_rows} rows, {len(ticker_dfs)} tickers")
    return ticker_dfs


def train_model(model, train_loader, evaluate_fn, epochs=70, patience=15, lr=1e-3, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss, _ = evaluate_fn(model, criterion, device)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} - Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def evaluate_train_market_split(model, per_ticker_train, split, criterion, device, batch_size=64):
    """Evaluate the train market's own val/train split (per-ticker scalers)."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for ticker, splits in per_ticker_train.items():
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
    metrics = evaluate_predictions(np.array(all_true), np.array(all_pred)) if all_true else None
    return avg_loss, metrics


def evaluate_full_series(model, per_ticker_full, device, batch_size=64):
    """Evaluate every ticker's full-series dataset (own scaler), pooled across tickers."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for ticker, entry in per_ticker_full.items():
            ds = entry["data"]
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=batch_size)
            target_scaler = entry["target_scaler"]
            for x, y in loader:
                x = x.to(device)
                pred = model(x).squeeze(-1)
                pred_np = target_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).flatten()
                true_np = target_scaler.inverse_transform(y.numpy().reshape(-1, 1)).flatten()
                all_true.extend(true_np)
                all_pred.extend(pred_np)

    return evaluate_predictions(np.array(all_true), np.array(all_pred))


def run_experiment(train_market, test_market, horizon=5, epochs=70, patience=15, seq_length=22, batch_size=64):
    """Run a single cross-market experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: Train={train_market.upper()} -> Test={test_market.upper()} (horizon={horizon}d)")
    print(f"{'='*60}")

    # Load data — one DataFrame per ticker, never concatenated.
    train_ticker_dfs = load_market_data(train_market, horizon=horizon)
    test_ticker_dfs = load_market_data(test_market, horizon=horizon)

    target_col = f"target_{horizon}d"

    # Train market: per-ticker 85/15 train/val split (no separate test split needed here).
    per_ticker_train = build_per_ticker_datasets(
        train_ticker_dfs, FEATURE_COLS, target_col, seq_length=seq_length,
        train_ratio=0.85, val_ratio=0.15, test_ratio=0.0, date_column="date",
    )
    train_loader = DataLoader(
        ConcatDataset([per_ticker_train[t]["train"] for t in per_ticker_train]),
        batch_size=batch_size, shuffle=True,
    )

    # Test market: no train tranche of its own tickers exists (cross-market) — fit
    # scaler on its own full series, evaluate every window as "test".
    per_ticker_test_full = build_full_series_datasets(
        test_ticker_dfs, FEATURE_COLS, target_col, seq_length=seq_length, date_column="date",
    )

    # Train
    model = SimpleLSTM(input_size=3, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    print(f"[INFO] Training model: {sum(p.numel() for p in model.parameters()):,} parameters")

    def evaluate_fn(model, criterion, device):
        return evaluate_train_market_split(model, per_ticker_train, "val", criterion, device, batch_size)

    model, train_losses, val_losses = train_model(
        model, train_loader, evaluate_fn, epochs=epochs, patience=patience, device=device
    )

    # Evaluate on cross-market test
    test_metrics = evaluate_full_series(model, per_ticker_test_full, device, batch_size)

    print(f"\n[TEST] DirAcc: {test_metrics['directional_accuracy']:.2f}% - "
          f"RMSE: {test_metrics['rmse']:.6f} - QLIKE: {test_metrics['qlike']:.6f}")

    train_rows = sum(len(df) for df in train_ticker_dfs.values())
    test_rows = sum(len(df) for df in test_ticker_dfs.values())

    return {
        "train_market": train_market,
        "test_market": test_market,
        "forecast_horizon": horizon,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_market", default="sp500", choices=["sp500", "vn30"])
    parser.add_argument("--test_market", default="vn30", choices=["sp500", "vn30"])
    parser.add_argument("--forecast_horizon", type=int, default=5, choices=[1, 5, 10])
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = os.path.join(project_root, "results", f"cross_market_h{args.forecast_horizon}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    result = run_experiment(
        args.train_market, args.test_market, horizon=args.forecast_horizon,
        epochs=args.epochs, patience=args.patience,
    )

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[SUCCESS] Results saved to {results_dir}")


if __name__ == "__main__":
    main()
