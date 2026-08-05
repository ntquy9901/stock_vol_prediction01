"""Isolated 10-epoch sentiment baseline trainer.

Reuses train_epoch / validate / EarlyStopping / plot fn from the EXISTING trainer
(src/lstm_gat_hybrid/train_parallel_enhanced.py) via import - no modification.
Only this main() is new. Changes vs original:
  - 5 features (3 HAR + 2 daily sentiment) -> config.num_features_per_stock = 5
  - sentiment dataloader (data_dir + sentiment_dir)
  - 10 epochs, patience 5
  - output: results/sentiment_baseline_{graph_method}_{timestamp}/

Does NOT modify any existing file/folder.

Run:  python -m src.sentiment_baseline.train_sentiment_baseline
      python -m src.sentiment_baseline.train_sentiment_baseline --graph_method correlation --epochs 10
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.lstm_gat_hybrid.model_parallel import create_parallel_lstm_gat_model
from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.train_parallel_enhanced import (
    EarlyStopping, train_epoch, validate, plot_learning_curves_with_analysis,
)
from src.sentiment_baseline.dataset_sentiment import create_sentiment_dataloaders


def train_sentiment_baseline(graph_method='knn', num_epochs=10, resume_from=None,
                             sentiment_dir='data/sentiment_baseline'):
    print("=" * 80)
    print(f"SENTIMENT BASELINE (10-epoch quick run) - graph_method={graph_method}")
    print("Parallel LSTM-GNN + 2 daily lexicon sentiment features")
    print("=" * 80)
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

    config = LSTMGATConfig()
    config.num_features_per_stock = 5  # 3 HAR + 2 sentiment (CRITICAL - controls model input dim)
    config.learning_rate = 0.001
    config.batch_size = 11
    config.num_epochs = num_epochs
    config.patience = 5
    config.min_epochs = 5
    config.weight_decay = 1e-5
    config.gradient_clip = 0.5
    config.lstm_dropout = 0.2
    config.fusion_dropout = 0.15

    print(f"\nConfiguration:")
    print(f"  Features/stock: {config.num_features_per_stock} (3 HAR + 2 sentiment)")
    print(f"  Epochs: {config.num_epochs} | patience: {config.patience}")
    print(f"  LR: {config.learning_rate} | batch: {config.batch_size}")

    device = torch.device(config.device)
    print(f"  Device: {device}")

    if graph_method == 'correlation':
        graph_threshold, k_neighbors = 0.1, None
    else:
        graph_threshold, k_neighbors = None, 8

    train_loader, val_loader, test_loader, datasets = create_sentiment_dataloaders(
        data_dir='data/processed',
        sentiment_dir=sentiment_dir,
        seq_length=config.seq_length,
        forecast_horizon=config.forecast_horizon,
        graph_method=graph_method,
        graph_threshold=graph_threshold,
        k_neighbors=k_neighbors,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        normalize=True,
        remove_outliers=True,
        n_std=3.0,
        config=config,
    )

    model = create_parallel_lstm_gat_model(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    if resume_from:
        print(f"[resume] Loading model weights from {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"[resume] Resumed. Training {num_epochs} MORE epochs from this checkpoint.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta,
                                   min_epochs=config.min_epochs)

    train_losses, val_losses = [], []
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    results_dir = Path(f'results/sentiment_baseline_{graph_method}_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults dir: {results_dir}\n")

    best_val_loss, best_epoch, epoch_times = float('inf'), 0, []
    last_epoch = 0
    for epoch in range(config.num_epochs):
        last_epoch = epoch
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        train_losses.append(train_loss)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, datasets[1])
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f"{epoch + 1:>3}/{config.num_epochs} | train={train_loss:.6f} | val={val_loss:.6f} | "
              f"val DirAcc={val_metrics['directional_accuracy']:.2f}% | lr={lr:.6f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), results_dir / 'best_parallel_model.pth')

        if (epoch + 1) % 5 == 0:
            plot_learning_curves_with_analysis(train_losses, val_losses, results_dir, epoch)

        if early_stopping(val_loss, epoch):
            print(f"[early stop] epoch {epoch + 1}, best={best_epoch}", flush=True)
            break
        epoch_times.append(time.time() - t0)

    # Final test evaluation with best model
    print(f"\nLoading best model (epoch {best_epoch}) for test eval...")
    model.load_state_dict(torch.load(results_dir / 'best_parallel_model.pth'))
    test_loss, test_metrics = validate(model, test_loader, criterion, device, datasets[2])

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (sentiment baseline, {graph_method}, {last_epoch + 1} epochs):")
    print(f"  MSE:     {test_metrics['mse']:.6f}")
    print(f"  RMSE:    {test_metrics['rmse']:.6f}")
    print(f"  MAE:     {test_metrics['mae']:.6f}")
    print(f"  R2:      {test_metrics['r2']:.6f}")
    print(f"  QLIKE:   {test_metrics['qlike']:.6f}")
    print(f"  Dir Acc: {test_metrics['directional_accuracy']:.2f}%")
    print(f"{'='*60}")

    results = {
        'model': f'Sentiment Baseline (Parallel LSTM-GNN + sentiment, {graph_method})',
        'timestamp': timestamp,
        'architecture': 'Parallel LSTM-GNN + 2 daily sentiment features (lexicon scorer)',
        'graph_method': graph_method,
        'features': '3 HAR + 2 sentiment (sentiment_1d, news_count_1d)',
        'config': {
            'num_features_per_stock': config.num_features_per_stock,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'num_epochs_trained': last_epoch + 1,
            'best_epoch': best_epoch,
            'patience': config.patience,
            'weight_decay': config.weight_decay,
        },
        'training_summary': {
            'num_epochs_trained': last_epoch + 1,
            'best_epoch': best_epoch,
            'best_val_loss': float(best_val_loss),
            'total_time_minutes': float(sum(epoch_times) / 60) if epoch_times else 0.0,
        },
        'test_metrics': {k: float(test_metrics[k]) for k in
                         ['mse', 'rmse', 'mae', 'r2', 'qlike', 'directional_accuracy']},
    }
    with open(results_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_dir / 'training_results.json'}")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Sentiment baseline (10-epoch quick run)')
    parser.add_argument('--graph_method', default='knn', choices=['knn', 'correlation'],
                        help='Graph construction method (default: knn)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--resume_from', default=None,
                        help='Path to a saved best_parallel_model.pth to resume from (trains N MORE epochs)')
    parser.add_argument('--sentiment_dir', default='data/sentiment_baseline',
                        help='Folder with {TICKER}_sentiment.csv (default: data/sentiment_baseline)')
    args = parser.parse_args()
    train_sentiment_baseline(graph_method=args.graph_method, num_epochs=args.epochs,
                             resume_from=args.resume_from, sentiment_dir=args.sentiment_dir)
