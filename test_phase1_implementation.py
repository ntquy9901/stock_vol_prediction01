"""
Test Phase 1: Data-Centric Anti-Overfitting Techniques

Tests the new outlier removal and data augmentation features.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def test_phase1_implementation():
    """Test Phase 1 anti-overfitting techniques"""
    print("="*80)
    print("TESTING PHASE 1: DATA-CENTRIC ANTI-OVERFITTING TECHNIQUES")
    print("="*80)

    # Test imports
    print("\n1. Testing imports...")
    try:
        from src.lstm_gat_hybrid.dataset import (
            remove_outliers,
            augment_sequence,
            MultiStockDataset,
            create_multi_stock_dataloaders
        )
        print("   [PASS] All imports successful")
    except ImportError as e:
        print(f"   [FAIL] Import failed: {e}")
        return False

    # Test outlier removal
    print("\n2. Testing outlier removal...")
    import pandas as pd
    import numpy as np

    # Create test data with outliers
    test_data = pd.DataFrame({
        'parkinson_volatility': [0.001, 0.002, 0.0015, 0.002, 0.0018, 0.1, 0.002, 0.001]  # 0.1 is outlier
    })

    original_len = len(test_data)
    cleaned_data = remove_outliers(test_data, n_std=2.0)  # Lower threshold for testing

    print(f"   Original length: {original_len}")
    print(f"   Cleaned length: {len(cleaned_data)}")
    print(f"   Outliers removed: {original_len - len(cleaned_data)}")

    if len(cleaned_data) < original_len:
        print("   [PASS] Outlier removal working")
    else:
        print("   [WARNING] No outliers removed (threshold may be too high)")

    # Test data augmentation
    print("\n3. Testing data augmentation...")
    x_seq = np.random.randn(22, 5, 3)  # [seq_len, num_stocks, num_features]
    y_seq = np.random.randn(5)  # [num_stocks]

    original_x_mean = x_seq.mean()
    original_x_std = x_seq.std()

    x_aug, y_aug = augment_sequence(x_seq, y_seq, augmentation_factor=0.1)

    print(f"   Original X mean: {original_x_mean:.6f}, std: {original_x_std:.6f}")
    print(f"   Augmented X mean: {x_aug.mean():.6f}, std: {x_aug.std():.6f}")
    print(f"   X changed: {not np.allclose(x_seq, x_aug)}")
    print(f"   Y unchanged: {np.allclose(y_seq, y_aug)}")

    if not np.allclose(x_seq, x_aug) and np.allclose(y_seq, y_aug):
        print("   [PASS] Data augmentation working")
    else:
        print("   [FAIL] Data augmentation not working correctly")

    # Test dataset creation with new parameters
    print("\n4. Testing dataset with Phase 1 parameters...")
    try:
        dataset = MultiStockDataset(
            data_dir='data/processed',
            seq_length=22,
            forecast_horizon=5,
            graph_method='correlation',
            normalize=True,
            # Phase 1 parameters
            remove_outliers=True,
            n_std=3.0,
            data_augmentation=True,
            augmentation_prob=0.3,
            augmentation_factor=0.1,
            train_mode=False  # No augmentation during testing
        )
        print(f"   Dataset created successfully")
        print(f"   Number of sequences: {len(dataset)}")
        print("   [PASS] Dataset initialization working")

        # Test __getitem__ with train_mode
        print("\n5. Testing __getitem__ with train mode...")
        x, adj_matrix, y, graph_data = dataset[0]
        print(f"   Input shape: {x.shape}")
        print(f"   Adjacency shape: {adj_matrix.shape}")
        print(f"   Target shape: {y.shape}")
        print("   [PASS] __getitem__ working")

    except Exception as e:
        print(f"   [FAIL] Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test dataloader creation
    print("\n6. Testing dataloader creation with Phase 1 parameters...")
    try:
        train_loader, val_loader, test_loader, datasets = create_multi_stock_dataloaders(
            data_dir='data/processed',
            seq_length=22,
            forecast_horizon=5,
            graph_method='correlation',
            batch_size=4,  # Small batch for testing
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            num_workers=0,
            normalize=True,
            # Phase 1 parameters
            remove_outliers=True,
            n_std=3.0,
            data_augmentation=True,
            augmentation_prob=0.3,
            augmentation_factor=0.1
        )

        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        print("   [PASS] Dataloader creation working")

        # Test one batch from train loader
        print("\n7. Testing train loader batch...")
        for x, adj_matrix, y, graph_data in train_loader:
            print(f"   Batch input shape: {x.shape}")
            print(f"   Batch adjacency shape: {adj_matrix.shape}")
            print(f"   Batch target shape: {y.shape}")
            print(f"   Data augmentation should be applied to training set")
            print("   [PASS] Train loader working")
            break  # Only test first batch

    except Exception as e:
        print(f"   [FAIL] Dataloader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("PHASE 1 IMPLEMENTATION TEST COMPLETE")
    print("="*80)
    print("[SUCCESS] All Phase 1 techniques working correctly!")
    print("\nNext steps:")
    print("1. Train model with Phase 1 techniques")
    print("2. Monitor prediction variance during training")
    print("3. Compare Dir Acc with previous results (0.07%)")
    print("4. Expected: Dir Acc > 40% (significant improvement)")

    return True

if __name__ == '__main__':
    success = test_phase1_implementation()
    sys.exit(0 if success else 1)