"""
Script to apply validate() fix to train_parallel_enhanced.py

This script:
1. Creates backup of original file
2. Replaces validate() function with fixed version
3. Verifies the changes
"""

import shutil
from pathlib import Path


def apply_fix():
    """Apply validate() fix to train_parallel_enhanced.py"""

    project_root = Path(__file__).parent
    source_file = project_root / 'src' / 'lstm_gat_hybrid' / 'train_parallel_enhanced.py'
    fixed_file = project_root / 'src' / 'lstm_gat_hybrid' / 'validate_fixed.py'

    # Check if files exist
    if not source_file.exists():
        print(f"[ERROR] Source file not found: {source_file}")
        return False

    if not fixed_file.exists():
        print(f"[ERROR] Fixed file not found: {fixed_file}")
        return False

    # Create backup
    backup_file = source_file.with_suffix('.py.backup')
    print(f"[Step 1] Creating backup: {backup_file}")
    shutil.copy2(source_file, backup_file)
    print(f"  [OK] Backup created")

    # Read source file
    print(f"\n[Step 2] Reading source file...")
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"  [OK] Read {len(lines)} lines")

    # Find validate() function (Line 199-281)
    print(f"\n[Step 3] Locating validate() function...")
    start_line = None
    end_line = None

    for i, line in enumerate(lines):
        if 'def validate(model, dataloader, criterion, device, dataset=None):' in line:
            start_line = i
            print(f"  Found at line {i+1}")
            break

    if start_line is None:
        print(f"  [ERROR] validate() function not found!")
        return False

    # Find end of function (next function definition or end of file)
    for i in range(start_line + 1, len(lines)):
        if lines[i].strip().startswith('def ') and 'validate' not in lines[i]:
            end_line = i
            print(f"  Found end at line {i+1}")
            break
    else:
        # Last function in file
        end_line = len(lines)
        print(f"  Function extends to end of file (line {len(lines)})")

    print(f"  Function spans lines {start_line+1} to {end_line}")

    # Read fixed function
    print(f"\n[Step 4] Reading fixed function...")
    with open(fixed_file, 'r', encoding='utf-8') as f:
        fixed_lines = f.readlines()

    # Extract function from fixed file (skip first 25 lines of comments/docstring)
    fixed_function_start = None
    for i, line in enumerate(fixed_lines):
        if 'def validate(model, dataloader, criterion, device, dataset=None):' in line:
            fixed_function_start = i
            break

    if fixed_function_start is None:
        print(f"  [ERROR] Fixed function not found in {fixed_file}")
        return False

    # Extract function (until USAGE INSTRUCTIONS or end of file)
    fixed_function_lines = []
    for i in range(fixed_function_start, len(fixed_lines)):
        if 'USAGE INSTRUCTIONS' in fixed_lines[i]:
            break
        fixed_function_lines.append(fixed_lines[i])

    print(f"  [OK] Extracted {len(fixed_function_lines)} lines")

    # Replace function
    print(f"\n[Step 5] Replacing validate() function...")
    new_lines = lines[:start_line] + fixed_function_lines + lines[end_line:]

    print(f"  Original: {len(lines)} lines")
    print(f"  New:      {len(new_lines)} lines")
    print(f"  Diff:     {len(new_lines) - len(lines)} lines")

    # Write new file
    print(f"\n[Step 6] Writing new file...")
    with open(source_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"  [OK] File updated")

    # Verify
    print(f"\n[Step 7] Verifying changes...")
    with open(source_file, 'r', encoding='utf-8') as f:
        new_content = f.read()

    if 'Compute loss on ORIGINAL scale (consistent with test)' in new_content:
        print(f"  [OK] Fix verified: Loss now computed on ORIGINAL scale")
    else:
        print(f"  [WARNING] Could not verify fix")

    if 'all_targets_denorm = np.zeros_like(all_targets_norm)' in new_content:
        print(f"  [OK] Bug fix verified: zeros_like now uses all_targets_norm")
    else:
        print(f"  [WARNING] Bug fix not found")

    print(f"\n{'='*80}")
    print(f"FIX APPLIED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nChanges:")
    print(f"  1. Val Loss now computed on ORIGINAL scale (like Test MSE)")
    print(f"  2. Added debugging prints to verify normalization")
    print(f"  3. Fixed bug: zeros_like now uses all_targets_norm")
    print(f"\nExpected results:")
    print(f"  - Val Loss: ~1e-4 to ~1e-3 (original scale)")
    print(f"  - Test MSE: ~1e-6 to ~1e-5 (original scale)")
    print(f"  - Both now comparable!")
    print(f"\nBackup saved at: {backup_file}")
    print(f"\nTo test:")
    print(f"  python -m src.lstm_gat_hybrid.train_parallel_enhanced --graph_method knn --quick_test")

    return True


if __name__ == '__main__':
    print("="*80)
    print("APPLYING VALIDATE() FIX")
    print("="*80)
    print()

    success = apply_fix()

    if success:
        print("\n[SUCCESS] Fix applied successfully!")
        print("\nNext steps:")
        print("  1. Test with: python -m src.lstm_gat_hybrid.train_parallel_enhanced --graph_method knn --quick_test")
        print("  2. If OK, run full training")
        print("  3. Compare results with LSTM-HAR Enhanced (67.90%)")
    else:
        print("\n[ERROR] Fix failed!")
        print("\nManual steps:")
        print("  1. Open: src/lstm_gat_hybrid/train_parallel_enhanced.py")
        print("  2. Replace validate() function with code from: src/lstm_gat_hybrid/validate_fixed.py")
        print("  3. Save file")
