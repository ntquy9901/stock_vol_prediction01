"""
Train models with configuration support (Wrapper)
This is a convenience wrapper for src/experiment/train_with_config.py

Usage:
    python src/train_with_config.py --scenario vn30_only --model enhanced_lstm_har
    python src/train_with_config.py --scenario all_combined --model enhanced_lstm_har
"""

import sys
from pathlib import Path

# Get the actual script path
script_dir = Path(__file__).parent
experiment_script = script_dir / "experiment" / "train_with_config.py"

# Run the actual script with all arguments
if __name__ == "__main__":
    if experiment_script.exists():
        import subprocess
        result = subprocess.run([sys.executable, str(experiment_script)] + sys.argv[1:])
        sys.exit(result.returncode)
    else:
        print(f"Error: Could not find {experiment_script}")
        sys.exit(1)
