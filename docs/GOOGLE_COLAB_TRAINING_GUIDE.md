# TimesFM LoRA Fine-Tuning - Google Colab Training Guide

**Date:** 2026-06-20  
**Purpose:** Train TimesFM 2.5 LoRA model on Google Colab using GitHub repository

---

## Table of Contents

1. [Overview](#overview)
2. [Step 1: Prepare Local Repository](#step-1-prepare-local-repository)
3. [Step 2: Push Code & Data to GitHub](#step-2-push-code--data-to-github)
4. [Step 3: Setup Google Colab](#step-3-setup-google-colab)
5. [Step 4: Run Training](#step-4-run-training)
6. [Step 5: Save Results](#step-5-save-results)
7. [Troubleshooting](#troubleshooting)
8. [Tips & Best Practices](#tips--best-practices)

---

## Overview

### Workflow

```
Local Machine                     GitHub                      Google Colab
     │                               │                             │
     ├─ Check in code                │                             │
     ├─ Check in data                │                             │
     ├─ git commit                   │                             │
     ├─ git push ───────────────────>│                             │
     │                               │                             │
     │                               ├─ Pull code ──────────────>│
     │                               ├─ Pull data ──────────────>│
     │                               │                             │
     │                               │                             ├─ Setup env
     │                               │                             ├─ Train model
     │                               │                             ├─ Save results
     │                               │                             │
     │<─────────────────────────────│<──────── Pull results ──────┤
```

### Advantages of Google Colab

- ✅ **Free GPU** (Tesla T4 for free, A100/V100 for Pro)
- ✅ **Pre-installed dependencies** (PyTorch, CUDA, etc.)
- ✅ **No setup required** (just clone and run)
- ✅ **Fast training** (2-3 hours vs 8-10 hours on CPU)

---

## Step 1: Prepare Local Repository

### 1.1. Check Git Status

```bash
cd D:\bmad-projects\stock_vol_prediction01
git status
```

### 1.2. Create .gitignore for Large Files (If Not Exists)

If you want to exclude large files from git, create `.gitignore`:

```bash
# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# Data (optional - only if you don't want to commit data)
# data/raw/
# data/processed/

# Models (too large for git)
models/*.pt
models/*.pth
models/*.safetensors
models/*/

# Results
results/*/
*.png
*.jpg

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
```

**Note:** If you want to commit data to git, remove the `data/` lines from `.gitignore`.

### 1.3. Add All Files

```bash
# Add all files (including data if not in .gitignore)
git add .

# Check what will be committed
git status

# If you want to exclude specific large files:
# git reset path/to/large/file
```

---

## Step 2: Push Code & Data to GitHub

### 2.1. Commit Changes

```bash
# Commit all changes
git commit -m "Add TimesFM LoRA implementation with data

- TimesFM 2.5 LoRA fine-tuning implementation
- Dataset classes for Parkinson volatility
- Training script with hyperparameter validation
- All 40 bugs fixed from 3 adversarial reviews
- Test suite with 34 tests (100% pass rate)
- Processed VN30 volatility data (2006-2026)
- Documentation and lessons learned
"
```

### 2.2. Push to GitHub

```bash
# Push to main branch
git push origin master

# Or if using different branch name
git push origin main
```

### 2.3. Verify on GitHub

1. Go to your GitHub repository
2. Verify all files are uploaded
3. Check file sizes (GitHub has limits: 100MB per file, 1GB total per repo)
4. For large data files, consider using Git LFS (see [Troubleshooting](#troubleshooting))

---

## Step 3: Setup Google Colab

### 3.1. Open Google Colab

1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Click **"New Notebook"**
3. Name it: `TimesFM_LoRA_Fine_Tuning.ipynb`

### 3.2. Enable GPU Runtime

**CRITICAL:** Must enable GPU before running any code!

1. Click **Runtime** → **Change runtime type**
2. Select **T4 GPU** (free) or **A100/V100** (Colab Pro)
3. Click **Save**

### 3.3. Verify GPU

```python
# Run this cell to verify GPU is available
!nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| 28°C   X%    XW  X MiB /  15109MiB |      X%   Default |                   N/A |
+-------------------------------+----------------------+----------------------+
```

---

## Step 4: Clone Repository and Install Dependencies

### 4.1. Clone Repository

```python
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Change to repository directory
%cd YOUR_REPO_NAME

# Verify files are present
!ls -la
```

**Replace:** `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub details.

### 4.2. Install Dependencies

```python
# Install PyTorch (if not already installed)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required packages
!pip install transformers==4.36.0
!pip install peft==0.7.1
!pip install accelerate==0.25.0
!pip install bitsandbytes==0.41.0
!pip install datasets==2.15.0

# Install other dependencies
!pip install pandas numpy scikit-learn matplotlib mlflow

# Verify installations
!pip list | grep -E "torch|transformers|peft|accelerate"
```

### 4.3. Verify Installation

```python
# Verify PyTorch can use GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Step 5: Run Training

### 5.1. Quick Test (Single Stock, 5 Epochs)

```python
# Quick test to verify everything works
!python src/timesfm_baseline/train_timesfm_lora.py \
    --data_path data/processed/vn30_parkinson_volatility.parquet \
    --stock_id VIC \
    --output_dir results/timesfm_test \
    --epochs 5 \
    --batch_size 32 \
    --context_len 64 \
    --horizon_len 5 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42
```

**Expected time:** ~5-10 minutes on GPU

### 5.2. Full Training (All Stocks, 70 Epochs)

```python
# Full training with all stocks
!python src/timesfm_baseline/train_timesfm_lora.py \
    --data_path data/processed/vn30_parkinson_volatility.parquet \
    --multi_stock \
    --output_dir results/timesfm_full_$(date +%Y%m%d_%H%M%S) \
    --epochs 70 \
    --batch_size 32 \
    --context_len 64 \
    --horizon_len 5 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42
```

**Expected time:** ~2-3 hours on GPU (Tesla T4)

### 5.3. Monitor Training

```python
# Watch training progress in real-time
!tail -f results/timesfm_full_*/training.log
```

Or use MLflow UI (if enabled):

```python
# Start MLflow UI (in separate cell)
!mlflow ui --port 5000

# Then open the URL shown (usually http://localhost:5000)
# NOTE: In Colab, use ngrok to expose MLflow UI
```

---

## Step 6: Save Results

### 6.1. View Results

```python
# Check training results
!ls -la results/timesfm_full_*/

# View final metrics
!cat results/timesfm_full_*/final_metrics.json

# View learning curves
from IPython.display import Image, display
import glob

display(Image(filename=glob.glob('results/timesfm_full_*/learning_curves_*.png')[0]))
```

### 6.2. Download Results to Local Machine

**Option 1: Download from Colab UI**

1. Click 📁 **Files** icon in left sidebar
2. Navigate to `results/timesfm_full_*/`
3. Right-click file/folder → **Download**

**Option 2: Use Google Drive (Recommended)**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Google Drive
!cp -r results/timesfm_full_* /content/drive/MyDrive/TimesFM_Results/

# Verify
!ls -la /content/drive/MyDrive/TimesFM_Results/
```

**Option 3: Commit and Push to GitHub**

```python
# Add results to git
!git add results/timesfm_full_*

# Commit results
!git commit -m "Add TimesFM LoRA training results

- Trained on 30 VN30 stocks
- 70 epochs, early stopping with patience=15
- Final RMSE: X.XXXX
- Final Dir Acc: XX.XX%
- Best model saved to results/timesfm_full_*/best_model_*/
"

# Push to GitHub
!git push origin master
```

### 6.3. Download Best Model

```python
# Download best model (LoRA adapters)
!zip -r best_model.zip results/timesfm_full_*/best_model_*

# Download the zip file
from google.colab import files
files.download('best_model.zip')
```

---

## Step 7: Use Model for Prediction

### 7.1. Load Trained Model

```python
from src.timesfm_baseline.timesfm_lora_finetuning import TimesFMLoRAFineTuner

# Initialize finetuner
finetuner = TimesFMLoRAFineTuner(
    context_len=64,
    horizon_len=5,
    device='cuda'
)

# Load trained model
finetuner.load_model(
    model_id='google/timesfm-2.5-500m',
    adapter_path='results/timesfm_full_*/best_model_*/'
)

print("Model loaded successfully!")
```

### 7.2. Make Predictions

```python
import pandas as pd
from src.common.parkinson_utils import calculate_parkinson_volatility

# Load test data
test_data = pd.read_parquet('data/processed/vn30_parkinson_volatility.parquet')

# Select a stock for prediction
stock_data = test_data[test_data['stock_id'] == 'VIC'].tail(100)

# Create dataset
from src.timesfm_baseline.volatility_dataset import TimeSeriesLastWindowDataset
dataset = TimeSeriesLastWindowDataset(
    [stock_data['parkinson_volatility'].values],
    context_len=64,
    horizon_len=5
)

# Get prediction
context, target = dataset[0]
context = context.unsqueeze(0).to('cuda')  # Add batch dimension

with torch.no_grad():
    output = finetuner.model(
        past_values=context,
        forecast_context_len=64
    )
    prediction = output.mean_predictions[0, :5].cpu().numpy()

print(f"Predicted volatility (next 5 days): {prediction}")
```

---

## Troubleshooting

### Issue 1: GitHub File Size Limit (100MB)

**Problem:** Data files > 100MB cannot be committed directly.

**Solution A: Use Git LFS**

```bash
# Install Git LFS (local machine)
git lfs install

# Track large files
git lfs track "data/processed/*.parquet"
git lfs track "data/raw/*.csv"

# Commit .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"

# Push files
git add data/processed/*.parquet
git commit -m "Add processed data"
git push origin master
```

**Solution B: Use External Storage**

- Upload data to Google Drive, Dropbox, or AWS S3
- Download directly in Colab:

```python
# Download from Google Drive
!gdown --id YOUR_FILE_ID

# Or download from URL
!wget https://example.com/data/vn30_volatility.parquet
```

### Issue 2: Out of Memory (OOM)

**Problem:** Training crashes with CUDA OOM error.

**Solutions:**

1. **Reduce batch size:**
```python
# From batch_size=32 to batch_size=16 or 8
!python src/timesfm_baseline/train_timesfm_lora.py --batch_size 16 ...
```

2. **Reduce context length:**
```python
# From context_len=64 to context_len=32
!python src/timesfm_baseline/train_timesfm_lora.py --context_len 32 ...
```

3. **Use gradient accumulation:**
```python
# Add gradient accumulation (edit train_timesfm_lora.py)
# Then train with accumulation_steps=4
```

### Issue 3: Colab Session Timeout

**Problem:** Colab session disconnects after 90 minutes of inactivity.

**Solutions:**

1. **Run training in background with nohup:**
```python
# Run training in background
!nohup python src/timesfm_baseline/train_timesfm_lora.py \
    --data_path data/processed/vn30_parkinson_volatility.parquet \
    --multi_stock \
    --output_dir results/timesfm_full \
    --epochs 70 \
    --batch_size 32 \
    > training.log 2>&1 &

# Monitor progress
!tail -f training.log
```

2. **Use JavaScript to keep Colab alive (run in console):**
```javascript
// Run in browser console (F12)
function ClickConnect(){
  console.log("Working...");
  document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)
```

### Issue 4: Model Download Fails

**Problem:** TimesFM model download fails or times out.

**Solution:** Use huggingface-cli login

```python
# Install huggingface-cli
!pip install huggingface_hub

# Login (you'll need Hugging Face token)
!huggingface-cli login

# Then retry training
!python src/timesfm_baseline/train_timesfm_lora.py ...
```

### Issue 5: Data Not Found After Clone

**Problem:** Data files missing after cloning.

**Solutions:**

1. **Verify data was committed:**
```bash
# Check git history
git log --all --full-history -- data/processed/

# If data not in history, need to add and push
git add data/processed/
git commit -m "Add processed data"
git push origin master
```

2. **Use Git LFS (see Issue 1):**
```bash
git lfs pull
```

---

## Tips & Best Practices

### Training Tips

1. **Start with quick test:** Always run 5-epoch test first before full training
2. **Monitor GPU usage:** Use `!nvidia-smi` to ensure GPU is being utilized
3. **Save checkpoints:** Use default checkpoint saving (every half epoch)
4. **Use MLflow:** Track experiments and compare hyperparameters

### Performance Tips

1. **Use pin_memory=True:** Already enabled in code (faster data loading)
2. **Use num_workers=4:** Parallel data loading (already enabled)
3. **Use mixed precision:** Consider using `torch.cuda.amp` for faster training
4. **Profile code:** Use `torch.profiler` to identify bottlenecks

### Data Management Tips

1. **Compress data:** Use parquet format (already done, efficient)
2. **Cache features:** Pre-compute features if training multiple times
3. **Use subsampling:** Start with 5 stocks before training all 30

### Colab Tips

1. **Save to Google Drive:** Prevents data loss on session timeout
2. **Use Pro for longer training:** A100 GPU is 3-5x faster than T4
3. **Monitor credit usage:** Check usage in Colab settings
4. **Save frequently:** Results are auto-saved to Google Drive

---

## Example: Complete Training Session

```python
# ============================================================================
# TimesFM LoRA Fine-Tuning on Google Colab - Complete Example
# ============================================================================

# Step 1: Setup
!nvidia-smi  # Verify GPU

# Step 2: Clone repository
!git clone https://github.com/ntquy99/stock_vol_prediction01.git
%cd stock_vol_prediction01

# Step 3: Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers==4.36.0 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.0
!pip install pandas numpy scikit-learn matplotlib mlflow

# Step 4: Mount Google Drive (for saving results)
from google.colab import drive
drive.mount('/content/drive')

# Step 5: Quick test (5 epochs)
!python src/timesfm_baseline/train_timesfm_lora.py \
    --data_path data/processed/vn30_parkinson_volatility.parquet \
    --stock_id VIC \
    --output_dir results/timesfm_quick_test \
    --epochs 5 \
    --batch_size 32 \
    --context_len 64 \
    --horizon_len 5 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42

# Step 6: Full training (70 epochs) - run if test successful
!python src/timesfm_baseline/train_timesfm_lora.py \
    --data_path data/processed/vn30_parkinson_volatility.parquet \
    --multi_stock \
    --output_dir results/timesfm_full_$(date +%Y%m%d_%H%M%S) \
    --epochs 70 \
    --batch_size 32 \
    --context_len 64 \
    --horizon_len 5 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42

# Step 7: Copy results to Google Drive
!cp -r results/timesfm_full_* /content/drive/MyDrive/TimesFM_Results/

# Step 8: View results
!cat /content/drive/MyDrive/TimesFM_Results/*/final_metrics.json

# Step 9: Download best model
!zip -r best_model.zip /content/drive/MyDrive/TimesFM_Results/*/best_model_*/
from google.colab import files
files.download('best_model.zip')

print("Training complete! Results saved to Google Drive.")
```

---

## Summary

### Checklist

- [ ] Local code committed to git
- [ ] Data committed to git (or Git LFS)
- [ ] Pushed to GitHub
- [ ] Opened Colab with GPU enabled
- [ ] Cloned repository
- [ ] Installed dependencies
- [ ] Ran quick test (5 epochs)
- [ ] Verified test successful
- [ ] Ran full training (70 epochs)
- [ ] Saved results to Google Drive
- [ ] Downloaded best model

### Expected Training Time

| Configuration | Time (Tesla T4) | Time (A100) |
|--------------|-----------------|-------------|
| 1 stock, 5 epochs | ~5-10 min | ~2-5 min |
| 1 stock, 70 epochs | ~1-1.5 hr | ~20-30 min |
| 30 stocks, 70 epochs | ~2-3 hr | ~30-45 min |

### File Sizes (Approximate)

| File | Size |
|------|------|
| Base TimesFM 2.5 model (download once) | ~950 MB |
| LoRA adapters (trained) | ~6 MB |
| Processed data (30 stocks) | ~5-10 MB |
| Best model checkpoint | ~6 MB |
| Training logs | ~1-2 MB |
| Learning curves plot | ~100 KB |

---

**Last Updated:** 2026-06-20  
**Author:** Stock Volatility Prediction Team  
**Status:** ✅ Ready for Production Use
