# 🚀 Quick Start: Tạo GitHub Public Repository (10 phút)

## Bước 1: Tạo Repository trên GitHub (5 phút)

1. **Vào:** https://github.com/new
2. **Điền:**
   - Repository name: `stock_vol_prediction01`
   - Description: `TimesFM 2.5 LoRA Fine-Tuning for VN30 Volatility`
   - ✅ **Public**
   - ❌ **KHÔNG** check README
   - ❌ **KHÔNG** check .gitignore
   - ❌ **KHÔNG** check License
3. **Click:** Create repository
4. **Copy URL:** `https://github.com/YOUR_USERNAME/stock_vol_prediction01.git`

---

## Bước 2: Tạo Personal Access Token (3 phút)

1. **Vào:** https://github.com/settings/tokens
2. **Click:** Generate new token → Generate new token (classic)
3. **Điền:**
   - Note: `Stock Volatility`
   - Expiration: `90 days`
   - ✅ **Check:** repo (full control)
4. **Click:** Generate token
5. **COPY TOKEN** (format: `ghp_xxxxxxxxxxxxxxxxxxx`)

---

## Bước 3: Push Code (2 phút)

**Mở Git Bash:**

```bash
cd D:\bmad-projects\stock_vol_prediction01

# Thay YOUR_USERNAME và chạy script
bash setup_github_repo.sh YOUR_USERNAME
```

**Khi hỏi password:**
- Username: `YOUR_USERNAME`
- Password: `ghp_xxxxxx` (paste token)

**Hoặc nếu script lỗi, chạy trực tiếp:**

```bash
git remote add origin https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
git push -u origin master
```

---

## Bước 4: Verify (1 phút)

**Vào:** `https://github.com/YOUR_USERNAME/stock_vol_prediction01`

Kiểm tra:
- ✅ Files hiển thị (500 files)
- ✅ 3 commits
- ✅ Data có (data/raw/prices/*.csv)

---

## Bước 5: Train Trên Google Colab

```python
# Clone
!git clone https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
%cd stock_vol_prediction01

# Install
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers==4.36.0 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.0
!pip install pandas numpy scikit-learn matplotlib mlflow

# Quick test (5 epochs)
!python src/timesfm_baseline/train_timesfm_lora.py \
    --data_path data/raw/prices/ACB_ohlcv.csv \
    --stock_id ACB \
    --output_dir results \
    --epochs 5

# Full training (70 epochs)
!python src/timesfm_baseline/train_timesfm_lora.py \
    --multi_stock \
    --data_path data/raw/prices \
    --output_dir results \
    --epochs 70
```

---

## ✅ Checklist

- [ ] Repository created on GitHub (public)
- [ ] Personal Access Token created
- [ ] Code pushed successfully
- [ ] Files verified on GitHub
- [ ] Colab opened with GPU
- [ ] Clone command tested

---

## 🆘 Troubleshooting

| Error | Solution |
|-------|----------|
| `remote origin already exists` | `git remote remove origin` then run script again |
| `Authentication failed` | Check token has `repo` scope |
| `failed to push some refs` | Create repo on GitHub first |
| `Script not working on Windows` | Use manual commands above |

---

## 📚 Detailed Guides

- **CREATE_PUBLIC_REPO_GUIDE.md** - Full guide with screenshots
- **docs/GITHUB_SETUP_GUIDE.md** - Technical reference
- **docs/GOOGLE_COLAB_TRAINING_GUIDE.md** - Colab training guide

---

**Total Time:** 10 minutes
**Result:** Public GitHub repository ready for Colab training!

🎉 **Done!** Ready to train on Google Colab with GPU!
