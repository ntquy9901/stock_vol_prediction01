# ⚡ Quick Start - 3 Minutes Setup

**Đã có 4 commits sẵn sàng push lên GitHub!**

---

## 🎯 CÁCH 1: Dùng Script Automation (Khuyến Nghị)

### Bước 1: Tạo GitHub Repository (2 phút)

1. Vào: https://github.com/new
2. Điền: `stock_vol_prediction01`
3. ✅ Public
4. ❌ KHÔNG check README, .gitignore, License
5. Click: **Create repository**

### Bước 2: Tạo Personal Access Token (1 phút)

1. Vào: https://github.com/settings/tokens
2. Click: **Generate new token (classic)**
3. ✅ Check: **repo**
4. Click: **Generate token**
5. **Copy token:** `ghp_xxxxxxxxxxxxxx`

### Bước 3: Push Code (30 giây)

```bash
cd D:\bmad-projects\stock_vol_prediction01

# Chạy script (thay YOUR_USERNAME)
bash setup_github_repo.sh YOUR_USERNAME
```

**Khi hỏi password:**
- Username: YOUR_USERNAME
- Password: <paste token>

✅ **Xong!** Code đã push lên GitHub.

---

## 🎯 CÁCH 2: Manual (Nếu Script Lỗi)

```bash
cd D:\bmad-projects\stock_vol_prediction01

# Thêm remote (thay YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/stock_vol_prediction01.git

# Push
git push -u origin master
```

**Khi hỏi password:**
- Username: YOUR_USERNAME
- Password: <paste token>

---

## ✅ Verify

**Vào:** `https://github.com/YOUR_USERNAME/stock_vol_prediction01`

Bạn sẽ thấy:
- ✅ 500 files
- ✅ 4 commits
- ✅ Data (data/raw/prices/*.csv)

---

## 🚀 Train Trên Google Colab

```python
# Clone
!git clone https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
%cd stock_vol_prediction01

# Install
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers==4.36.0 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.0
!pip install pandas numpy scikit-learn matplotlib mlflow

# Quick test (5 epochs, 5-10 phút)
!python src/timesfm_baseline/train_timesfm_lora.py \
    --data_path data/raw/prices/ACB_ohlcv.csv \
    --stock_id ACB \
    --output_dir results \
    --epochs 5

# Full training (70 epochs, 2-3 giờ)
!python src/timesfm_baseline/train_timesfm_lora.py \
    --multi_stock \
    --data_path data/raw/prices \
    --output_dir results \
    --epochs 70
```

---

## 📚 Chi Tiết

- `QUICK_START_GITHUB.md` - Quick reference (1 page)
- `CREATE_PUBLIC_REPO_GUIDE.md` - Full guide with screenshots
- `docs/GITHUB_SETUP_GUIDE.md` - Technical troubleshooting
- `docs/GOOGLE_COLAB_TRAINING_GUIDE.md` - Colab training guide

---

**Thời gian:** 3 phút setup + 2-3 giờ training
**Kết quả:** TimesFM 2.5 LoRA trained model ready!

🎉 **Let's go!**
