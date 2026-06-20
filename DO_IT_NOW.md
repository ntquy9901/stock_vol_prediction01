# 🎯 BẠN CẦN LÀM GÌ NGAY BÂY GIỜ

**Thời gian:** 5 phút setup → 2-3 giờ training

---

## 1️⃣ Tạo GitHub Repository (2 phút)

### Bước 1: Tạo Repo

1. **Click:** https://github.com/new
2. **Điền:**
   - Repository name: `stock_vol_prediction01`
   - ✅ Public
   - ❌ KHÔNG check README
   - ❌ KHÔNG check .gitignore
   - ❌ KHÔNG check License
3. **Click:** Create repository
4. **Copy URL:** `https://github.com/YOUR_USERNAME/stock_vol_prediction01.git`

### Bước 2: Tạo Token

1. **Click:** https://github.com/settings/tokens
2. **Click:** Generate new token (classic)
3. ✅ Check: **repo**
4. **Click:** Generate token
5. **Copy token:** `ghp_xxxxxxxxxxxxxxxxxxx`

---

## 2️⃣ Push Code (1 phút)

**Mở Git Bash:**

```bash
cd D:\bmad-projects\stock_vol_prediction01
bash setup_github_repo.sh YOUR_USERNAME
```

**Khi hỏi password:**
- Username: `YOUR_USERNAME`
- Password: `<paste token>`

✅ **Xong!** Code đã push lên GitHub.

---

## 3️⃣ Train Trên Google Colab (2-3 giờ)

### Mở Colab: https://colab.research.google.com/

### Bật GPU:
- Click **Runtime** → **Change runtime type**
- Chọn **T4 GPU**
- Click **Save**

### Paste Code Vào Colab:

```python
# Cell 1: Clone
!git clone https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
%cd stock_vol_prediction01

# Cell 2: Install
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers==4.36.0 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.0
!pip install pandas numpy scikit-learn matplotlib mlflow

# Cell 3: Verify GPU
!nvidia-smi

# Cell 4: Quick test (5 epochs, 5-10 phút)
!python src/timesfm_baseline/train_timesfm_lora.py \
    --data_path data/raw/prices/ACB_ohlcv.csv \
    --stock_id ACB \
    --output_dir results \
    --epochs 5

# Cell 5: Full training (70 epochs, 2-3 giờ)
!python src/timesfm_baseline/train_timesfm_lora.py \
    --multi_stock \
    --data_path data/raw/prices \
    --output_dir results \
    --epochs 70
```

---

## ✅ Verify Push Thành Công

**Vào:** `https://github.com/YOUR_USERNAME/stock_vol_prediction01`

Bạn sẽ thấy:
- ✅ 500+ files
- ✅ 5 commits
- ✅ Code + Data

---

## 🆘 Nếu Script Lỗi

**Dùng manual commands:**

```bash
cd D:\bmad-projects\stock_vol_prediction01
git remote add origin https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
git push -u origin master
```

---

## 📚 Guides (Chi Tiết)

- `QUICK_START_GITHUB.md` - 3-minute setup
- `CREATE_PUBLIC_REPO_GUIDE.md` - Full guide
- `docs/GITHUB_SETUP_GUIDE.md` - Troubleshooting
- `docs/GOOGLE_COLAB_TRAINING_GUIDE.md` - Colab training

---

**🎯 Bắt đầu ngay:**

1. Tạo repo: https://github.com/new
2. Tạo token: https://github.com/settings/tokens
3. Chạy script: `bash setup_github_repo.sh YOUR_USERNAME`
4. Train Colab: https://colab.research.google.com/

**🚀 Done in 5 minutes!**
