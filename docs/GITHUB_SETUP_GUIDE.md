# GitHub Repository Setup Guide

**Purpose:** Tạo repository mới trên GitHub và push TimesFM code + data

---

## Bước 1: Tạo Repository trên GitHub

### 1.1. Đăng nhập GitHub

1. Vào [https://github.com](https://github.com)
2. Đăng nhập với tài khoản của bạn
3. Nếu chưa có tài khoản, click **"Sign up"** để đăng ký

### 1.2. Tạo Repository Mới

1. Click **+** icon ở góc trên bên phải → **"New repository"**
2. Điền thông tin sau:

   ```
   Repository name: stock_vol_prediction01
   Description: TimesFM 2.5 LoRA Fine-Tuning for VN30 Volatility Prediction
   ```
   
3. **QUAN TRỌNG:**
   - ❌ **KHÔNG** check "Add a README file" (đã có README.md sẵn)
   - ❌ **KHÔNG** check "Add .gitignore" (đã có .gitignore)
   - ❌ **KHÔNG** check "Choose a license" (có thể thêm sau)
   
   **Tại sao?** Vì nếu check các tùy chọn này, GitHub sẽ tạo file mới và gây conflict với code đã có!

4. Chọn **Public** hoặc **Private**:
   - **Public**: Ai cũng xem được (good cho portfolio)
   - **Private**: Chỉ bạn xem được (good cho bảo mật)

5. Click **"Create repository"**

### 1.3. Lưu URL của Repository

Sau khi tạo, GitHub sẽ hiển thị URL như sau:

```
https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
```

**Copy URL này** - sẽ cần ở bước tiếp theo!

---

## Bước 2: Thêm Remote và Push Code

### 2.1. Kiểm Tra Remote Hiện Tại

```bash
cd D:\bmad-projects\stock_vol_prediction01
git remote -v
```

Nếu không có output nào = chưa có remote (good!)

### 2.2. Thêm GitHub Remote

**Thay thế `YOUR_USERNAME` bằng username GitHub của bạn:**

```bash
# Ví dụ: git remote add origin https://github.com/johndoe/stock_vol_prediction01.git
git remote add origin https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
```

**Xác nhận remote đã thêm:**

```bash
git remote -v
```

Expected output:
```
origin  https://github.com/YOUR_USERNAME/stock_vol_prediction01.git (fetch)
origin  https://github.com/YOUR_USERNAME/stock_vol_prediction01.git (push)
```

### 2.3. Push Code Lên GitHub

```bash
# Push lần đầu (set upstream)
git push -u origin master
```

**Lưu ý:**
- `-u` flag = set upstream (lần sau chỉ cần `git push`)
- `master` = tên branch hiện tại
- Quá trình này sẽ mất 1-2 phút tùy tốc độ internet

### 2.4. Nhập Username và Password

GitHub sẽ yêu cầu xác thực:

**Option A: Personal Access Token (Recommended)**

GitHub không còn chấp nhận password từ 2021. Bạn cần tạo Personal Access Token:

1. Vào [https://github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Điền:
   ```
   Note: Stock Volatility Prediction - Push Access
   Expiration: 90 days
   ```
4. Check scopes:
   - ✅ **repo** (full control of private repositories)
   - ✅ **workflow** (if you want to use GitHub Actions)
5. Click **"Generate token"**
6. **Copy token** (chỉ hiện 1 lần!)

**Khi push:**
- Username: `YOUR_USERNAME`
- Password: `YOUR_PERSONAL_ACCESS_TOKEN` (không phải password GitHub!)

**Option B: GitHub CLI (Easier)**

Nếu cài GitHub CLI:

```bash
# Login với GitHub CLI
gh auth login

# Push code (sẽ không cần username/password)
git push -u origin master
```

---

## Bước 3: Xác Nhận Push Thành Công

### 3.1. Kiểm Tra trên GitHub

1. Vào repository URL: `https://github.com/YOUR_USERNAME/stock_vol_prediction01`
2. Bạn sẽ thấy:
   - Tất cả files đã được upload
   - Commit message: "Add TimesFM 2.5 LoRA implementation with complete codebase and data"
   - File count: ~494 files
   - Size: ~10-15 MB

### 3.2. Kiểm Tra Git Log

```bash
git log --oneline -1
```

Expected output:
```
8924215 Add TimesFM 2.5 LoRA implementation with complete codebase and data
```

### 3.3. Kiểm Tra Remote Branch

```bash
git branch -vv
```

Expected output:
```
* master 8924215 [origin/master] Add TimesFM 2.5 LoRA implementation...
```

---

## Bước 4: Clone và Train Trên Google Colab

### 4.1. Mở Google Colab

1. Vào [https://colab.research.google.com/](https://colab.research.google.com/)
2. Click **"New Notebook"**
3. Đặt tên: `TimesFM_Training.ipynb`

### 4.2. Bật GPU

1. Click **Runtime** → **Change runtime type**
2. Chọn **T4 GPU** (free) hoặc **A100** (Pro)
3. Click **Save**

### 4.3. Clone Repository

```python
# Clone repository từ GitHub của bạn
!git clone https://github.com/YOUR_USERNAME/stock_vol_prediction01.git

# Chuyển vào thư mục
%cd stock_vol_prediction01

# Kiểm tra files
!ls -la
```

### 4.4. Install Dependencies

```python
# Install PyTorch và các thư viện cần thiết
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers==4.36.0 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.0
!pip install pandas numpy scikit-learn matplotlib mlflow

# Verify GPU
!nvidia-smi
```

### 4.5. Chạy Quick Test (5 epochs)

```python
# Test với 1 stock, 5 epochs (~5-10 phút)
!python src/timesfm_baseline/train_timesfm_lora.py \
    --data_path data/raw/prices/ACB_ohlcv.csv \
    --stock_id ACB \
    --output_dir results/timesfm_quick_test \
    --epochs 5 \
    --batch_size 32 \
    --context_len 64 \
    --horizon_len 5 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42
```

### 4.6. Chạy Full Training (70 epochs)

```python
# Full training với tất cả stocks (~2-3 giờ)
!python src/timesfm_baseline/train_timesfm_lora.py \
    --multi_stock \
    --data_path data/raw/prices \
    --output_dir results/timesfm_full_$(date +%Y%m%d_%H%M%S) \
    --epochs 70 \
    --batch_size 32 \
    --context_len 64 \
    --horizon_len 5 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42
```

### 4.7. Lưu Kết Quả Vào Google Drive

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy results
!cp -r results/timesfm_full_* /content/drive/MyDrive/TimesFM_Results/

# Verify
!ls -la /content/drive/MyDrive/TimesFM_Results/
```

---

## Troubleshooting

### Lỗi 1: "fatal: remote origin already exists"

**Giải pháp:**

```bash
# Xóa remote cũ
git remote remove origin

# Thêm remote mới
git remote add origin https://github.com/YOUR_USERNAME/stock_vol_prediction01.git

# Push
git push -u origin master
```

### Lỗi 2: "Authentication failed"

**Giải pháp:** Sử dụng Personal Access Token (xem Bước 2.3)

### Lỗi 3: "failed to push some refs"

**Nguyên nhân:** Có thay đổi trên GitHub chưa pull về

**Giải pháp:**

```bash
# Pull trước (để merge changes)
git pull origin master --allow-unrelated-histories

# Sau đó push
git push -u origin master
```

### Lỗi 4: GitHub không cho push file lớn (>100MB)

**Nguyên nhân:** GitHub limit 100MB per file

**Giải pháp A: Sử dụng Git LFS**

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.csv"
git lfs track "*.parquet"

# Commit lại
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push origin master
```

**Giải pháp B: Xóa files lớn**

```bash
# Xóa khỏi git (nhưng giữ local)
git rm --cached data/processed_all/*.csv

# Commit
git commit -m "Remove large files from git"
git push origin master
```

### Lỗi 5: Timeout khi push

**Giải pháp:** Tăng timeout và retry

```bash
# Tăng timeout (5 phút)
git config http.timeout 300

# Push lại
git push -u origin master
```

---

## Checklist: Xác Nhân Hoàn Tất

Trước khi train trên Colab, kiểm tra:

- [ ] Repository đã tạo trên GitHub
- [ ] Remote đã thêm (`git remote -v`)
- [ ] Code đã push (`git push -u origin master`)
- [ ] Files hiển thị trên GitHub (494 files)
- [ ] Data files đã có (data/raw/prices/*.csv)
- [ ] Documentation files đã có (docs/ folder)
- [ ] Commit message hiển thị đúng

---

## Command Reference Sheet

**Tóm tắt commands:**

```bash
# 1. Tạo repository trên GitHub (manual qua browser)

# 2. Thêm remote
git remote add origin https://github.com/YOUR_USERNAME/stock_vol_prediction01.git

# 3. Push code
git push -u origin master

# 4. Xác nhận
git log --oneline -1
git remote -v
git branch -vv

# 5. Trên Colab - Clone
!git clone https://github.com/YOUR_USERNAME/stock_vol_prediction01.git

# 6. Trên Colab - Train
!python stock_vol_prediction01/src/timesfm_baseline/train_timesfm_lora.py \
    --data_path stock_vol_prediction01/data/raw/prices/ACB_ohlcv.csv \
    --stock_id ACB \
    --output_dir results \
    --epochs 5
```

---

## Next Steps

Sau khi push thành công:

1. **Test nhanh trên Colab:**
   - Clone repository
   - Chạy quick test (5 epochs)
   - Verify kết quả

2. **Full training:**
   - Chạy với tất cả 30 stocks
   - 70 epochs (~2-3 giờ)
   - Lưu kết quả vào Google Drive

3. **Pull results về local:**
   - Download từ Google Drive
   - Hoặc pull từ GitHub nếu commit results

---

**Created:** 2026-06-20  
**Status:** ✅ Ready to Execute  
**Estimated Time:** 15 minutes (setup) + 5-10 minutes (quick test) + 2-3 hours (full training)
