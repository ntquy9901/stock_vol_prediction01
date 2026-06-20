# 🚀 Tạo Public GitHub Repository - Hướng Dẫn Chi Tiết

**Mục tiêu:** Tạo public repository và push toàn bộ TimesFM code + data để train trên Google Colab

**Thời gian:** 10-15 phút

---

## 📋 Tổng Quan

Bạn sẽ làm theo 3 bước:

1. **Tạo repository trên GitHub** (5 phút) - Manual qua browser
2. **Chạy script push code** (2 phút) - Tự động
3. **Verify và test** (3 phút) - Kiểm tra

---

## BƯỚC 1: Tạo Repository Trên GitHub (5 phút)

### 1.1. Đăng Nhập GitHub

1. Mở browser, vào: **https://github.com**
2. Đăng nhập với tài khoản của bạn

> **Chưa có tài khoản?** Click **"Sign up"** để đăng ký (miễn phí)

### 1.2. Tạo Repository Mới

1. Click **"+"** icon (góc trên bên phải) → **"New repository"**

2. Điền thông tin:

   ```
   ┌─────────────────────────────────────────────────────────┐
   │ Repository name *                                        │
   │ stock_vol_prediction01                                   │
   │                                                          │
   │ Description                                              │
   │ TimesFM 2.5 LoRA Fine-Tuning for VN30 Volatility       │
   │                                                          │
   │ ◉ Public  ○ Private                                     │
   │                                                          │
   │ ☐ Add a README file        ← KHÔNG CHECK!               │
   │ ☐ Add .gitignore            ← KHÔNG CHECK!               │
   │ ☐ Choose a license          ← KHÔNG CHECK!               │
   │                                                          │
   │         [Create repository]                              │
   └─────────────────────────────────────────────────────────┘
   ```

3. **QUAN TRỌNG:**
   - ✅ **Public** - để public repo
   - ❌ **KHÔNG** check "Add a README file" - đã có sẵn
   - ❌ **KHÔNG** check "Add .gitignore" - đã có sẵn
   - ❌ **KHÔNG** check "Choose a license" - thêm sau cũng được

4. Click **"Create repository"**

### 1.3. GitHub Sẽ Hiển Thị Setup Screen

GitHub sẽ show commands để push existing repository:

```
…or push an existing repository from the command line
git remote add origin https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
git branch -M main
git push -u origin main
```

**LƯU LẠI URL CỦA BẠN:**
```
https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
```

**TỚ DỪNG!** Đừng copy commands này - chúng ta sẽ dùng script automation tốt hơn ở Bước 2.

---

## BƯỚC 2: Push Code Với Script (2 phút)

### 2.1. Tạo Personal Access Token (BẮT BUỘC)

GitHub không còn chấp nhận password từ 2021. Bạn cần tạo token.

1. **Vào:** https://github.com/settings/tokens

2. Click **"Generate new token"** → **"Generate new token (classic)"**

3. Đền form:

   ```
   ┌───────────────────────────────────────────────┐
   │ Note (required)                               │
   │ Stock Volatility - Push Access                 │
   │                                               │
   │ Expiration (required)                         │
   │ 90 days [▼]                                   │
   │                                               │
   │ Select scopes                                 │
   │ ☑ repo [▼]  ← CHECK NÀY!                     │
   │   ☑ repo:status                              │
   │   ☑ repo_deployment                          │
   │   ☑ public_repo                              │
   │   ☑ repo:invite                              │
   │   ☑ security_events                         │
   │                                               │
   │         [Generate token]                      │
   └───────────────────────────────────────────────┘
   ```

4. Click **"Generate token"**

5. **COPY TOKEN NGAY LẬP TỨC!**
   - Token chỉ hiển thị 1 lần
   - Format: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - Lưu vào nơi an toàn (notepad, password manager)

### 2.2. Chạy Script Push

Tôi đã tạo script automation cho bạn. Chạy như sau:

**Mở Git Bash hoặc Terminal:**

```bash
cd D:\bmad-projects\stock_vol_prediction01

# Chạy script (thay YOUR_USERNAME)
bash setup_github_repo.sh YOUR_USERNAME
```

**Ví dụ:**
```bash
bash setup_github_repo.sh johndoe
```

**Script sẽ:**
1. ✅ Kiểm tra existing remotes
2. ✅ Thêm GitHub remote
3. ✅ Hiển thị thống kê repository
4. ✅ Push code (với upstream)
5. ✅ Hiển thị kết quả

### 2.3. Authentication

Khi push, GitHub sẽ yêu cầu xác thực:

```
Username for 'https://github.com': YOUR_USERNAME
Password for 'https://YOUR_USERNAME@github.com': 
```

**Điền:**
- Username: `YOUR_USERNAME` (GitHub username)
- Password: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` (Personal Access Token)

> **Lưu ý:** Password không hiển thị khi gõ - paste token và enter, ngay cả khi không thấy gì.

### 2.4. Kết Quả Script

Nếu thành công, bạn sẽ thấy:

```
✓ Successfully pushed to GitHub!

Repository URL: https://github.com/YOUR_USERNAME/stock_vol_prediction01

Next steps:
  1. Visit your repository on GitHub
  2. Verify all files are uploaded
  3. Clone on Google Colab:
     !git clone https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
```

---

## BƯỚC 3: Verify và Test (3 phút)

### 3.1. Verify Trên GitHub

1. Vào URL repository của bạn:
   ```
   https://github.com/YOUR_USERNAME/stock_vol_prediction01
   ```

2. Kiểm tra:
   - ✅ Folder structure hiển thị đúng
   - ✅ Code files có (`src/`, `tests/`, `docs/`)
   - ✅ Data files có (`data/raw/prices/*.csv`)
   - ✅ README.md hiển thị
   - ✅ 2 commits hiển thị:
     1. "Add TimesFM 2.5 LoRA implementation..."
     2. "Add GitHub setup guide..."

3. Click vào **"Insights"** → **"Commits"** để xem commit history

### 3.2. Test Clone

**Mở Git Bash hoặc Terminal:**

```bash
# Clone về thư mục test (không ảnh hưởng project chính)
cd D:\bmad-projects
git clone https://github.com/YOUR_USERNAME/stock_vol_prediction01.git test_clone

# Kiểm tra files
cd test_clone
ls -la

# Kiểm tra data
ls -la data/raw/prices/ | head -10

# Xóa thư mục test
cd ..
rm -rf test_clone
```

Expected output:
```
ACB_ohlcv.csv
BCM_ohlcv.csv
BID_ohlcv.csv
...
```

### 3.3. Copy Clone Command Cho Colab

Lưu lại command này cho Google Colab:

```bash
# Clone command cho Colab
!git clone https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
```

---

## 🎯 Training Trên Google Colab

Sau khi push xong, bạn có thể train ngay trên Colab:

### Mở Colab Notebook

1. Vào: https://colab.research.google.com/
2. Click **"New Notebook"**
3. Đặt tên: `TimesFM_Training.ipynb`

### Bật GPU

1. Click **Runtime** → **Change runtime type**
2. Chọn **T4 GPU** (free) hoặc **A100** (Pro)
3. Click **Save**

### Clone và Train

Copy paste code này vào Colab cells:

```python
# Cell 1: Clone repository
!git clone https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
%cd stock_vol_prediction01

# Cell 2: Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers==4.36.0 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.0
!pip install pandas numpy scikit-learn matplotlib mlflow

# Cell 3: Verify GPU
!nvidia-smi

# Cell 4: Quick test (5 epochs, ~5-10 phút)
!python src/timesfm_baseline/train_timesfm_lora.py \
    --data_path data/raw/prices/ACB_ohlcv.csv \
    --stock_id ACB \
    --output_dir results/timesfm_quick_test \
    --epochs 5 \
    --batch_size 32 \
    --context_len 64 \
    --horizon_len 5

# Cell 5: Full training (70 epochs, ~2-3 giờ)
!python src/timesfm_baseline/train_timesfm_lora.py \
    --multi_stock \
    --data_path data/raw/prices \
    --output_dir results/timesfm_full \
    --epochs 70 \
    --batch_size 32
```

---

## 🆘 Troubleshooting

### Lỗi: "remote origin already exists"

**Giải pháp:**

```bash
# Xóa remote cũ
git remote remove origin

# Chạy script lại
bash setup_github_repo.sh YOUR_USERNAME
```

### Lỗi: "Authentication failed"

**Nguyên nhân:** Sai username hoặc token

**Giải pháp:**

1. Kiểm tra username: `https://github.com/YOUR_USERNAME`
2. Kiểm tra token: Có scope `repo`?
3. Tạo token mới nếu cần

### Lỗi: "failed to push some refs"

**Nguyên nhân:** GitHub repo có files chưa có ở local

**Giải pháp:**

```bash
# Force push (CẨN THẢN! sẽ overwrite GitHub)
git push -u origin master --force
```

### Lỗi: Script không chạy được trên Windows

**Giải pháp:** Dùng commands trực tiếp:

```bash
cd D:\bmad-projects\stock_vol_prediction01

# Thay YOUR_USERNAME
git remote add origin https://github.com/YOUR_USERNAME/stock_vol_prediction01.git
git push -u origin master
```

### Lỗi: Repository chưa tạo trên GitHub

**Giải pháp:** Tạo repository trước:

1. Vào: https://github.com/new
2. Điền `stock_vol_prediction01`
3. **KHÔNG** check README, .gitignore, license
4. Click **"Create repository"**
5. Chạy script lại

---

## 📊 Repository Statistics

Sau khi push, repository của bạn sẽ có:

- **Total files:** ~500 files
- **Code lines:** ~450,000 lines
- **Languages:** Python, Markdown, Shell
- **Size:** ~10-15 MB
- **Commits:** 2 commits
- **Branches:** 1 branch (master)

**Content breakdown:**
```
src/                    - Source code (TimesFM, LSTM, HAR, etc)
tests/                  - Test suite (34 tests, 100% pass)
docs/                   - Documentation (guides, lessons learned)
data/                   - VN30 data (30 stocks, 2006-2026)
  ├── raw/prices/       - OHLCV data
  └── processed/        - Processed features
models/                 - Trained models (empty initially)
results/                - Training results (empty initially)
```

---

## ✅ Final Checklist

Trước khi train trên Colab:

- [ ] Repository đã tạo trên GitHub (public)
- [ ] Script đã chạy thành công
- [ ] Code đã push lên GitHub
- [ ] Files hiển thị đúng trên GitHub UI
- [ ] Clone test thành công
- [ ] Personal Access Token đã lưu an toàn
- [ ] Google Colab đã mở với GPU enabled
- [ ] Clone command đã copy cho Colab

---

## 📚 Related Documents

- `docs/GOOGLE_COLAB_TRAINING_GUIDE.md` - Hướng dẫn Colab chi tiết
- `docs/GITHUB_SETUP_GUIDE.md` - GitHub technical guide
- `docs/LESSONS_LEARNED_TIMESFM_ADVERSARIAL_REVIEWS.md` - 40 bugs + fixes

---

## 🎉 Next Steps

Sau khi push xong:

1. ✅ Share GitHub repository URL với team
2. ✅ Mở Colab và bắt đầu training
3. ✅ Save results vào Google Drive
4. ✅ Pull results về local machine
5. ✅ Update README.md với results

---

**Created:** 2026-06-20  
**Status:** ✅ Ready to Execute  
**Time Estimate:** 10-15 minutes

**Need help?** Check `docs/GITHUB_SETUP_GUIDE.md` for detailed troubleshooting
