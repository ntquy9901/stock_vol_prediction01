# Python PATH Configuration Guide for Windows

**Issue:** Python command not found → Need to add Python to system PATH

---

## Step 1: Find Your Python Installation Location

**Check common Python installation paths:**

```powershell
# Check common locations
dir "C:\Python*" /s /b 2>nul | findstr "python.exe"
dir "C:\Program Files\Python*" /s /b 2>nul | findstr "python.exe"
dir "%USERPROFILE%\AppData\Local\Programs\Python*" /s /b 2>nul | findstr "python.exe"

# Or check where Python Launcher is
where py
where py
```

**Common Python installation paths:**
- `C:\Python311\python.exe` (Python 3.11)
- `C:\Python312\python.exe` (Python 3.12)
- `C:\Python310\python.exe` (Python 3.10)
- `C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe`

---

## Step 2: Add Python to PATH (Permanent Fix)

### Method A: Via Settings UI (Easiest)

1. **Open System Environment Variables**
   - Press `Windows Key + R`
   - Type: `sysdm.cpl` → Enter
   - Go to **Advanced** tab → **Environment Variables**

2. **Edit PATH**
   - Under **User variables** (not System variables)
   - Find `Path` variable → Click **Edit**
   - Click **New** → Add Python paths:
     
   **Add these 2 paths:**
   ```
   C:\Python311
   C:\Python311\Scripts
   ```
   
   *(Replace `C:\Python311` with your actual Python path found in Step 1)*

3. **Save changes**
   - Click **OK** on all dialogs
   - **Restart your terminal/command prompt**

4. **Verify**
   ```bash
   python --version
   # Should show: Python 3.11.x
   ```

### Method B: Via PowerShell (Admin)

```powershell
# Run as Administrator
$pythonPath = (Get-Command python).Source
$pythonDir = Split-Path $pythonPath -Parent

# Add to PATH (current session)
$env:Path += ";$pythonDir;$pythonDir\Scripts"

# Add to PATH (permanent - modifies registry)
[Environment]::SetEnvironmentVariable("Path", $env:Path, "User")

# Verify
python --version
```

---

## Step 3: Verify Python Installation

```bash
# Check Python version
python --version

# Check pip
python -m pip --version

# Check Python location
where python
where pip
```

**Expected output:**
```
Python 3.11.x
pip 23.x.x
C:\Python311\python.exe
C:\Python311\Scripts\pip.exe
```

---

## Step 4: Test Crawler (After PATH Fixed)

```bash
# Navigate to project
cd D:\bmad-projects\stock_vol_prediction01

# Test Python
python --version

# Test crawler (if python works)
python src/sentiment/data_collection/per_stock_crawl.py \
    --ticker VCB \
    --start 2026-06-01 \
    --end 2026-06-30
```

---

## Troubleshooting

**Problem:** "Python command not found" even after PATH fix

**Solution A: Restart Terminal**
- Close ALL terminal windows
- Open new terminal
- Try `python --version` again

**Solution B: Use Python Launcher (Workaround)**
```bash
# Use 'py' launcher (usually installed with Python)
py -m src.sentiment.data_collection.per_stock_crawl --ticker VCB --start 2026-06-01 --end 2026-06-30
```

**Solution C: Reinstall Python (Last Resort)**
- Download Python from python.org
- During installation, ✅ **CHECK** "Add Python to PATH"
- Restart terminal

---

## Quick Fix for This Session (Without Restart)

If you can't restart terminal right now, use full path:

```bash
# Replace with your Python path found in Step 1
C:\Python311\python.exe src/sentiment/data_collection/per_stock_crawl.py --ticker VCB --start 2026-06-01 --end 2026-06-30
```

---

## After PATH is Fixed

Once `python --version` works, run the crawler test:

```bash
cd D:\bmad-projects\stock_vol_prediction01

python src/sentiment/data_collection/per_stock_crawl.py \
    --ticker VCB \
    --start 2026-06-01 \
    --end 2026-06-30
```

**Expected:**
- Takes 2-5 minutes
- Creates: `data/sentiment/raw/VCB/2026-06.csv`
- Logs: ~20-50 articles for VCB

---

**Report back after fixing PATH and testing crawler!**
