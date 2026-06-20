# Timestamp Naming Rule - Implementation Summary

**Date:** 2026-06-17
**Rule:** Rule 8 - Timestamp Naming Rule
**Status:** ✅ Implemented in both projects

---

## What Was Added

### 1. Updated stock_vol_prediction01/CLAUDE.md

**Section:** 3. Project Organization Rules
**Rule Number:** Rule 8 (was 7 Golden Rules, now 8 Golden Rules)

**Rule Content:**
```
Rule 8: Timestamp Naming Rule ⭐ CRITICAL

✅ ALL generated files MUST include timestamp in filename
✅ Applies to: Markdown reports, result files, model files, logs, etc.
✅ Format: {name}_{YYYY-MM-DD_HHMMSS}.{ext}
❌ NO static filenames without timestamp (except source code)

Purpose: Track versions, identify latest results, prevent overwriting
```

**Examples by file type:**

**Markdown Reports:**
```
✅ DATA_ORGANIZATION_REPORT_2026-06-17_234500.md
❌ DATA_ORGANIZATION_REPORT.md
```

**Result Files:**
```
✅ test_metrics_2026-06-17_225000.csv
✅ training_curves_2026-06-17_225000.png
❌ test_metrics.csv
```

**Model Files:**
```
✅ best_simple_lstm_2026-06-17_225000.pth
✅ training_history_2026-06-17_225000.json
❌ best_simple_lstm.pth
```

**Exceptions (NO timestamp):**
- Source code (.py in src/)
- Configuration (config.yaml, .env)
- README files

### 2. Updated ml-ds-common-rules/COMMON_RULES.md

**Section:** File Management and Archiving → Generated Files Naming Convention

**Added Content:**
- CRITICAL rule with timestamp requirement
- Python implementation examples
- FileHandler class for automatic timestamping
- Directory organization examples
- Quick verification script

**Implementation Example:**
```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

# Result files
results_file = f"experiment_results_{timestamp}.csv"

# Model files
model_file = f"best_model_{timestamp}.pth"

# Log files
log_file = f"training_log_{timestamp}.txt"
```

---

## Purpose & Benefits

### Why This Rule?

1. **Version Tracking:** Know which file is the latest version
2. **Prevent Overwriting:** Don't lose previous results
3. **Audit Trail:** Track when files were generated
4. **Reproducibility:** Link files to specific experiments
5. **Comparison:** Easy to compare files across time

### Benefits

| Before (Static Names) | After (Timestamped) |
|----------------------|---------------------|
| `results.csv` → Overwritten | `results_2026-06-17_225000.csv` → Preserved |
| `model.pth` → Which version? | `model_2026-06-17_225000.pth` → Clear |
| `log.txt` → Lost history | `log_2026-06-17_225000.txt` → Full history |
| `report.md` → Can't compare | `report_2026-06-17_225000.md` → Easy compare |

---

## Implementation Guide

### For stock_vol_prediction01 Project

**Files that need updating:**

1. **Reports (docs/):**
   - `DATA_ORGANIZATION_REPORT.txt` → `DATA_ORGANIZATION_REPORT_2026-06-17_234500.txt`

2. **Results (results/):**
   - Already using timestamped folders: `results/simple_lstm_{timestamp}/`
   - Individual files should also have timestamps

3. **Models (models/):**
   - Already using timestamped folders: `models/lstm_baseline_{timestamp}/`
   - Model files should also have timestamps

### For ml-ds-common-rules Project

**Reusable across all projects:**

```python
# Use this pattern in any ML/DS project
from datetime import datetime

class GeneratedFileHandler:
    """Handle all generated files with automatic timestamping."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def save_results(self, data: dict, filename: str):
        """Save results with timestamp."""
        timestamped_filename = f"{filename}_{self.timestamp}.csv"
        # ... save logic
```

---

## Verification

### Quick Check Script

```bash
# Check if generated files have timestamps
find results/ models/ logs/ -type f \
  \( -name "*.csv" -o -name "*.pth" -o -name "*.png" -o -name "*.txt" \) | \
  while read f; do
    if ! echo "$f" | grep -qE "_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}"; then
      echo "WARNING: No timestamp in: $f"
    fi
  done
```

---

## Impact

### Projects Affected

1. **stock_vol_prediction01** ✅
   - CLAUDE.md updated with Rule 8
   - 8 Golden Rules (was 7)

2. **ml-ds-common-rules** ✅
   - COMMON_RULES.md updated with Generated Files Naming Convention
   - Reusable across all future projects

3. **Future projects** 🎯
   - Can adopt rule from ml-ds-common-rules
   - Universal standard for file naming

### Code Quality Impact

- ✅ Better version control
- ✅ Clear audit trail
- ✅ Prevents accidental data loss
- ✅ Easier experiment tracking
- ✅ Standardized across projects

---

## Next Steps

1. **Update existing files** (optional):
   - Rename static files with timestamps
   - Update code to use timestamped filenames

2. **Add validation** (optional):
   - Pre-commit hook to check timestamps
   - CI/CD check for timestamp compliance

3. **Document in README** (optional):
   - Add note about timestamp requirement
   - Include examples

---

## Summary

✅ **Rule 8 added to stock_vol_prediction01/CLAUDE.md**
✅ **Generated Files Naming Convention added to ml-ds-common-rules**
✅ **Universal standard for timestamping generated files**
✅ **Implementation examples provided**
✅ **Verification script included**

**Status:** Ready for adoption across all ML/DS projects!

---

**Last Updated:** 2026-06-17
**Version:** 1.0
