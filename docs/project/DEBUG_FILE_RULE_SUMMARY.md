# Debug File Rule - Implementation Summary

**Date:** 2026-06-17
**Rule:** Debug files in experiment folder
**Status:** ✅ Implemented in both projects

---

## What Was Added

### 1. Updated stock_vol_prediction01/CLAUDE.md

**Section:** Rule 7: Experimental Code Rule
**Enhanced:** Added debug file guidelines

**New Content:**
```
Rule 7: Experimental Code Rule

✅ Debug scripts MUST be in src/experiment/
✅ Debug files are TEMPORARY - delete after issue resolved
✅ Naming: Start with debug_ or test_ prefix

Examples:
  ✅ src/experiment/debug_training_failure.py
  ✅ src/experiment/test_scaling_bug.py
  ❌ src/lstm_baseline/debug_xxx.py (wrong location)

Debug File Guidelines:
- Purpose: Quick investigation, NOT production code
- Lifetime: Temporary - delete after bug fixed
- Documentation: Minimal - comment what issue being debugged
- Cleanup: Remove from git after resolving
```

### 2. Updated ml-ds-common-rules/COMMON_RULES.md

**Section:** Experimentation Code
**Added:** Debug and Temporary Files subsection

**New Content:**
```markdown
### Debug and Temporary Files

✅ GOOD Debug Practices:
- ALL debug files in experiments/ or src/experiment/
- Clear naming: debug_xxx.py, test_xxx.py, verify_xxx.py
- Document issue being investigated
- Delete after resolution

❌ BAD Debug Practices:
- Debug files cluttering main source
- Multiple debug versions left behind
- Unclear which file is current

Debug File Lifecycle:
1. Create in src/experiment/ with clear name
2. Use to investigate/verify specific issue
3. Document what being debugged
4. Resolve or understand root cause
5. Delete file (or archive if reference needed)
```

---

## Purpose & Benefits

### Why This Rule?

1. **Keep main source clean** - Production code folders uncluttered
2. **Clear separation** - Debug vs production code obvious
3. **Temporary nature** - Debug files marked as ephemeral
4. **Easy cleanup** - Know what can be safely deleted
5. **Prevents confusion** - No ambiguity about current code

### Benefits

| Before (Debug Anywhere) | After (Debug in experiment/) |
|-------------------------|------------------------------|
| `src/lstm_baseline/debug.py` | `src/experiment/debug_training.py` |
| Clutters main folders | Clean separation |
| Unsure what to delete | Clear cleanup target |
| Production + debug mixed | Obvious what's temp |

---

## File Organization

### Correct Structure:

```
project/
├── src/
│   ├── experiment/              # ALL debug/test files
│   │   ├── debug_training_failure.py
│   │   ├── test_scaling_bug.py
│   │   └── verify_gradient_flow.py
│   ├── lstm_baseline/          # Production code only
│   │   ├── model.py
│   │   ├── train.py
│   │   └── dataset.py
│   └── common/                 # Production utilities
│       └── evaluation.py
└── experiments/                # Experiment configs
    └── exp_1_config.json
```

### Debug File Template:

```python
"""
Debug: [Short description of issue]

Issue: [What problem being investigated]
Root cause: [What was found]
Status: [Resolved / Pending]
Date: [YYYY-MM-DD]
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Investigation code...

if __name__ == "__main__":
    print("Investigation complete")
```

---

## Examples from Current Project

### Debug Files Created:

1. **`src/experiment/debug_training_failure.py`**
   - Purpose: Investigate LSTM training failure
   - Issue: QLIKE 0.74, Dir Acc 0.5%
   - Root cause: Target scaler bug (mean 0.74, std 2.34 instead of ~0, ~1)
   - Status: Root cause identified, pending fix

### Files Moved:

- ❌ Was: `src/lstm_baseline/debug_training_failure.py`
- ✅ Now: `src/experiment/debug_training_failure.py`

---

## Cleanup Guide

### When to Delete Debug Files:

✅ **Delete immediately when:**
- Bug is fixed and verified
- Root cause understood and documented
- Alternative production solution implemented
- Debug code no longer needed

✅ **Keep temporarily when:**
- Issue not fully resolved
- Need reference for similar future issues
- Lesson learned documentation in progress

❌ **Never keep when:**
- Bug fixed and production working
- Duplicate of production code
- No longer relevant to current project

### Cleanup Commands:

```bash
# List all debug files
find src/experiment/ -name "debug_*.py" -o -name "test_*.py"

# Archive before deletion (optional)
mkdir -p archive/debug_files_$(date +%Y%m%d)
mv src/experiment/debug_*.py archive/debug_files_$(date +%Y%m%d)/

# Delete old debug files (older than 30 days)
find src/experiment/ -name "debug_*.py" -mtime +30 -delete
```

---

## Integration with Other Rules

### Related Rules:

1. **Rule 6: Baseline Organization**
   - Each baseline has own folder
   - Debug files DON'T belong in baseline folders

2. **Rule 8: Timestamp Naming**
   - Debug files can skip timestamp (temporary nature)
   - BUT if kept long-term, should add timestamp

3. **File Management (ml-ds-common-rules)**
   - Archive old versions, don't delete
   - Debug files are exception - can delete after use

---

## Self-Reflection & Learning

### Issue That Led to This Rule:

**Problem:** Debug file `debug_training_failure.py` created in `src/lstm_baseline/`

**Impact:**
- Cluttered production source folder
- Unclear if file is production or debug
- No guidance on cleanup

**Resolution:**
- Moved to `src/experiment/`
- Added clear rule for future debug files
- Documented cleanup process

### Prevention:

**For Future Projects:**
1. ✅ Always create debug files in `src/experiment/`
2. ✅ Use `debug_` or `test_` prefix
3. ✅ Document issue being investigated
4. ✅ Delete after resolution
5. ✅ Keep main source folders clean

---

## Verification

### Quick Check:

```bash
# Verify no debug files in main source
find src/lstm_baseline/ src/common/ src/har_baseline/ -name "debug_*.py" -o -name "test_*.py"

# Should return nothing if clean

# Verify debug files in experiment folder
find src/experiment/ -name "debug_*.py" -o -name "test_*.py"

# Should list debug files
```

---

## Summary

✅ **Rule added to stock_vol_prediction01/CLAUDE.md** - Rule 7 enhanced
✅ **Rule added to ml-ds-common-rules/COMMON_RULES.md** - New subsection
✅ **Debug file moved** - `src/experiment/debug_training_failure.py`
✅ **Guidelines provided** - Lifecycle, cleanup, examples
✅ **Integration documented** - Related rules, verification

**Status:** Ready for adoption across all ML/DS projects!

---

**Last Updated:** 2026-06-17
**Version:** 1.0 - Initial implementation
