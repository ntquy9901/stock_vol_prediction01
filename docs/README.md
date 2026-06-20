# Documentation Structure

This directory contains all project documentation organized by topic.

## Directory Organization

### `docs/lstm/`
LSTM model documentation:
- Architecture analysis and design
- Implementation guides
- Comparison with other models

### `docs/har/`
HAR-R baseline documentation:
- Feature engineering guides
- Model implementation details
- Performance analysis

### `docs/project/`
Project management documentation:
- Project context and overview
- Refactoring summaries
- Development workflows

### `docs/baseline/`
Documentation for other baseline models:
- Alternative approaches
- Experimental models
- Comparative studies

### `docs/common-rules/`
ML/DS common clean code rules:
- General coding standards
- Best practices
- Quick reference guides

## File Naming Conventions

- Use descriptive names: `MODEL_TOPIC.md`
- Examples:
  - ✅ `ENHANCED_LSTM_GUIDE.md`
  - ✅ `HAR_FEATURES_GUIDE.md`
  - ❌ `guide.md` (too generic)

## Creating New Documentation

When adding new documentation:

1. **Choose appropriate subdirectory** based on topic
2. **Use descriptive filename** with model/topic prefix
3. **Add clear title and description** in file header
4. **Update this README** if adding new categories

## Root Directory Files

Only these files should be in project root:
- **README.md**: Project overview and quick start
- **CLAUDE.md**: Project rules and context

All other .md files should be in `docs/` subdirectories.
