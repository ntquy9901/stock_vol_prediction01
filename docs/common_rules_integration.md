# ML/DS Common Rules Integration Strategy

**Project:** Stock Volatility Prediction for VN30 Stocks  
**Date:** 2026-06-15  
**Decision:** Integrate common rules into project context and documentation

---

## 🎯 **ANALYSIS: Common Rules Reference in Requirements**

### **Current Status:**
```python
# In docs/requirements.md - Non-Functional Requirements
NFR-1: Code Quality Standards
- Follow ML/DS Common Rules (`D:\bmad-projects\ml-ds-common-rules`)
- 85%+ test coverage maintained
- Comprehensive tests, documented code
- Production-ready quality
```

**Problem:** Common rules exist externally but not integrated into project workflow.

---

## 📊 **RECOMMENDATION: YES - INTEGRATE COMMON RULES**

### **✅ WHY INTEGRATE?**

**1. Consistency & Standards:**
- ✅ Ensure team follows proven coding practices
- ✅ Maintain code quality across development
- ✅ Facilitate code review and collaboration

**2. AI Assistant Context:**
- ✅ Make rules accessible to Claude Code during development
- ✅ Provide consistent guidance in code generation
- ✅ Enable automatic rule checking during work

**3. Project Success:**
- ✅ Reduce bugs through better practices
- ✅ Improve maintainability
- ✅ Accelerate onboarding for team members

**4. Documentation Completeness:**
- ✅ Central location for all project standards
- ✅ Easy reference for all stakeholders
- ✅ Clear expectation setting

---

## 🔧 **RECOMMENDED INTEGRATION APPROACH**

### **🏆 METHOD 2: GIT SUBMODULE (RECOMMENDED)**

**Best cho:** Long-term project, want automatic updates, team collaboration

**Implementation Steps:**

```bash
# 1. Add ML/DS common rules as git submodule
cd D:\bmad-projects\stock_vol_prediction01
git submodule add https://github.com/ntquy9901/ml-ds-common-rules.git docs/common-rules

# 2. Commit submodule
git add .gitmodules docs/common-rules
git commit -m "Add ML/DS common rules as git submodule"

# 3. Update later (get latest improvements)
git submodule update --remote docs/common-rules
```

**Benefits:**
- ✅ Automatic updates from upstream
- ✅ Clean separation of common vs project-specific rules
- ✅ Easy to maintain and sync
- ✅ Version control for rules themselves

---

## 📁 **PROJECT STRUCTURE WITH INTEGRATION**

### **Recommended File Organization:**

```
stock_vol_prediction01/
├── docs/
│   ├── common-rules/              # 🆕 Git submodule (ML/DS rules)
│   │   ├── COMMON_RULES.md
│   │   ├── QUICK_REFERENCE.md
│   │   └── CLAUDE_TEMPLATE.md
│   ├── requirements.md            # ✅ Already exists
│   ├── technical_config.md         # ✅ Already exists  
│   ├── loss_functions.md          # ✅ Already exists
│   ├── data_schema.md              # ✅ Already exists
│   └── single_horizon_strategy.md  # ✅ Already exists
├── CLAUDE.md                      # 🆕 Main project documentation
├── project-context.md              # 🆕 AI assistant context
├── src/
├── data/
└── experiments/
```

---

## 📝 **CLAUDE.md STRUCTURE (RECOMMENDED)**

### **Create Main Project Documentation:**

```markdown
# Stock Volatility Prediction - VN30

**Project:** Multi-horizon volatility forecasting cho VN30 stocks  
**Date:** 2026-06-15  
**Methodology:** HAR-R with Parkinson volatility, 5-day focus initially

---

## 1. Project Overview

### **Objective**
Build robust volatility prediction system cho 30 VN30 stocks using daily OHLCV data, implementing HAR methodology adapted cho daily frequency.

### **Key Requirements**
- **Primary Target:** 5-day ahead volatility forecast ✅ CURRENT FOCUS
- **Secondary Targets:** 1, 10, 22-day forecasts (future expansion)
- **Input Data:** Daily OHLCV (30 stocks, 2006-2026)
- **Approach:** HAR-R baseline, enhanced with ML models (LSTM, GNN, TimesFM)

### **Success Criteria**
- RMSE < 0.20 cho 5-day forecasts
- Directional accuracy > 55%
- QLIKE loss function (academic standard)
- 85%+ test coverage
- ML/DS common rules compliance

---

## 2. Common ML/DS Research Rules ⭐

**This project follows the ML/DS common clean code rules.**

**Reference:** `docs/common-rules/COMMON_RULES.md`  
**Quick Reference:** `docs/common-rules/QUICK_REFERENCE.md`

### **Core Principles**
1. **Code is read much more than written** - Write for future readers
2. **Leave code better than you found it** - Boy scouts rule
3. **Keep it simple** - Simple > Clever
4. **Match quality to maturity** - Don't over-engineer POCs

### **Critical Rules cho This Project**

**Naming Conventions:**
```python
# ✅ CORRECT
volatility_forecast = model.predict(data)  # Descriptive
train_accuracy = 0.95                    # Clear meaning
parkinson_volatility = calculate_volatility(data)  # Specific

# ❌ AVOID  
vol = model.predict(data)                 # Too generic
acc = 0.95                              # Abbreviated  
pred = forecast(data)                    # Unclear
```

**Function Design:**
- Small functions (< 30 lines)
- Single responsibility
- Few parameters (< 3 preferred)
- No side effects

**Code Organization:**
- One concern per file
- High-level functions at top, low-level below
- Related code grouped together

**Documentation:**
- Explain WHY not HOW in comments
- Use docstrings cho public functions
- Delete outdated comments

**Testing:**
- 85%+ coverage target
- Unit tests cho data processing (90%)
- Model training (80%)
- Integration tests (30%)
- E2E tests (10%)

**Research Best Practices:**
- Version control all experiments
- Track hyperparameters with MLflow
- Use fixed random seeds (42)
- Save intermediate checkpoints
- Plot learning curves (mandatory)

### **Project-Specific Adaptations**

**Financial ML Context:**
```python
# Time series specific
train_split = int(len(data) * 0.8)  # Chronological split
assert train_index < test_index     # Validate temporal order

# Volatility specific  
volatility_clipped = np.clip(volatility, -5, 5)  # Prevent extreme values
returns = np.log(prices).diff()  # Log returns not raw returns

# Financial data validation
assert all(ohlc['high'] >= ohlc['close'])  # Validate OHLC consistency
assert all(ohlc['close'] >= ohlc['low'])   # Validate price relationships
```

---

## 3. Project-Specific Rules

### **Data Processing Rules**
- **Temporal integrity:** Never use future data in training
- **Validation checks:** Ensure OHLCV consistency (High ≥ Close ≥ Low)
- **Missing data:** Forward-fill max 5 days, then interpolation
- **Outlier detection:** Statistical methods for price anomalies

### **Feature Engineering Rules**
- **HAR features:** Always use confirmed 22-day monthly window
- **Volatility measures:** Parkinson estimator as primary
- **Target creation:** shift(-5) cho 5-day forecasts only (Phase 1)
- **Feature naming:** Use `har_*`, `target_*` prefixes for clarity

### **Model Training Rules**
- **Fixed seeds:** random.seed(42), np.random.seed(42), torch.manual_seed(42)
- **Cross-validation:** Rolling window for time series
- **Baseline comparison:** Always compare against random walk baseline
- **Overfitting detection:** Monitor train/test performance gap

### **Documentation Rules**
- **Function docstrings:** All public functions必须有 docstrings
- **Experiment tracking:** MLflow for all experiments
- **Code comments:** Explain financial context, not obvious operations
- **README files:** Each module must have README.md

### **Testing Rules**
- **Data quality tests:** Validate input data integrity
- **Model reproducibility:** Same seed → same results
- **Performance tests:** Inference < 100ms per stock
- **Coverage targets:** 85%+ overall, 90% cho critical paths

---

## 4. Technical Architecture

### **System Architecture**
```
Data Input → HAR Features → Model Training → 5-Day Predictions
     ↓              ↓              ↓            ↓
  OHLCV        Parkinson Vol    HAR-R/LSTM    Evaluation
```

### **Technology Stack**
- **Language:** Python 3.11+
- **Data:** pandas, numpy
- **ML:** scikit-learn, PyTorch
- **Evaluation:** QLIKE, RMSE, MAE, R²
- **Experiment Tracking:** MLflow

### **Key Design Decisions**
- **5-day focus first:** Validate single horizon before multi-horizon
- **22-day consistency:** Monthly HAR feature = monthly forecast horizon
- **QLIKE primary:** Academic standard cho volatility loss
- **Parkinson estimator:** Range-based volatility cho daily data

---

## 5. Development Workflow

### **Code Review Process**
1. **Self-review:** Check against QUICK_REFERENCE.md
2. **Automated checks:** Run tests, coverage check
3. **Peer review:** Team member validates against common rules
4. **Approval:** Merge only when all checks pass

### **Quality Gates**
- **Pre-commit:** Tests pass, coverage sufficient
- **Pre-merge:** Code review approval, no critical issues
- **Pre-deploy:** Performance benchmarks met, documentation complete

### **Experiment Workflow**
1. **Setup:** Fixed seeds, create MLflow run
2. **Training:** Monitor learning curves, save checkpoints
3. **Evaluation:** Calculate metrics (QLIKE, RMSE, etc.)
4. **Documentation:** Update experiment logs, save artifacts

---

## 6. Communication & Collaboration

### **Team Standards**
- **Code reviews:** Use common rules checklist
- **Meetings:** Reference common rules when discussing code quality
- **Onboarding:** New members read common rules first (Day 1)

### **Documentation Maintenance**
- **Weekly review:** Check if rules need updates
- **Monthly sync:** Update submodule with latest common rules
- **Quarterly audit:** Remove deprecated practices, add new learnings

---

## 7. Tools & Automation

### **Automated Quality Checks**
```bash
# Pre-commit hooks (using common-rules validation)
python -m ml_ds_common_rules.code_quality.check_imports src/
python -m ml_ds_common_rules.testing.pre_flight_env
pytest --cov=src --cov-report=html
```

### **Code Review Integration**
```bash
# Automated checklist during PR
git hook pre-commit
→ Runs common rules validators
→ Checks test coverage
→ Validates docstrings
→ Only allows merge if all pass
```

---

## 8. Monitoring & Improvement

### **Quality Metrics**
- **Code coverage:** Track 85%+ target over time
- **Code review cycle time:** Monitor efficiency
- **Bug rate:** Track defects post-deployment
- **Technical debt:** Log and address systematically

### **Continuous Improvement**
- **Retrospectives:** Discuss what's working, what's not
- **Rule updates:** Propose improvements to common rules repo
- **Training:** Regular sessions on clean code practices

---

## 🎯 **IMPLEMENTATION PRIORITY**

### **Phase 1: Essential Integration (Week 1)**

**Step 1: Add Git Submodule**
```bash
git submodule add https://github.com/ntquy9901/ml-ds-common-rules.git docs/common-rules
git commit -m "Add ML/DS common rules integration"
```

**Step 2: Create CLAUDE.md**
- Project overview (this document)
- Reference to common rules
- Project-specific rules

**Step 3: Create project-context.md**
```bash
# For AI assistant context
# Make project-specific rules easily accessible
```

### **Phase 2: Team Adoption (Week 2)**

**Step 1: Team Training**
- Review common rules in team meeting
- Discuss project-specific adaptations
- Set up automated checks

**Step 2: Gradual Implementation**
- Apply to new code only
- Use boy scouts rule for existing code
- Monitor and refine approach

### **Phase 3: Full Integration (Week 3-4)**

**Step 1: Quality Gates**
- Pre-commit hooks with common rules validation
- Code review checklist integration
- Automated testing requirements

**Step 2: Documentation**
- Update all module README files
- Add examples following common rules
- Create troubleshooting guides

---

## 🚀 **NEXT ACTIONS - READY FOR IMPLEMENTATION**

### **Immediate Steps:**

**1. Add Common Rules Integration:**
```bash
# Execute in project root
git submodule add https://github.com/ntquy9901/ml-ds-common-rules.git docs/common-rules
git add .gitmodules docs/common-rules
git commit -m "Integrate ML/DS common rules"
```

**2. Create Main CLAUDE.md:**
```bash
# Create comprehensive project documentation
# Include common rules reference + project-specific rules
```

**3. Create project-context.md:**
```bash
# For AI assistant context during development
# Make rules and project info easily accessible
```

**4. Update Requirements Reference:**
```bash
# Update docs/requirements.md to reference integrated rules
# Make common rules easily accessible to team
```

---

## ✅ **FINAL RECOMMENDATION**

### **🎯 ADOPT INTEGRATION:**

**✅ YES - Integrate ML/DS Common Rules into project because:**

1. **Critical cho Success:** Proven practices reduce bugs, improve quality
2. **AI Assistant Benefits:** Makes rules accessible during code generation  
3. **Team Alignment:** Clear standards cho collaboration
4. **Maintainability:** Easier onboarding, consistent codebase

### **🎯 USE GIT SUBMODULE APPROACH:**

**Recommended Method:** Git Submodule integration
- **Pros:** Automatic updates, clean separation, version control
- **Cons:** Slight complexity, git knowledge required
- **Verdict:** **Best cho long-term project**

### **🎯 CREATE DOCUMENTATION STRUCTURE:**

**Essential Files:**
1. **CLAUDE.md** - Main project documentation
2. **project-context.md** - AI assistant context  
3. **docs/common-rules/** - Submodule with ML/DS rules
4. **Updated requirements.md** - Reference integrated rules

---

## 🎯 **EXPECTED BENEFITS**

**✅ Code Quality:**
- Consistent naming conventions
- Better documentation
- Improved test coverage
- Reduced technical debt

**✅ Team Efficiency:**
- Faster code reviews
- Clearer communication
- Easier onboarding
- Better collaboration

**✅ AI Assistant:**
- Context-aware code generation
- Consistent with project standards
- Automatic rule checking
- Better code suggestions

---

**Integration Strategy:** ✅ **READY FOR IMPLEMENTATION**  
**Next Step:** Begin integration with git submodule + CLAUDE.md creation

🚀 **Ready to proceed with common rules integration?**