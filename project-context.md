# Project Context - Stock Volatility Prediction VN30

**Project:** Multi-horizon volatility forecasting for VN30 stocks
**Focus:** 5-day ahead forecasts (Phase 1)
**Methodology:** HAR-R with Parkinson volatility, enhanced with LSTM, GNN, TimesFM
**Last Updated:** 2026-08-02 (Corrected stale Feature Categories/MODELS_TO_COMPARE claims per
`docs/report_2026-08-01/BAO_CAO_TONG_HOP.md`'s code audit — see paper-readiness audit report)

---

## 🎯 PROJECT OVERVIEW

### Objective
Build robust volatility prediction system for 30 VN30 stocks using daily OHLCV data (2006-2026), implementing HAR methodology adapted for daily frequency.

### Primary Target (Phase 1)
- **5-day ahead volatility forecast** ✅ CURRENT FOCUS
- **Secondary Targets:** 1, 10, 22-day forecasts (future expansion)
- **Input:** Daily OHLCV data for 30 stocks
- **Approach:** HAR-R baseline → Enhanced models (LSTM, GNN, TimesFM)

### Success Criteria
- **RMSE < 0.20** for 5-day forecasts
- **Directional accuracy > 55%**
- **QLIKE loss** (academic standard)
- **Test coverage > 85%**
- **ML/DS common rules compliance**

---

## 📋 KEY CONFIGURATIONS

### ✅ 22-Day Consistency (CONFIRMED)
```python
# HAR monthly feature = monthly forecast horizon
HAR_MONTHLY_FEATURE = 22  # trading days
MONTHLY_FORECAST_HORIZON = 22  # days ahead
```

### ✅ Standard Hyperparameters (ALL Models) - UPDATED 2026-06-19
```python
# Applied to ALL LSTM model training (6 files)
STANDARD_HYPERPARAMETERS = {
    'num_epochs': 70,      # Maximum training epochs (all models)
    'patience': 15,        # Early stopping patience (all models)
    'loss_function': 'MSE',  # Training loss (convex, stable)
    'optimizer': 'Adam',    # Default optimizer
}

# Applied to these files:
# - src/lstm_har_enhanced/train_with_validation.py
# - src/lstm_har_enhanced/train_enhanced.py
# - src/lstm_har_baseline/train_with_validation.py
# - src/lstm_har_baseline/train.py
# - src/lstm_baseline/train_with_validation.py
# - src/lstm_baseline/train.py
```

### Loss Function (Training) vs. Evaluation Metric Priority
Per CLAUDE.md §6 "Loss Functions": **training** uses MSE (convex, stable, differentiable
near zero volatility, where QLIKE is undefined/unstable); QLIKE is the **evaluation**-only
academic-standard metric for volatility. All training scripts (`nn.MSELoss()`) implement this
correctly — this section previously stated QLIKE as the training-loss priority, which
contradicted both CLAUDE.md and the actual code.
```python
TRAINING_LOSS = 'MSE'  # convex, stable, used in all train_epoch()/optimizer.step() calls

EVALUATION_METRIC_PRIORITY = {
    'primary': 'QLIKE',   # "Stylized favorite of volatility literature" — reported, not trained on
    'secondary': 'MSE',   # Comparison standard
    'tertiary': 'MAE'     # Robustness check
}
```

### ✅ Mandatory Metrics (6 Total) - UPDATED 2026-06-19
```python
# ALL models must report these 6 metrics in BOTH console and JSON
MANDATORY_METRICS = {
    1: 'MSE',           # Mean Squared Error (lower is better) ⭐ ADDED
    2: 'RMSE',          # Root Mean Squared Error (lower is better)
    3: 'MAE',           # Mean Absolute Error (lower is better)
    4: 'R²',            # Variance Explained (higher is better)
    5: 'QLIKE',         # Academic standard cho volatility (lower is better)
    6: 'Dir Acc'        # Directional Accuracy (higher is better)
}

# Output requirements:
# 1. Console: Print all 6 metrics for validation + test
# 2. JSON: Save all 6 metrics in validation_metrics, test_metrics, val_test_diff
# 3. Comparison table: Show all 6 metrics with differences
```

### Current Focus: 5-Day Horizon
```python
SINGLE_HORIZON_CONFIG = {
    'horizon': '5-day ahead',
    'target_column': 'target_5d',
    'features': ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol'],
    'loss': 'MSE'  # training loss; QLIKE is evaluation-only, see Loss Function section above
}
```

### Sentiment Analysis Integration (Research Phase - DEFERRED) 📊

**Status:** Research complete, implementation deferred until baseline complete

**Research Findings (2026-06-28):**
- ✅ **Date-based organization is industry standard** (FNSPID: 15.7M time-aligned news)
- ✅ **Per-ticker, per-date architecture** is proven approach
- ✅ **Integration pattern:** 3 HAR + 3 sentiment features = 6 total per stock
- ❌ **No Vietnamese historical dataset** available (GitHub: 1,005 articles, NO dates)
- ❌ **Cafef.vn blocked** (search returns 404, RSS 0% stock content)

**Decision Rationale:**
- LSTM-GNN baseline incomplete (68.02% Dir Acc < 70% target)
- Sentiment is enhancement, not core requirement
- Defer until baseline reaches 70%+ Dir Acc

**SOTA Fusion Architectures (2026-06-29 Research):** 🚀
- ✅ **Per-stock per-day news is CONFIRMED FEASIBLE** (FNSPID: 15.7M records)
- ✅ **MSGCA (2025) - TOP CHOICE:** Gated cross-attention, code available
- ✅ **Late Fusion PROVEN SUPERIOR:** 0.876 vs 0.828 accuracy (+5.8%)
- ✅ **LSTM-Transformer Hybrid:** Simple, matches current setup
- ✅ **Complete Pipeline Identified:** Data → Sentiment → Alignment → Fusion → Prediction

**Architecture Options:**
1. **MSGCA (Recommended):** Gated cross-attention, stable fusion, SOTA 2025
2. **LSTM-Transformer Hybrid:** Separate branches, late fusion, interpretable
3. **Temporal Fusion Transformer:** Multi-horizon (1d, 5d, 10d, 22d), variable selection

**Target Architecture (When Ready):**
```python
SENTIMENT_PIPELINE = {
    'data_source': 'vnstock OR Vietstock.vn crawler',
    'organization': 'per-ticker, per-date (FNSPID pattern)',
    'features': [
        'sentiment_score_3d',      # 3-day rolling mean
        'sentiment_confidence',    # Model confidence
        'news_count_norm'          # Normalized news count
    ],
    'sentiment_model': 'vinai/phobert-base (Vietnamese)',
    'integration': 'Late fusion (separate branches, fuse at decision layer)',
    'architecture': 'MSGCA or LSTM-Transformer hybrid',
    'timeline': '4 weeks (data → crawler → pipeline → fusion → training)'
}
```

**Data Structure (Planned):**
```
data/sentiment/
├── raw/by_ticker/{TICKER}/{DATE}/articles/*.json
├── raw/by_date/{DATE}/{TICKER}.json
└── processed/sentiment_features/{TICKER}_sentiment.csv
```

**Key Resources:**
- SOTA Fusion Research: `_bmad-output/planning-artifacts/research/technical-sentiment-volatility-fusion-sota-2026-06-29.md` (NEW!)
- Data Collection Research: `_bmad-output/planning-artifacts/research/technical-financial-news-crawling-dataset-research-2026-06-28.md`
- Memory SOTA: `memory/project_sentiment_volatility_fusion_sota.md` (NEW!)
- Memory Patterns: `memory/feedback_sentiment_volatility_fusion_architectures.md` (NEW!)
- Memory Data: `memory/feedback_financial_news_data_organization.md`
- Memory Links: `memory/reference_sentiment_analysis_resources.md`

**Next Actions (When Resuming):**
1. Test vnstock package for news functionality (2-3 days)
2. Manual inspection of Vietstock.vn (1 day)
3. Build proof-of-concept crawler (1 ticker, 1 week)

---

## 🏗️ TECHNICAL ARCHITECTURE

### Data Processing Pipeline
```
OHLCV (5 columns) → Parkinson Volatility → HAR Features (3) → 51+ Engineered Features
```

### Model Comparison Strategy
```python
MODELS_TO_COMPARE = [
    'HAR-R (Baseline)',           # Linear regression with HAR features
    'LSTM',                        # Deep learning temporal
    'LSTM + GNN',                  # Graph neural network enhancement
    'HAR-X + GNN',                 # Hybrid approach
    'TimesFM',                     # Foundation model
    'News-fusion (per-ticker gate)',  # Parallel LSTM-GNN + per-ticker-gated news branch,
                                       # baselines/2026-07-26_per_ticker_news_gate_baseline —
                                       # current primary result lineage, see BAO_CAO_TONG_HOP.md
]
```

### LSTM-GAT Hybrid Architecture (Advanced) 🚀
```python
LSTM_GAT_HYBRID = {
    'temporal_branch': 'LSTM encoder (per-stock temporal learning)',
    'spatial_branch': 'Graph Attention Network (cross-stock relationships)',
    'graph_construction': 'Dynamic correlation + volatility spillover',
    'attention_mechanism': 'Multi-head attention (4-8 heads)',
    'fusion_strategy': 'Concatenate + MLP for final prediction',
    'expected_improvement': 'RMSE 17% ↓, Dir Acc 7% ↑',
    'architecture_doc': 'docs/project/LSTM_GAT_ARCHITECTURE.md',
    'based_on': [
        'TemporalGAT (arXiv 2410.16858v1, 2024)',
        'FSTGAT (MDPI Symmetry, 2024)',
        'STGAT (MDPI Applied Sciences, 2025)'
    ]
}

# Input: (batch, seq_len, 30_stocks, 22_features)
# Output: (batch, 30_stocks, 1_prediction)
```

**Key Innovation:**
- Processes all 30 VN30 stocks simultaneously
- Dynamic graph captures market-wide dependencies
- Attention weights reveal influential stocks

### Single Model Architecture
- **One model for all 30 stocks** (not individual stock models)
- **Stock identifier as feature** or **panel data approach**
- **Unified training** across all VN30 stocks

---

## 📁 CRITICAL FILE LOCATIONS

### Documentation
- **Main documentation:** `CLAUDE.md` - Project overview, common rules, technical architecture
- **LSTM-GAT Architecture:** `docs/project/LSTM_GAT_ARCHITECTURE.md` - Advanced hybrid model design 🚀 NEW
- **Requirements:** `docs/requirements.md` - Functional and non-functional requirements
- **Technical config:** `docs/technical_config.md` - HAR configuration, feature engineering
- **Data schema:** `docs/data_schema.md` - 51+ features specification
- **Loss functions:** `docs/loss_functions.md` - QLIKE and evaluation metrics
- **Strategy:** `docs/single_horizon_strategy.md` - 5-day focus approach
- **Integration:** `docs/common_rules_integration.md` - ML/DS rules integration

### ML/DS Common Rules
- **Submodule location:** `docs/common-rules/`
- **Main rules:** `docs/common-rules/COMMON_RULES.md`
- **Quick reference:** `docs/common-rules/QUICK_REFERENCE.md`
- **Template:** `docs/common-rules/CLAUDE_TEMPLATE.md`

### Data
- **Raw data:** `data/raw/prices/` - 30 stocks OHLCV files
- **Collection summary:** `data/raw/prices/collection_summary.csv`
- **Processed data:** `data/processed/` (to be created)

---

## 🔧 CRITICAL IMPLEMENTATION RULES

### Financial ML Context
```python
# Time series integrity - chronological split only
train_split = int(len(data) * 0.8)
assert train_index < test_index  # Validate temporal order

# Volatility calculation
parkinson_vol = (np.log(high / low) ** 2) / (4 * np.log(2))

# OHLCV consistency validation
assert all(ohlc['high'] >= ohlc['close'])
assert all(ohlc['close'] >= ohlc['low'])
```

### Naming Conventions (ML/DS Common Rules)
```python
# ✅ CORRECT - Descriptive and clear
volatility_forecast = model.predict(data)
parkinson_volatility = calculate_parkinson_volatility(data)
train_accuracy = 0.95

# ❌ AVOID - Too generic or abbreviated
vol = model.predict(data)
pred = forecast(data)
acc = 0.95
```

### Code Quality Standards
- **Function size:** < 30 lines per function
- **Parameters:** < 3 preferred
- **Single responsibility:** One concern per function
- **Documentation:** Docstrings for public functions
- **Comments:** Explain WHY not HOW

### Testing Requirements
```python
COVERAGE_TARGETS = {
    'overall': 0.85,
    'data_processing': 0.90,
    'model_training': 0.80,
    'integration': 0.30,
    'e2e': 0.10
}
```

### Research Best Practices
- **Fixed seeds:** random.seed(42), np.random.seed(42), torch.manual_seed(42)
- **Experiment tracking:** MLflow for all experiments
- **Reproducibility:** Same seed → same results
- **Learning curves:** Plot training progress (mandatory)
- **Checkpoint saving:** Save intermediate results

---

## 📊 FEATURE ENGINEERING

### HAR Features (Confirmed 22-day)
```python
def create_har_features(volatility_series):
    """Create HAR features with confirmed 22-day monthly window."""
    return pd.DataFrame({
        'har_daily_vol': volatility_series.rolling(1).mean(),
        'har_weekly_vol': volatility_series.rolling(5).mean(),
        'har_monthly_vol': volatility_series.rolling(22).mean()  # ✅ CONFIRMED
    })
```

### Target Variables
```python
def create_forecast_targets(volatility_series):
    """Create multi-horizon targets with 22-day monthly horizon."""
    targets = pd.DataFrame()
    targets['target_1d'] = volatility_series.shift(-1)
    targets['target_5d'] = volatility_series.shift(-5)   # Phase 1 focus
    targets['target_10d'] = volatility_series.shift(-10)
    targets['target_22d'] = volatility_series.shift(-22)  # ✅ CONFIRMED
    return targets
```

### Feature Categories

**As actually implemented (verified by full `src/` code audit,
`docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` §3):**
- **HAR features (3):** Daily, weekly (5d), monthly (22d) volatility — the only features the
  principal model families (HAR-R, LSTM-HAR, LSTM-GAT hybrid, news-fusion) actually consume.
- News-fusion lineage additionally uses per-ticker news embeddings (see
  `baselines/2026-07-25_dual_group_news_embedding_baseline` onward).

**Planned in early Phase-1 design, never implemented — do NOT cite as existing:**
- Lagged returns (4), volume indicators (2), technical indicators (RSI/MACD/Bollinger),
  calendar/temporal features (day-of-week, month, quarter, VN holiday flags). A one-off
  calendar-feature experiment was tried 2026-08-01
  (`baselines/2026-08-01_calendar_news_gate_baseline`) and found null — not adopted.
- The original "51+ engineered features from 5 raw OHLCV columns" total below reflected this
  planned-but-unbuilt set; the shipped models use only the 3 HAR features (+ news, where used).

---

## 🎯 IMPLEMENTATION STRATEGY

### Phase 1: 5-Day Baseline (Week 1-4)
```python
PHASE_1_GOALS = {
    'Week 1-2': 'HAR-R baseline for 5-day forecasts',
    'Week 3-4': 'Enhanced models (LSTM, extended features)',
    'Success': 'RMSE < 0.20, Directional Accuracy > 55%'
}
```

### Phase 2: Multi-Horizon Expansion (Week 5+, Conditional)
```python
PHASE_2_CONDITION = {
    'if': '5-day model meets success criteria',
    'then': 'Expand to 1, 5, 10, 22-day forecasts',
    'else': 'Continue 5-day refinement'
}
```

### Phase 3: Advanced LSTM-GAT Hybrid (Week 9-12) 🚀
```python
PHASE_3_GOALS = {
    'Week 9': 'Data preparation (technical indicators, graph utilities)',
    'Week 10': 'Model development (LSTM encoder, GAT layers, fusion)',
    'Week 11': 'Training & evaluation (hyperparameter tuning, comparison)',
    'Week 12': 'Analysis & deployment (attention visualization, ablation)',
    'Target': 'RMSE < 0.15, Dir Acc > 75% (vs 0.18, 67.90% current)',
    'Architecture': 'LSTM (temporal) + Graph Attention Network (spatial)',
    'Documentation': 'docs/project/LSTM_GAT_ARCHITECTURE.md'
}

# Key improvements expected
IMPROVEMENT_TARGETS = {
    'RMSE': '0.18 → < 0.15 (17% ↓)',
    'Dir_Acc': '67.90% → > 75% (7% ↑)',
    'QLIKE': '~0.12 → < 0.10 (17% ↓)',
    'R²': '~0.65 → > 0.75 (15% ↑)'
}
```

### Quality Gates
- **Pre-commit:** Tests pass, coverage sufficient
- **Pre-merge:** Code review approval, no critical issues
- **Pre-deploy:** Performance benchmarks met, documentation complete

---

## 🚀 NEXT ACTIONS

### Immediate (Sprint 1)
1. **Data processing pipeline**
   - Implement Parkinson volatility calculation
   - Create HAR features (22-day confirmed)
   - Generate 5-day targets only
   - Validate OHLCV consistency

2. **HAR-R baseline**
   - Train linear regression with HAR features
   - Evaluate with QLIKE, RMSE, directional accuracy
   - Compare vs random walk baseline

3. **Testing infrastructure**
   - Set up pytest with 85%+ coverage target
   - Create data quality tests
   - Implement reproducibility tests (fixed seed)

### Documentation
- **All functions:** Public functions must have docstrings
- **Experiment tracking:** MLflow for all runs
- **Code reviews:** Use ML/DS common rules checklist
- **Onboarding:** New members read common rules (Day 1)

---

## 📖 KEY REFERENCES

### Academic Sources
- **HAR methodology:** Corsi (2009) - HAR-R for daily data
- **QLIKE loss:** "Stylized favorite of volatility forecasting literature"
- **22-day standard:** ~22 trading days per month (industry convention)
- **FNSPID dataset:** Zdong et al. (2024) - 15.7M time-aligned news records

### Project Documentation
- **Project requirements:** `docs/requirements.md`
- **Technical configuration:** `docs/technical_config.md`
- **ML/DS common rules:** `docs/common-rules/COMMON_RULES.md`
- **LSTM-GAT Architecture:** `docs/project/LSTM_GAT_ARCHITECTURE.md`

### External Resources
- **ML/DS common rules repo:** `D:\bmad-projects\ml-ds-common-rules`
- **HAR-X paper:** `docs/paper/1-s2.0-S1544612323003641-main.pdf`
- **FNSPID GitHub:** [FNSPID_Financial_News_Dataset](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)

### Sentiment Analysis Resources
- **Research:** `_bmad-output/planning-artifacts/research/technical-financial-news-crawling-dataset-research-2026-06-28.md`
- **Memory:** `memory/` (project_sentiment_research_status.md, feedback_financial_news_data_organization.md, reference_sentiment_analysis_resources.md)

---

**Project Status:** ✅ Documentation complete, ready for Sprint 1 implementation
**Current Focus:** 5-day HAR-R baseline with QLIKE loss function
**Quality Standard:** 85%+ test coverage, ML/DS common rules compliance

---

## 📝 UPDATE HISTORY

### 2026-08-05 - global-benchmark branch (S&P500): applied VN30 audit findings, merged master twice

**Note:** continuation of the 2026-08-02 entry below, same branch/clone
(`C:\luanvan\stock_vol_prediction01_branchGlobal`, own independent `.git`, not a git worktree of
the VN30 clone despite CLAUDE.md's "git worktree" terminology — confirmed both have separate
`.git` directories).

**Applied VN30's 2026-08-02 audit reports (`docs/reports/2026-08-02_1056_paper_readiness_audit_report.md`,
`_152253_summaryOfUpdate_report.md`, `_152758_summaryOfUpdate_report.md`) to this branch, scoped to
what actually applies:** verified directly (not by inference) that AUD-001 (normalizer fit on 100%
of data before split) and AUD-002 (per-ticker outlier removal desyncs cross-stock row alignment)
are present in `src/lstm_gat_hybrid/dataset.py` — the exact file this branch's S&P500 baseline
imports read-only — while AUD-003/004/005/006/010/011/012/017/018/020/021 do not apply (verified
each, not assumed). Wrote 2 `xfail(strict=True)` regression tests
(`baselines/2026-08-01_lstm_gnn_sp500_baseline/test/test_dataset_leakage_and_alignment.py`) proving
both bugs against real code paths — left unfixed on purpose (dataset.py is shared with VN30 master,
which was fixing it independently) so the tests flip to XPASS (loudly, via `strict=True`) once that
fix is merged in, rather than silently diverging with an independent fix here.

**Adopted the full 11-gate verification spec** from `docs/reports/2026-08-02_152758_summaryOfUpdate_report.md`
into `CLAUDE.md` ("Verification Gates (Evidence-Based)" section) at the user's explicit request for
the complete version (not a scoped-down subset) — evidence directory schema, manifest.json,
acceptance_traceability.csv, all 11 gates, with the "skills to compose" list adapted to this
project's actual tooling (no `bmad-*` skills here). Also added a new CLAUDE.md §7 rule mandating
regular `git fetch`/merge from master, motivated directly by this session's dataset.py-bug
situation.

**Installed 2 Superpowers skills standalone** (`systematic-debugging`, `test-driven-development`,
from `github.com/obra/superpowers`) at project-level `.claude/skills/` — deliberately not the full
plugin. Also un-gitignored `.claude/` entirely (was previously fully ignored) per explicit user
decision, confirmed twice, so project skills + `settings.local.json` are now tracked in git.

**Merged `origin/master` into `global-benchmark` twice** (fixed a single-branch-clone
`remote.origin.fetch` restriction first — this clone could not see `origin/master` at all until
the refspec was widened to `+refs/heads/*:refs/remotes/origin/*`). First merge: 1 real conflict in
`src/common/evaluation.py` (kept both branches' directional-accuracy fixes — this branch's
`evaluate_predictions_grouped`/`directional_accuracy_grouped`, used directly by
`train_sp500_lstm_gnn.py`, alongside master's `directional_accuracy_per_ticker`/`n_stocks=`
parameter, a different fix for the same underlying flatten bug on master's data layout); surfaced
and fixed a real regression afterward (master's new AUD-018 empty-partition guard in
`temporal_split_dataframe()` rejected ANY zero-size partition unconditionally, breaking this
branch's intentional `test_ratio=0.0` support in `src/common/multi_ticker_dataset.py` — narrowed
the guard to only reject *unintentionally* empty partitions). Second merge (13 more master commits):
0 conflicts. Confirmed after both merges: `src/lstm_gat_hybrid/dataset.py` itself still untouched
by master — the 2 xfail tests remain correctly red, not a false XPASS.

**Full test suite re-verified clean after every step:** 52 passed, 2 xfailed (expected) as of the
final push. Full detail in auto-memory (Claude Code project memory directory, not this repo's own
`memory/` convention which doesn't exist on this branch) —
`project_sp500_lstm_gnn_baseline_status.md`, `project_vn30_lstm_gnn_missing_seed.md`,
`project_vn30_diracc_flattening_and_master_sync.md`.

### 2026-08-02 - global-benchmark branch (S&P500): seed + checkpoint-selection bugs found and fixed

**Note:** this entry documents work on the `global-benchmark` branch/worktree
(`C:\luanvan\stock_vol_prediction01_branchGlobal`), a separate git worktree from the VN30 work
described elsewhere in this file (per CLAUDE.md §7). Baseline: `baselines/2026-08-01_lstm_gnn_sp500_baseline/`
(30 S&P500 tickers, reuses `src/lstm_gat_hybrid/{model_parallel,dataset,config}.py` read-only).

**Bug 1 — missing random seed:** 4 smoke runs (2026-08-01, 2 epochs, unseeded) on identical config
produced test R² ranging from 0.9999 to -777 and QLIKE up to 19620. Root cause: no
`torch.manual_seed`/`np.random.seed`/`random.seed` call anywhere in the training script — at 2
epochs, unseeded weight initialization alone accounts for the variance. Fixed by adding
`set_seed(42)` + `--seed` CLI argument. The same gap was confirmed present in the VN30 master
project's `src/lstm_gat_hybrid` trainers (`train.py`, `train_parallel.py`,
`train_parallel_enhanced.py`, `train_simplified.py`) and in
`baselines/2026-07-26_per_ticker_news_gate_baseline/code/train_per_ticker_gate.py` — none of them
set a seed, meaning the 2026-07-26 per-ticker-gate result (QLIKE 0.5473) has not been confirmed
reproducible. Not yet fixed on the master project as of this entry.

**Bug 2 — checkpoint selection:** selecting the best checkpoint by minimum validation loss (MSE)
picked a degenerate epoch where validation directional accuracy had collapsed toward 0% while
validation loss kept decreasing. Fixed by selecting the checkpoint with maximum validation
directional accuracy instead (`is_new_best()`). The VN30 master project's trainers still select by
minimum validation loss.

**Post-fix results (seed=42, 5-10 epoch exploratory runs, all below the 70-epoch project
standard):** test directional accuracy near random across horizons — 1-day: 53.07%, 5-day:
49.26-49.54%, 22-day: 49.35%, 10-day: run incomplete at time of writing. Test R² above 0.9999 in
all cases; this reflects agreement on volatility magnitude only, not directional skill, given HAR
target smoothing.

**Not yet complete:** `code_review/` for this baseline has no adversarial review recorded; per
CLAUDE.md §3.F this baseline is not "done." Full detail:
`memory/project_sp500_lstm_gnn_baseline_status.md` and `memory/project_vn30_lstm_gnn_missing_seed.md`
(auto-memory, Claude Code project memory directory).

### 2026-07-26 - Dual-group rebuild + SOTA pivot (spillover/QLIKE) + per-ticker isolated gate BREAKTHROUGH + VN30 universe audit

**Context:** continuing the 2026-07-07→07-25 news-fusion baseline lineage (see entries below and
`docs/reports/2026-07-25_*` / `2026-07-26_*`). Session covered: (1) GPU cache expansion cleanup +
embedding-cache documentation, (2) dual-group panel rebuild with 12 newly-classified sources,
(3) SOTA literature research → 2 new baselines, (4) a genuine breakthrough after ~10 null
results, (5) an audit that found the project's stock universe is stale vs. current VN30.

**1. Docs + cleanup:** `docs/EMBEDDING_STORAGE_SPECIFICATION.md` +
`docs/EMBEDDING_USAGE_IMPLEMENTATION_GUIDE.md` (cache structure, cross-project reuse, train/val/
test split handling for the PhoBERT embedding cache). Deleted 2 old cache backups (~8.8GB).

**2. Dual-group panel rebuilt (12 new sources)** — **regression, not improvement**: Test DirAcc
68.25% vs. 68.71% pre-rebuild (-0.46pp), QLIKE/R²/RMSE all slightly worse. More news volume from
mostly-general-press sources diluted rather than helped.

**3. SOTA research → 2 untouched knobs identified:** every prior baseline kept the inter-stock
graph symmetric/same-day (`graph_correlation.py`) and the loss plain MSE (QLIKE only used for
eval). 2025-2026 literature (Zhang/Pu/Cucuringu/Dong IJF 2025; Chi et al. 2026) flags both as
gaps. Built `baselines/2026-07-26_spillover_qlike_baseline` (directed lead-lag graph + MSE+QLIKE
loss) — **also null** (Test DirAcc 68.23% vs. 68.25%, no real lift).

**4. BREAKTHROUGH — `baselines/2026-07-26_per_ticker_news_gate_baseline`:** per-ticker learnable
gate (`gate_logits`, one free scalar per stock, NOT shared weights like `gated_crossattn`'s
`gate_mlp`) — gradient PROVEN isolated per ticker (direct perturbation test, not just reasoning).
Trained to epoch 40 (4× resumed 10-epoch runs, user-approved after each). **Result: first clear
win after ~10 consecutive null results** — new project-best QLIKE (0.5473 @ epoch 20), R² ties/
edges the previous record. BUT: (a) aggregate metrics peaked ~epoch 20, degraded slightly by
epoch 40 (mild overfitting); (b) per-ticker gate values took ~30 epochs to stabilize (r=0.98
epoch30-vs-40) — single-epoch snapshots are NOT reliable; (c) even fully-converged gate values
still don't strongly match the independent ablation's per-ticker usefulness signal (r=0.35,
p=0.053 — best of 5 methods tried, still not conclusive). Full detail:
`memory/project_null_result_pattern_and_sota_pivot.md` (auto-memory).

**5. VN30 ticker universe audit (verified against official HOSE PDF, Kỳ 1/2026):** this
project's 32-ticker price universe (`data/processed/`) is STALE vs. the current official VN30
(30 tickers) by **5 extra** (BCM, BVH, NVL, PDR, POW — no longer in VN30) **+ 3 missing** (DGC,
LPB, VPL — added in rebalances since this project's data was collected). Separately, and
independently, the news ticker-mention regex (`vendor_config.py::VN30_TICKERS`) is ALSO missing
VPB/VRE vs. the project's own 32-ticker universe — meaning `dual_group_news_panel.parquet` has
**zero rows** for VPB/VRE; any per-ticker "news usefulness" reading for those 2 tickers in ANY
baseline is a zero-input artifact, not a real signal. Neither issue fixed yet (both flagged,
pending decision — fixing #1 requires collecting new price history + full retrain; #2 is a small
regex+rebuild fix). Full detail: `memory/project_vn30_ticker_universe_mismatch.md` (auto-memory).

**Reports:** `docs/reports/2026-07-26_2000_summaryOfUpdate_report.md`,
`2026-07-26_2230_summaryOfUpdate_report.md`, `2026-07-26_2245_summaryOfUpdate_report.md`,
`2026-07-26_2330_summaryOfUpdate_report.md` (this consolidated one).

### 2026-07-08 - Embedding baseline: learning curves + 6-ep verify
- Train script thêm learning curve mỗi 5 epoch (`--plot_every`, reuse `plot_learning_curves_with_analysis` từ `train_parallel_enhanced.py`) → đóng gap CLAUDE.md **§3.C**. Verify: 2 PNG/run (epoch 5 + final) trong `results/embedding_baseline_*/`.
- **Pending:** full 20-epoch go/no-go thật (val DirAcc embedding vs scalar sentiment matched-epoch).

### 2026-07-07 - Embedding baseline BUILT + adversarial review + real PhoBERT extraction
**Baseline (NEW convention `baselines/YYYY-MM-DD_<name>/`):** `baselines/2026-07-07_embedding_baseline/` với 5 sub-folder `requirements/design/code/code_review/test` (rule mới CLAUDE.md **§3.F**). Code: `extract_embeddings.py` (PhoBERT frozen → PCA 768→64), `dataset_embedding.py` (5-tuple `(x_har, adj, x_emb, mask, y)`), `model_embedding.py` (`ArticleSetAttentionPooling` + HAR reuse `ParallelLSTMGNN.get_embeddings` + concat), `train_embedding_baseline.py`.
**Code review (skill `/code-review`, adversarial):** 10 findings (3 HIGH, 7 MEDIUM) → **ALL FIXED + re-verified** (pytest 6 pass). File `code_review/code_review_2026-07-07.md`.
**Real extraction DONE:** PhoBERT → 3,442 ticker-matched articles → PCA 64-d (fit 625 train articles, explained var 82.9%, no leakage) → 30 caches `data/sentiment_embedding/{TICKER}_emb.npz`.
**5-epoch real training:** DirAcc val 70.29% / test 68.44% (undertrained — chưa go/no-go).
**Rule additions (CLAUDE.md §3.F):** rule 5 (pytest mandatory, installed 9.1.1), rule 7 (test issues phải fix hết).
**Env:** `npx skills add K-Dense-AI/scientific-agent-skills` → ~200 scientific skills (aeon, scikit-learn, statistical-analysis... relevant time-series).

### 2026-07-06 - News aggregation (9 sources) + sparsity analysis + 2 design docs (thầy góp ý)
**Aggregation (NEW `src/data_aggregation/`):** gộp 9 nguồn crawled → `crawl_data/aggregated/unified_articles.csv` (58,755 raw → **21,107 unique** sau dedup). Script `aggregate_news_sources.py` (cô lập). 2 họ schema unified, 2 format date (ISO + DD/MM/YYYY).
**Sparsity analysis (`analyze_news_sparsity.py`):** article-level đều (gap lấp; 2021-2026 ~10.5K bài) NHƯNG per-stock-day **vẫn cực thưa** — test chỉ **5.5%** stock-day có tin (match-rate ~20% bài match mã VN30). → "test mù tin" giải quyết ở article, KHÔNG ở per-stock-day. Bottleneck là match-rate, không phải số bài.
**2 design docs:** `docs/project/SENTIMENT_LATENT_SPACE_TECHNIQUES.md` (VIB/VAE/noise injection — kỹ thuật "random latent vector" thầy gợi ý), `SENTIMENT_NEWS_EMBEDDING_ARCHITECTURE.md` (scalar → embedding, SOTA FNSPID). + timestamped copy `SENTIMENT_ANALYSIS_DESIGN_2026-07-06.md` (mục 2.3 cập nhật sparsity mới).

### 2026-07-04 - Sentiment Baseline Implementation (ISOLATED, lexicon done; phobert pending)

**Status:** Sentiment integration moved from research → working baseline experiment. Isolated package `src/sentiment_baseline/` (does NOT modify `src/lstm_gat_hybrid/` or `data/processed/`). Design doc: `docs/project/SENTIMENT_ANALYSIS_DESIGN.md`.

**Built:**
- `lexicon.py` (default scorer), `phobert_scorer.py` (HuggingFace XLM-RoBERTa, lazy-load).
- `process_news_to_sentiment.py` → `data/sentiment_baseline/{TICKER}_sentiment.csv` (cols: date, sentiment_1d, news_count_1d, news_titles for review).
- `dataset_sentiment.py` (subclass + dataloader copy, 5 features = 3 HAR + 2 sentiment).
- `train_sentiment_baseline.py` (reuses existing train_epoch/validate; supports --epochs/--resume_from/--sentiment_dir).

**Design decisions (simplicity-first):** daily sentiment (not rolling — LSTM aggregates 22-day window itself); dropped rolling, news_coverage_flag, has_news (redundant), T+1 alignment (no timestamp; 5-day horizon doesn't need it; windowing prevents lookahead).

**Results (lexicon, k-NN):** DirAcc 68.17% (10ep) → 68.57% (20ep); R² 0.714 (= HAR baseline). NOT clearly better than HAR-only 69.98%@70ep — unfair (need HAR-only at same epochs as control). Lexicon quality: 57% news-days non-zero, misses 43% (generic titles) + negation errors. More news data did NOT help with lexicon → bottleneck is scorer quality.

**⚠️ Gotcha:** transformers 5.x breaks XLM-R tokenizer (tiktoken conversion on sentencepiece). MUST pin `transformers<5` (4.57.6) + install `sentencepiece tiktoken`.

**Data status:** 12,212 unique news articles; test set (2021-2026) = 6,822 articles, all years covered → "test blind" problem SOLVED.

**Pending:** phobert processing+train run (interrupted); HAR-only control at 10/15/20 ep for fair comparison.

### 2026-06-28 - Sentiment Analysis Research (DEFERRED)
**Changes:**
- ✅ **Comprehensive research completed:** 25 academic papers + technical resources analyzed
- ✅ **Confirmed date-based organization as industry standard** (FNSPID gold standard)
- ✅ **Defined target architecture:** Per-ticker, per-date structure with 3 sentiment features
- ✅ **Created memory system:** 3 memory files for patterns, resources, status tracking
- ⏸️ **Implementation deferred:** Baseline LSTM-GNN incomplete (68.02% < 70% target)

**Research Findings:**
- **FNSPID (2024):** 15.7M time-aligned news + 29.7M prices (4,775 stocks, 1999-2023)
- **Organization:** Per-ticker, per-date is industry standard
- **Integration:** 3 HAR + 3 sentiment features = 6 total per stock
- **Vietnam market:** No public historical dataset, Cafef.vn blocked

**Decision Rationale:**
- Sentiment is enhancement, not core requirement
- Priority: Complete baseline first (70%+ Dir Acc target)
- Timeline: 4 weeks for full sentiment pipeline (deferred)

**Files Created:**
- `vietnam-stock-news-data-sources-plan.md` - 5 data sources comparison
- `_bmad-output/planning-artifacts/research/technical-financial-news-crawling-dataset-research-2026-06-28.md` - 25 sources, implementation plan
- `memory/project_sentiment_research_status.md` - Current status, blockers, action plan
- `memory/feedback_financial_news_data_organization.md` - Mandatory patterns (date-based, FNSPID pattern)
- `memory/reference_sentiment_analysis_resources.md` - 29 categorized resource links
- `MEMORY.md` - Memory index for future reference

**Project Context Updated:**
- Added "Sentiment Analysis Integration" section with research findings
- Updated "Key References" with FNSPID and sentiment resources
- Documented defer rationale and next actions

**Next Actions (When Resuming):**
1. Test vnstock package (2-3 days)
2. Manual Vietstock.vn inspection (1 day)
3. Build proof-of-concept crawler (1 week)

### 2026-06-29 - SOTA Sentiment-Volatility Fusion Research (DEFERRED)
**Changes:**
- ✅ **SOTA architectures identified:** MSGCA (2025), LSTM-Transformer Hybrid, TFT
- ✅ **Per-stock per-day CONFIRMED FEASIBLE:** FNSPID proves 15.7M records possible
- ✅ **Late fusion PROVEN SUPERIOR:** 0.876 vs 0.828 accuracy (+5.8%), +11.7% recall
- ✅ **Complete pipeline defined:** Data → Sentiment → Alignment → Fusion → Prediction
- ✅ **Implementation details provided:** Code examples for all architectures
- ⏸️ **Implementation deferred:** Baseline LSTM-GNN incomplete (68.02% < 70% target)

**SOTA Architectures (2025-2026):**
- **MSGCA (2025) - TOP CHOICE:** Gated cross-attention, code available, stable fusion
- **LSTM-Transformer Hybrid:** Separate branches, late fusion, interpretable
- **Temporal Fusion Transformer:** Multi-horizon (1d, 5d, 10d, 22d), variable selection
- **Gated Multi-Task Fusion:** FinBERT + CNN-BiLSTM, dynamic weighting

**Key Findings:**
- **Per-Stock Per-Day:** YES, standard practice (FNSPID, Kaggle datasets)
- **Late vs Early Fusion:** Late fusion superior (5.8% better accuracy, 11.7% better recall)
- **Expected Improvement:** +2-5% Dir Acc, -5-10% RMSE when integrated
- **Vietnamese Models:** PhoBERT (vinai/phobert-base) for sentiment extraction

**Files Created:**
- `_bmad-output/planning-artifacts/research/technical-sentiment-volatility-fusion-sota-2026-06-29.md` - 16 sources, complete guide
- `memory/project_sentiment_volatility_fusion_sota.md` - SOTA status, architecture selection
- `memory/feedback_sentiment_volatility_fusion_architectures.md` - SOTA patterns, implementation
- `MEMORY.md` - Updated with new memory files

**Project Context Updated:**
- Added SOTA architectures section with MSGCA, LSTM-Transformer, TFT
- Updated integration: Late fusion (not early fusion)
- Added SOTA research to key resources
- Updated next actions with architecture selection

**Implementation Readiness:**
- All SOTA architectures documented with code examples
- Complete data pipeline pattern provided
- Expected performance improvements: 70-73% Dir Acc (vs 68.02% baseline)
- Timeline: 4 weeks when ready (data → crawler → pipeline → fusion → training)

**Next Actions (When Resuming):**
1. Complete LSTM-GNN baseline (70%+ Dir Acc target)
2. Choose architecture (MSGCA recommended)
3. Implement per-stock per-date crawler
4. Train with 6 features (3 HAR + 3 sentiment)

### 2026-06-19 - Standardization Update
**Changes:**
- ✅ **Standardized hyperparameters:** 70 epochs, 15 patience for ALL 6 LSTM models
- ✅ **Added MSE as 6th mandatory metric** (was 5 metrics: RMSE, MAE, R², QLIKE, Dir Acc)
- ✅ **Mandatory output format:** All 6 metrics must appear in console + JSON
- ✅ **Updated all training files:** lstm_har_enhanced, lstm_har_baseline, lstm_baseline
- ✅ **Enhanced comparison tables:** Added MSE to all val/test comparisons

**Impact:**
- All models now use consistent hyperparameters for fair comparison
- Complete metrics reporting (all 6: MSE, RMSE, MAE, R², QLIKE, Dir Acc)
- Better reproducibility and model comparison

**Files Updated:**
- `CLAUDE.md` - Added standard hyperparameters section, updated metrics section
- `src/common/evaluation.py` - Added MSE to evaluate_predictions()
- All 6 training files - Updated epochs, patience, and MSE output

### 2026-06-15 - Initial Documentation
**Created:**
- Project context document
- Technical architecture
- Implementation strategy
- Quality standards
