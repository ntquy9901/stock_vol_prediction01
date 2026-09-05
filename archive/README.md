# Archive Directory

> **⛔ OUT OF SCOPE FOR ALL AUDITS AND REVIEWS.** Everything under `archive/` (any depth) is
> intentionally retired — not in active use, not part of the paper, not a target for fixes.
> Code review, adversarial audits, lint, coverage, and any AI-driven "find issues"/"audit the
> repo" pass MUST skip this entire tree. Do not report findings, propose fixes, or flag bugs for
> anything under `archive/` — if you're an AI agent auditing this repo and you weren't explicitly
> asked to review `archive/` itself, treat it as if it doesn't exist. See CLAUDE.md's Definition
> of Done → "Audit/Review scope" for the project-wide rule this enforces.

This directory contains deprecated and archived code from the stock volatility prediction project.

> **Restoration log:** every batch move in this directory is also recorded, with old→new paths
> and the reason for each, in a dedicated log file for easy reversal — see
> `2026-08-02_2200_archive_batch_log.md` for the largest batch (12 baselines + `src/sentiment_baseline/`
> + 3 historical report folders + 3 cascading-orphan `data/` folders, all cross-referenced against
> 5 project reports). Git history (`git log --follow`) also works for any single file.

---

## 📁 Archive Structure

```
archive/
├── data_leakage_scripts/     ← Scripts with CRITICAL data leakage bugs
│   ├── README.md             ← Detailed documentation
│   ├── train_all_baselines.py
│   ├── train_enhanced_lstm_har_vn30.py
│   └── train_enhanced_lstm_har_vn30_progress.py
│
├── baselines/                ← Full timestamped baselines no longer in scope for the paper
│   └── 2026-07-18_resttext_baseline/  ← Best raw QLIKE (0.5431) of all 23 baselines, but not
│                                         seed-verified and dropped from paper scope 2026-08-02
│                                         (user decision) in favor of the per-ticker-gate
│                                         news-fusion lineage. Not a bug archive — moved here
│                                         only to shrink `baselines/`'s active scope.
│
├── data_raw/                 ← Raw price folders confirmed unused by any live script
│   │                            (verified via repo-wide grep before moving, 2026-08-02)
│   ├── hnx/, hnx_enhanced/   ← HNX exchange data, unrelated to the VN30 project scope.
│   │                            Was gitignored in its old location (data/raw/hnx/) — .gitignore
│   │                            updated to the new path, so these stay untracked here too.
│   ├── vn30_enhanced/        ← Duplicate ticker list of data/raw/vn30/ — now itself merged away
│   │                            (see `vn30/` entry below), referenced by nothing except a one-off
│   │                            comparison script (src/experiment/compare_crawl_results.py).
│   ├── vn30/                 ← **[2026-08-02]** data/raw/vn30/ merged into data/raw/prices/ (the
│   │                            folder process_parkinson_pipeline.py actually reads by default):
│   │                            VPB/VRE (the only 2 tickers actually used from here) `git mv`'d
│   │                            directly into `data/raw/prices/`; the other 26 tickers were exact
│   │                            duplicates of files already in `prices/` and read by nothing;
│   │                            BSR/VPL (crawled 2026-08-02, excluded from the training universe —
│   │                            see the VN30 data source audit report) came along too. `prices/`
│   │                            is now the single canonical raw source for all 32 live tickers.
│   ├── vn30_sentiment/       ← **[2026-08-02]** Not the sentiment data actually in use — the live
│   │                            sentiment pipeline (src/sentiment/processing/sentiment_integration.py)
│   │                            reads `data/processed/vn30_sentiment/` (still live, NOT archived —
│   │                            different path, do not confuse the two). This raw/ counterpart had
│   │                            12 files under `news/` read only by 2 scripts
│   │                            (`data_scripts/generate_realistic_news.py`,
│   │                            `data_scripts/process_realistic_news_with_finbert.py`, both moved
│   │                            alongside it) — both scripts also hardcoded a stale
│   │                            `D:/bmad-projects/...` path (same bug class as commit `e434b1a`,
│   │                            never fixed since not worth fixing unused code). 3 subfolders
│   │                            (`analyst_reports/`, `press_releases/`, `social_media/`) were
│   │                            already empty and removed, not archived.
│   ├── vn100/, vn100_enhanced/ ← VN100 track dropped from project scope entirely, 2026-08-02
│   │                            (user decision — "không xử lý vn100"). Was gitignored in its old
│   │                            location; .gitignore updated to the new path.
│   └── test/, test_combined/ ← Only referenced by an already-archived script
│                                 (archive/data_scripts/quick_test_crawl.py).
│
├── data_processed/
│   └── vn100_only/           ← Processed output of the now-archived VN100 raw data, 2026-08-02.
│
├── data/                     ← **[2026-08-02]** Root-level data/ subfolders confirmed unread by
│   │                            any live script (repo-wide grep, per a dedicated classification
│   │                            pass — see docs/reports/ for that agent's full findings):
│   ├── sentiment_decay_0.7/, _0.95/, _0.99/  ← One-off decay-parameter sweep outputs; only
│   │                            decay=0.9 (data/sentiment_decay/, NOT archived, still live) is
│   │                            actually referenced by design.md/reports.
│   ├── sentiment_baseline_new/ ← Stale one-off run, 0 code references (only a 2026-07-04 log).
│   ├── sentiment_baseline_phobert/, sentiment_embedding_body/ ← Empty (0 bytes), no data lost.
│   ├── sentiment_embedding_body_pilot/ ← Real embedding data, but 0 code references anywhere.
│   └── processed_all/        ← Orphaned: both its producer (process_all_available.py) and only
│                                 consumer (train_all_baselines.py) were already archived earlier
│                                 the same day.
│   NOT archived from this pass: `objective_embedding/` (422K, write-only — produced by
│   extract_objective_embeddings.py but never read by anything, since
│   2026-07-15_objective_news_baseline never got a training script). Left alone pending a
│   decision on that baseline's own status (finish vs. formally close) — archiving the data
│   without that decision would be premature.
│
├── lstm_gat_hybrid_legacy/   ← **[2026-08-02, user decision]** The original
│   │                            `MultiStockDatasetWithGraphMethod` class + its non-"_fixed"
│   │                            wrapper `create_multi_stock_dataloaders_with_graph_method`
│   │                            (both formerly in src/lstm_gat_hybrid/dataset_with_graph_method.py)
│   │                            — confirmed dead code: every live baseline routes through
│   │                            `_load_raw_stock_data`/`_split_raw_data_by_date`/
│   │                            `create_multi_stock_dataloaders_with_graph_method_fixed` (the
│   │                            "_fixed" split-first pipeline, kept, contains the P1.2
│   │                            date-alignment fix) or `MultiStockDatasetWithPreSplitData`
│   │                            (dataset_presplit.py) instead. The class's own 9 tests were
│   │                            failing (confirmed cause: this class shares remove_outliers()
│   │                            with the live pipeline, and the P1.2 fix changed it from
│   │                            dropping outlier rows to winsorizing them — one test assumed
│   │                            the old drop behavior) — rather than patch tests for dead code,
│   │                            removed the code + its exclusive dependents:
│   ├── test_simple_fix.py, test_data_leakage_fix.py, check_data_leakage.py,
│   │   check_training_metrics.py  ← root-level ad-hoc debug scripts, confirmed to import ONLY
│   │                                 the now-removed class/wrapper, nothing else live.
│   └── tests/test_dataset_data_leakage.py, test_dataset_edge_cases.py ← the class's own
│                                    dedicated test files (formerly tests/lstm_gat_hybrid/).
│       NOT archived/moved: tests/lstm_gat_hybrid/test_date_alignment_fix.py (tests the live
│       "_fixed" pipeline instead, 9/9 passing) and tests/test_graph_utils.py's 4 spillover-graph
│       failures (unrelated — missing optional `statsmodels` dependency, not dead code; spillover
│       graph itself is used by baselines/2026-07-26_spillover_qlike_baseline, still live).
│
├── vn100_scripts/            ← 5 VN100-exclusive root-level scripts, 2026-08-02 (user decision):
│                                 train_and_test_vn100.py, process_vn100_data.py, evaluate_vn100.py,
│                                 evaluate_vn100_simple.py, evaluate_vn100_train_only.py.
│                                 Confirmed not imported by any other live script before moving.
│                                 NOT archived: src/experiment/train_with_config.py,
│                                 src/data/crawl_vietnam_stocks.py, src/data/combine_datasets.py,
│                                 src/sentiment/data_collection/tickers.py — these are generic,
│                                 multi-universe utilities also used for VN30/other scopes, not
│                                 VN100-exclusive, so archiving them would remove VN30 functionality.
│
│   `data_raw/all_available/` + `data_scripts/process_all_available.py` — **[ARCHIVED 2026-08-02,
│   user decision]** distinct, broader "generalization test" universe (210 stocks), not literally
│   "VN100", but confirmed out of paper scope too. Verified via grep: `process_all_available.py`
│   was the only file referencing `all_available` anywhere, and nothing imports that script.
│   `data/processed_all/` (the script's output) was left in place pending a separate downstream
│   usage check — see the data/ subfolder classification pass, same date.
│
├── baselines/ (continued) — **[2026-08-02, large batch]** 12 more null/rejected/inconclusive
│   baselines, cross-referenced against docs/report_2026-{06-27,07-11,07-18,07-25,08-01}/ (5
│   parallel investigation passes, one per report). Full list + reasons + what was deliberately
│   NOT archived (dual_group_news_embedding_baseline, expand_news_cache_baseline, macro_news_baseline,
│   objective_news_baseline): see `2026-08-02_2200_archive_batch_log.md` in this directory.
│
├── src_legacy/
│   └── sentiment_baseline/   ← src/sentiment_baseline module, only importers archived same batch.
│
├── docs_reports_legacy/
│   ├── report_2026-06-27/    ← historical snapshot, zero live references, 04_code/ confirmed
│   │                            stale duplicate of live src/lstm_gat_hybrid/ modules.
│   ├── report_2026-07-11/    ← historical snapshot, zero live references.
│   └── report_2026-07-18/    ← historical snapshot, zero live references.
│       NOT archived: docs/report_2026-07-25/ (still cited by the current main report) or
│       docs/report_2026-08-01/ (the current main report).
│
└── [future archives]
```

---

## 🚨 Data Leakage Scripts Archive

**Location:** `archive/data_leakage_scripts/`

**Status:** 🔴 **CRITICAL - DO NOT USE**

**What Happened:**
These scripts used `torch.utils.data.random_split` for time series data, causing:
- Future data to leak into training set
- Inflated performance metrics
- 17.8x worse test performance in reality

**Impact:**
- Test RMSE: 0.009943 (with leakage) vs 0.000557 (correct) → **17.8x worse**
- Val-Test Gap: 0.00937 (with leakage) vs 0.000094 (correct) → **99x larger**

**For Detailed Information:**
See: `archive/data_leakage_scripts/README.md`

---

## 🧪 PatchTST-GAT baseline (negative control) — archived 2026-09-04

**Location:** `archive/baselines/2026-09-04_patchtst_gat/` (moved from `baselines/` via `git mv`, history preserved; 12 tracked files).

**What it is:** a baseline that swaps the LSTM temporal backbone of VolGA for a from-scratch PatchTST
(patch transformer) encoder feeding the same vol→PK GAT. Two variants (no-graph / +GAT).

**Why archived (not deleted):**
- The GPU walk-forward sweep was abandoned: PatchTST ran ~50× slower than the LSTM (one fold ≈ 1.5 h;
  a full vn100+vn30 × 4-horizon sweep ≈ 11 days), infeasible on the RTX 4060.
- The wiring smoke (not a result) showed it is not competitive with HAR (PatchTST QLIKE 0.514 vs
  HAR 0.467; +GAT overfits to 1.216) — consistent with the standing finding that HAR is hard to beat
  and non-LSTM backbones (CryptoMamba, TimesNet) did not help.
- Its tests pass from the baseline `code/` dir (14/14) but **segfault** (MKL/OpenMP einsum access
  violation) under the pre-push gate's repo-root `pytest` invocation (a sibling baseline imports
  torch multi-threaded first, so the baseline's thread-pin applies too late). This blocked pushes.

**Scope:** kept as a reference implementation of a transformer backbone + GAT; NOT part of the paper
(cited only as an abandoned negative-control probe). Out of scope for all audits/reviews per the
banner above. No file outside the baseline imports it (verified by grep before the move).

---

## 🗃️ Non-clean S&P 500 enriched — archived 2026-09-05

**Location:** `archive/data_processed_enriched/sp500/` (moved from `data/processed_enriched/sp500/`;
gitignored data folder, `mv` not `git mv` — 0 tracked files, no history to preserve; 544 tickers).

**What it is:** the earlier enriched S&P 500 output, built before the vintage/liquidity screen. Superseded
by `data/processed_enriched/sp500_clean/` (504 tickers), the canonical dataset produced by
`scripts/etl_audit/clean_sp500_vintage.py` (vintage cut rows < 2000-01-01 + ≤50% zero-range liquidity
screen + <252-row history screen + P3 ETL row-clean).

**Why archived (not deleted):**
- No active code reads it — `grep -rn "processed_enriched/sp500"` over `src/ scripts/ baselines/` hits only
  `sp500_clean` (in `clean_sp500_vintage.py` and `sp500_dirtiness_html.py`); nothing imports/loads the
  non-clean folder.
- Not an input to the clean pipeline: `clean_sp500_vintage.py` derives `sp500_clean` from **raw**
  (`data/raw/prices/sp500`), not from this enriched folder — so archiving it cannot break regeneration.
- The SP500 Colab/A100 training bundle uses `sp500_clean`, confirming clean is the live dataset.

**Scope:** kept local-only (large; SP500 redistribution-restricted) — `.gitignore` updated to ignore the
new path. Out of scope for all audits/reviews per the banner above.

**Deliberately NOT moved:** `data/processed_enriched/sp500_clean/` (the live canonical dataset),
`data/raw/prices/sp500` and `data/raw/prices/sp500_clean` (raw sources), and `data/processed/sp500`.

---

## ✅ Safe to Use (Not Archived)

These files are CORRECT and use proper temporal split:

### In Project Root:
- ✅ `train_all_models_vn30.py` - Calls correct scripts
- ✅ `train_all_with_validation.py` - Calls correct scripts

### In Source Directories:
- ✅ `src/lstm_har_enhanced/train_with_overfitting_prevention.py` - Best choice
- ✅ `src/lstm_har_enhanced/archive/train_with_validation_DEPRECATED_2026-06-20.py` - Old version
- ✅ `src/lstm_baseline/train_with_validation.py` - Temporal split
- ✅ `src/lstm_har_baseline/train_with_validation.py` - Temporal split

---

## 📚 Key Rules

### Time Series Data Splitting:

**❌ NEVER:**
```python
# Random split shuffles time → data leakage
train, val, test = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
```

**✅ ALWAYS:**
```python
# Temporal split maintains chronology
from src.common.temporal_split import TemporalSplitter
splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train, val, test = splitter.create_dataloaders(dataset)
```

---

## 🎯 Purpose of Archive

1. **Educational:** Examples of what NOT to do
2. **Historical:** Track project evolution
3. **Comparison:** Show before/after fixes
4. **Prevention:** Prevent future mistakes

---

**Last Updated:** 2026-06-21
**Maintained by:** Stock Volatility Prediction Team
