# Repo Root Cleanup Audit — 2026-08-16

Scope: audit root-level clutter in `C:\luanvan\stock_vol_prediction01` and archive long-unused,
non-referenced files under `archive/`, preserving git history. Conservative policy: a file is moved
only when it is BOTH stale (last commit before ~2026-07) AND not referenced/imported by any active
code, config, hook, or `CLAUDE.md`. When in doubt: recommend, do not move.

Hard-excluded from this pass (untouched): `.git`, `.worktrees/`, `data/`, `results/`, `models/`,
`mlruns/`, `.claude`, `.superpowers`, `src/`, `scripts/`, `tests/`, `baselines/`, `requirements/`,
`docs/` (no bulk archive), `requirements*.txt`, `pytest.ini`, `ruff.toml`, `.mcp.json`, `CLAUDE.md`,
`README.md`, `project-context.md`, `AGENTS.md`, `skills-lock.json`.

Only non-`.py` clutter was moved. `.py` moves are deferred to a follow-up pass because the pre-push
TDD gate (`scripts/git_hooks/pre-push`, step 0/4) blocks any non-test `.py` change — including the
deletion side of a `git mv` — that ships without an accompanying test change. `QG_SKIP` was not used.

## (a) Archived items

Method per item: `git log -1` for staleness; grep of the active tree (excluding `archive/` and
`.worktrees/`) across `*.py`, `scripts/`, config, hooks, `README.md`, `project-context.md`,
`CLAUDE.md` for references. All items below are stale and had no reference from any kept/active file
(cross-references among the archived reports themselves move together).

### Reports and summaries → `archive/root_reports_legacy/`

| Old path | New path | Last commit | Why unused |
|---|---|---|---|
| ADVERSARIAL_REVIEW_CODE_EXHIBIT.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| BAO_CAO_PARALLEL_LSTM_GNN_UPDATED.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| BAO_CAO_SENTIMENT_ANALYSIS_CO_THAY.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| CREATE_PUBLIC_REPO_GUIDE.md | archive/root_reports_legacy/ | 2026-06-20 | no active reference |
| CRYPTOMAMBA_ENHANCED_V2_GUIDE.md | archive/root_reports_legacy/ | 2026-06-20 | no active reference (code const `CRYPTOMAMBA_CONFIG_V2` is unrelated) |
| CRYPTOMAMBA_ENHANCED_V2_IMPLEMENTATION_COMPLETE.md | archive/root_reports_legacy/ | 2026-06-20 | no active reference |
| CRYPTOMAMBA_ENHANCED_V2_READY.md | archive/root_reports_legacy/ | 2026-06-20 | no active reference |
| DATA_LEAKAGE_FIX_COMPLETED.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| DATA_LEAKAGE_FIX_QUICK_START.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| DEEPSEEK_API_PRICING_ANALYSIS.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| GEMINI_REVIEW_INTEGRATION.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| HAR_FEATURES_LEAKAGE_FIX.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| HAR_FIX_SOLUTION.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| HAR_LEAKAGE_FIX_SUMMARY.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| LLM_AGENT_PROMPTING_GUIDE.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| LOCAL_LLM_MCP_IMPLEMENTATION_SUMMARY.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| LOCAL_LLM_MCP_SETUP_GUIDE.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| LOCAL_LLM_QUICK_REFERENCE.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| MODEL_COMPARISON_FINAL_REPORT.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| MODEL_COMPARISON_SUMMARY.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| MODEL_PERFORMANCE_REPORT.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| OPEN_SOURCE_CODE_REVIEW_RESEARCH.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| QUICK_START_GITHUB.md | archive/root_reports_legacy/ | 2026-06-20 | no active reference |
| QUICK_START_LOCAL_LLM.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| README_GITHUB.md | archive/root_reports_legacy/ | 2026-06-20 | no active reference (not linked from README.md) |
| REALISTIC_SENTIMENT_ANALYSIS_SUMMARY.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| REAL_NEWS_SENTIMENT_REPORT_2026_06_26.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| SENTIMENT_ANALYSIS_REPORT_NGUYEN_QUY.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| STORAGE_ANALYSIS_SUMMARY.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| VN30_HAR_BASELINE_REPORT.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| test_results_summary.md | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| ALL_METRICS_COMPARISON.txt | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| REPORT.txt | archive/root_reports_legacy/ | 2026-06-28 | no active reference |
| quick_test_output.txt | archive/root_reports_legacy/ | 2026-06-28 | stale scratch output; no active reference |
| training_output.txt | archive/root_reports_legacy/ | 2026-06-20 | stale scratch output; no active reference |

### One-off scripts (non-`.py`) → `archive/root_oneoff_scripts/`

| Old path | New path | Last commit | Why unused |
|---|---|---|---|
| check_storage.ps1 | archive/root_oneoff_scripts/ | 2026-06-28 | only referenced by (also-archived) STORAGE_ANALYSIS_SUMMARY.md |
| setup_github_repo.sh | archive/root_oneoff_scripts/ | 2026-06-20 | one-off GitHub bootstrap; no active reference |

### Notebook artifact → `archive/root_artifacts_legacy/`

| Old path | New path | Last commit | Why unused |
|---|---|---|---|
| TimesFM_Training_Colab.ipynb | archive/root_artifacts_legacy/ | 2026-06-20 | standalone Colab notebook; only appears in a historical `.claude` permission entry (not a functional dependency) |

Total archived: 38 files (35 reports/txt, 2 shell scripts, 1 notebook). No `.py` moved; no `.py`
staged (verified: `git diff --cached --name-only -- '*.py'` empty).

## (b) Recommended for review — NOT moved

### `.py` deferred by the TDD gate (stale, no importer found in a light grep)

These are stale root one-off `.py` scripts. They cannot be `git mv`-ed in this pass without tripping
the pre-push TDD gate. Recommend a dedicated follow-up (e.g. move together with a trivial test
touch, or with an explicitly justified bypass approved by the user). Re-grep importers before moving.

| Path | Last commit |
|---|---|
| display_training_summary.py | 2026-06-28 |
| list_gemini_models.py | 2026-06-28 |
| run_full_training.py | 2026-06-28 |
| run_quick_test_no_warnings.py | 2026-06-28 |
| visualize_learning_curves.py | 2026-06-28 |
| visualize_lstm_har_results.py | 2026-06-28 |
| visualize_simple_lstm_results.py | 2026-06-28 |
| explore_dataset.py | 2026-07-11 |

### `.py` that are one-off in nature but NOT stale — user judgment needed

A large batch of root `debug_*.py`, `test_*.py` (not under `tests/`), `quick_test_*.py`, `show_*.py`,
`analyze_*.py`, `compare_models.py`, `detailed_lstm_analysis.py`, `investigate_lstm_underperformance.py`,
`monitor_training.py`, `evaluate_timesnet_checkpoint.py`, `apply_validate_fix.py`, `explore_raw_data.py`,
`generate_full_metrics_comparison.py`, `train_all_with_validation.py`, `gemini_review_example.py`,
`gemini_review_wrapper.py` all carry a last-commit date of 2026-08-04 (a bulk re-commit). They look
like one-off debug/analysis scripts, but by the staleness rule they are NOT stale, so they were left
in place pending user confirmation. TDD gate also applies.

### Referenced files kept in place (not clutter to remove blindly)

| Path | Reason kept |
|---|---|
| gemini_mcp_server.py | referenced by `.mcp.json` (active MCP server) |
| local_coder_mcp_server.py | referenced by `.mcp.json` (active MCP server) |
| VN30_PERFORMANCE_REPORT.md | referenced by active `display_results.py` / `display_training_summary.py` |
| GEMINI_MCP_SERVER_GUIDE.md | referenced by `test_gemini_mcp.py` |
| LSTM_HAR_VN30_REPORT.md | referenced by `display_training_summary.py` / `check_training_progress.py` |
| SIMPLE_LSTM_VN30_REPORT.md | referenced by `display_training_summary.py` / `check_training_progress.py` |
| graph_comparison_test.png | write-target in active `src/lstm_gat_hybrid/graph_correlation.py` |
| DO_IT_NOW.md | linked from `README.md` |
| vietnam-stock-news-data-sources-plan.md | referenced by `project-context.md`; also recent (2026-07-11) |
| MEMORY.md | recent (2026-08-08); not stale |
| mlflow.db | tracked SQLite experiment-tracking DB; potential active state — recommend user confirm before archiving |

### Untracked / gitignored root artifacts (cannot be committed; user may delete manually)

Regenerable and already gitignored, so out of scope for `git mv`: `.coverage`, `coverage.xml`,
`coverage_final.xml`, `coverage_run.xml`, `*.log` (`quick_test_correlation.log`, `quick_test_lr001.log`,
`quick_test_output.log`, `results_sentiment_*.log`, `results_phobert_process.log`, `training_output.log`),
`TimesFM_Training_Colab_OLD.ipynb`, `__pycache__/`. Recommend manual deletion if disk tidiness is desired.

### Directories — recommend review (not moved; conservative on directories)

| Path | Last commit | Note |
|---|---|---|
| report_2026-06-27/ | (no direct commit found) | old timestamped scratch report dir; verify tracked status before acting |
| _research/ | 2026-06-28 | stale research scratch |
| temp/ | 2026-06-28 | scratch dir |
| research/ | 2026-08-01 | recent — likely active |
| notebooks/ | (n/a) | leave; may hold reference notebooks |

## (c) Before / after top-level (tracked root files)

Tracked root-level file count: 106 → 68 (38 files relocated under `archive/`).

Removed from root (now under `archive/`): all 38 files listed in section (a) — the `BAO_CAO_*`,
`CRYPTOMAMBA_ENHANCED_V2_*`, `DATA_LEAKAGE_FIX_*`, `HAR_*`, `LOCAL_LLM_*`, `MODEL_*`, `QUICK_START_*`,
`*SENTIMENT*`, `VN30_HAR_BASELINE_REPORT.md`, `ALL_METRICS_COMPARISON.txt`, `REPORT.txt`,
`quick_test_output.txt`, `training_output.txt`, `check_storage.ps1`, `setup_github_repo.sh`,
`TimesFM_Training_Colab.ipynb`, etc.

Root markdown remaining after the pass: `AGENTS.md`, `CLAUDE.md`, `README.md`, `project-context.md`,
`MEMORY.md`, plus the referenced-and-kept `DO_IT_NOW.md`, `GEMINI_MCP_SERVER_GUIDE.md`,
`LSTM_HAR_VN30_REPORT.md`, `SIMPLE_LSTM_VN30_REPORT.md`, `VN30_PERFORMANCE_REPORT.md`,
`vietnam-stock-news-data-sources-plan.md` (all still referenced by active code/docs).
</content>
</invoke>
