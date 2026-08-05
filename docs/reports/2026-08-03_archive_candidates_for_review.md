# Archive candidates for review — 2026-08-03

Scope: repo-wide scan for files/folders that meet this project's own archive bar (confirmed dead
code, confirmed superseded/duplicate implementations, confirmed null/rejected/concluded
experiments, stale one-off debug scripts, orphaned data), following the standard in
`.claude/skills/archive-review/SKILL.md` and the taxonomy already in `archive/README.md` /
`archive/2026-08-02_2200_archive_batch_log.md`. `archive/` itself (any depth) was excluded from
this scan per its own out-of-scope banner.

**This is a listing only. Nothing was moved, archived, or otherwise modified.** Each candidate
needs a manual decision by the user.

Verification method used throughout: `git log -1 --format=%ad -- <path>` for last-touch date, and
`grep -rl "<name>"` scoped to `src/ baselines/ docs/ tests/` (excluding `archive/`) for real
importers/citations — not comment/docstring mentions of the same string.

---

## 1. Root-level files

The repo root carries ~100 tracked files that predate the current `baselines/`/`src/`/`docs/`
structure (CLAUDE.md §3.D, §3.F). All clusters below were verified to have **zero real
cross-references** from any currently-live `.py`/`.md` file outside their own cluster, and all but
one file were last touched 2026-06-20–2026-06-28 (i.e., before the baseline-folder convention
existed) or 2026-07-11.

### 1a. Pre-baseline-era report/guide docs (high confidence dead)

| Path | Why | Evidence |
|---|---|---|
| `ADVERSARIAL_REVIEW_CODE_EXHIBIT.md` | Historical, superseded by `docs/reports/` | last commit 2026-06-28, 0 refs |
| `BAO_CAO_PARALLEL_LSTM_GNN_UPDATED.md` | Historical LSTM-GNN report, superseded | last commit 2026-06-28, 0 refs (its own citation to `MODEL_COMPARISON_FINAL_REPORT.md` is unqualified/ambiguous, not a real inbound reference) |
| `BAO_CAO_SENTIMENT_ANALYSIS_CO_THAY.md` | Historical sentiment report | last commit 2026-06-28, 0 refs |
| `DATA_LEAKAGE_FIX_COMPLETED.md`, `DATA_LEAKAGE_FIX_QUICK_START.md` | Historical fix write-ups; the fix itself lives in `archive/data_leakage_scripts/README.md` already | last commit 2026-06-28, cross-reference each other only |
| `HAR_FEATURES_LEAKAGE_FIX.md`, `HAR_FIX_SOLUTION.md`, `HAR_LEAKAGE_FIX_SUMMARY.md` | Historical HAR leakage fix write-ups | last commit 2026-06-28, 0 refs |
| `LSTM_HAR_VN30_REPORT.md`, `MODEL_PERFORMANCE_REPORT.md`, `SIMPLE_LSTM_VN30_REPORT.md`, `VN30_HAR_BASELINE_REPORT.md`, `VN30_PERFORMANCE_REPORT.md` | Historical single-model reports, superseded by later baseline reports in `docs/` | last commit 2026-06-28, 0 refs |
| `MODEL_COMPARISON_FINAL_REPORT.md`, `MODEL_COMPARISON_SUMMARY.md` | Historical comparison reports | last commit 2026-06-28, 0 real refs to the **root** copy specifically (see note below) |
| `OPEN_SOURCE_CODE_REVIEW_RESEARCH.md`, `STORAGE_ANALYSIS_SUMMARY.md`, `DEEPSEEK_API_PRICING_ANALYSIS.md` | One-off research notes, no downstream use | last commit 2026-06-28, 0 refs |
| `REALISTIC_SENTIMENT_ANALYSIS_SUMMARY.md`, `REAL_NEWS_SENTIMENT_REPORT_2026_06_26.md`, `SENTIMENT_ANALYSIS_REPORT_NGUYEN_QUY.md` | Historical sentiment reports | last commit 2026-06-28, 0 refs |
| `REPORT.txt`, `test_results_summary.md` | Historical generated report text | last commit 2026-06-28, 0 refs |
| `vietnam-stock-news-data-sources-plan.md` | Planning doc, last touched 2026-07-11 (newer than the rest) | 0 refs found |

**Note on `MODEL_COMPARISON_FINAL_REPORT.md`:** three current docs (`docs/reports/2026-07-25_0712_all_baselines_comparison_report.md`,
`docs/report_2026-07-25/BAO_CAO_CHO_THAY.md`) cite a file of the same name, but their citation is to
`docs/report_2026-06-27/01_main_report/MODEL_COMPARISON_FINAL_REPORT.md` — a path that is now
**inside `archive/docs_reports_legacy/report_2026-06-27/`** (moved there 2026-08-02). That citation
is already a dangling reference to an archived path (a third one, beyond the two the prior batch
patched) — separate issue from this scan, flagged here only because the name collision could
otherwise cause confusion. The **root-level** `MODEL_COMPARISON_FINAL_REPORT.md` evaluated here is
a distinct file with zero real inbound references.

### 1b. Abandoned "publish to public GitHub" initiative (worth reviewing)

| Path | Why | Evidence |
|---|---|---|
| `CREATE_PUBLIC_REPO_GUIDE.md`, `QUICK_START_GITHUB.md`, `README_GITHUB.md`, `setup_github_repo.sh` | Self-contained cluster about publishing this repo publicly; last commit 2026-06-20; no evidence the initiative proceeded (no public repo config found) | cross-reference only each other; 0 refs from anything outside the cluster |
| `DO_IT_NOW.md` | Same vintage/cluster (2026-06-20), covers old CryptoMamba/setup steps | **Still linked from root `README.md` line 15** ("🎯 Quick Start: See DO_IT_NOW.md") — this is a real reference, so flagged **worth reviewing**, not high-confidence-dead. The link itself may be stale (content predates current baseline structure) even though it's wired in. |

### 1c. Gemini / local-LLM MCP experimentation (high confidence dead)

| Path | Why | Evidence |
|---|---|---|
| `gemini_mcp_server.py`, `gemini_review_example.py`, `gemini_review_wrapper.py`, `list_gemini_models.py`, `local_coder_mcp_server.py` | Abandoned MCP-server experimentation track, unrelated to the volatility pipeline | last commit 2026-06-28; only cross-referenced by their own test files below |
| `test_gemini_20_flash.py`, `test_gemini_flash.py`, `test_gemini_mcp.py`, `test_gemini_mcp_simple.py`, `test_local_llm_mcp.py` | Ad-hoc test scripts for the above, at repo root — **not collected by pytest** (`pytest.ini` `testpaths = src, baselines, tests` only) | same cluster, 0 external refs |
| `GEMINI_MCP_SERVER_GUIDE.md`, `GEMINI_REVIEW_INTEGRATION.md`, `LLM_AGENT_PROMPTING_GUIDE.md`, `LOCAL_LLM_MCP_IMPLEMENTATION_SUMMARY.md`, `LOCAL_LLM_MCP_SETUP_GUIDE.md`, `LOCAL_LLM_QUICK_REFERENCE.md` | Docs for the same abandoned track | last commit 2026-06-28, cross-reference each other only |
| `requirements-gemini-mcp.txt`, `requirements-gemini.txt`, `requirements-local-llm.txt` | Dependency lists for the same track | 0 refs outside the cluster's own docs |
| `test_llm_agent_10_days.py` | Ad-hoc LLM-agent test at root, same era | 0 external refs, not pytest-collected |

### 1d. Root debug / one-off training scripts predating `src/`+`baselines/` structure (high confidence dead)

| Path | Why | Evidence |
|---|---|---|
| `analyze_results.py`, `apply_validate_fix.py`, `check_training_progress.py`, `compare_models.py`, `detailed_lstm_analysis.py`, `display_results.py`, `display_training_summary.py`, `evaluate_timesnet_checkpoint.py`, `generate_full_metrics_comparison.py`, `investigate_lstm_underperformance.py`, `monitor_training.py`, `run_full_training.py`, `run_quick_test_no_warnings.py`, `show_final_metrics.py`, `show_full_metrics.py`, `show_model_comparison.py`, `visualize_learning_curves.py`, `visualize_lstm_har_results.py`, `visualize_simple_lstm_results.py` | Pre-`baselines/`-structure debug/reporting scripts | last commit 2026-06-20 or 2026-06-28; grep hits for their own names only landed in **vendored** `.agents/skills/*` files (coincidental name collisions, e.g. a different `compare_models`/`display_results` function defined in an unrelated vendored skill script) — no real project reference |
| `debug_detailed_training.py`, `debug_fixed_lr.py`, `debug_model_predictions.py`, `debug_training_process.py` | Same cluster | last commit 2026-06-20, 0 refs |
| `train_all_models_vn30.py`, `train_all_with_validation.py`, `train_har_vn30.py`, `train_lstm_har_vn30.py`, `train_simple_lstm_vn30.py` | Pre-baseline top-level training entry points, superseded by `baselines/*/code/train*.py` per-baseline scripts | last commit 2026-06-20 to 2026-08-02; the one apparent "reference" to `train_lstm_har_vn30` is a function of the same name **defined inside** `train_all_models_vn30.py` itself (dead code calling dead code), not an external caller. Note: `archive/README.md`'s "Safe to Use" section (dated 2026-06-21, itself stale) previously listed these two as correct-vs-leaky examples — that note predates the current baseline structure and does not reflect current usage. |
| `test_har_leakage_fix.py`, `test_improved_graphs.py`, `test_normalized_training.py`, `test_parallel_model.py`, `test_phase1_implementation.py`, `test_sentiment_10_days.py` | Root ad-hoc test scripts, not pytest-collected (`testpaths` excludes root) | last commit 2026-06-28, 0 external refs |

### 1e. RSS/news one-off investigation scripts (high confidence dead)

| Path | Why | Evidence |
|---|---|---|
| `analyze_rss_content.py`, `test_alternative_feeds.py`, `test_cafef_rss.py`, `explore_dataset.py`, `explore_raw_data.py` | One-off RSS/data exploration scripts, last touched 2026-07-11 | 0 refs anywhere else in repo |
| `collect_real_news_friday.py` | One-off news collection script | last commit 2026-06-28, 0 refs |
| `quick_finbert_test.py`, `quick_finbert_test_trading_day.py`, `quick_finbert_test_windows.py`, `quick_test_single_stock.py`, `quick_test_training.py` | Ad-hoc FinBERT/training smoke scripts, cross-reference only each other | last commit 2026-06-20/28, 0 external refs |

### 1f. Stale generated artifacts left in repo root (high confidence dead / stray)

| Path | Why | Evidence |
|---|---|---|
| `ALL_METRICS_COMPARISON.txt`, `quick_test_output.txt`, `training_output.txt` | Git-tracked generated output text, 0 refs | last commit 2026-06-20/2026-06-28 |
| `graph_comparison_test.png` | Git-tracked generated image. Its producer function still exists in **live** code (`src/lstm_gat_hybrid/graph_correlation.py:299`, `visualize_graph_comparison(...)`), so this isn't dead code — just a stray committed output artifact from a one-off run. Worth reviewing (safe to delete/regenerate, arguably shouldn't be git-tracked at all) rather than high-confidence-dead. | last commit 2026-06-28 |
| `coverage.xml`, `coverage_final.xml`, `coverage_run.xml`, `mlflow.db`, `quick_test_correlation.log`, `quick_test_lr001.log`, `quick_test_output.log`, `results_phobert_process.log`, `results_sentiment_newdata.log`, `results_sentiment_resume.log`, `results_sentiment_resume2.log`, `results_sentiment_train.log`, `training_output.log` | **Not git-tracked** (`git status --ignored` confirms all `!!`) — local generated/gitignored artifacts, not part of the repository. No archive action applicable; listed only for completeness. | `git status --porcelain --ignored` |

### 1g. Notebooks and empty leftover directory

| Path | Why | Confidence |
|---|---|---|
| `TimesFM_Training_Colab_OLD.ipynb` | Name signals it's an explicit superseded duplicate of `TimesFM_Training_Colab.ipynb` (same dir, "_OLD" suffix); untracked by git (`git ls-files` returns nothing for it), unreferenced anywhere | High confidence dead |
| `report_2026-06-27/` (root-level, distinct from the already-archived `docs/report_2026-06-27/`) | Entirely empty directory tree (`find -type f` → 0 files; `01_main_report/`, `02_technical_docs/`, `03_training_results/`, `04_code/`, `docs/report_2026-06-27/...` are all empty subfolders). Not git-tracked, not gitignored. Likely a leftover shell from an earlier reorganization. | High confidence — trivially removable (nothing to lose since it's empty), not really an "archive" case since there's no content |

---

## 2. `src/`

| Path | Why | Confidence | Evidence |
|---|---|---|---|
| `src/experiment/*.py` (22 files: `calculate_all_metrics.py`, `check_extremes.py`, `compare_crawl_results.py`, `compare_models.py`, `compare_relu_models.py`, `debug_evaluation.py`, `debug_gradient_flow.py`, `debug_model_output.py`, `debug_predictions.py`, `debug_scaling.py`, `debug_training_failure.py`, `debug_val_loss.py`, `demonstrate_data_leakage.py`, `display_final_results.py`, `full_test_evaluation.py`, `optimize_lstm.py`, `optimize_lstm_fast.py`, `run_pipeline.py`, `show_metrics.py`, `show_mse_results.py`, `train_best_lstm.py`, `visualize_architecture.py`) | All last touched 2026-06-20; only inbound references found are from **historical docs** (`docs/project/FOLDER_REORGANIZATION_2026-06-18.md`, `DEBUG_FILE_RULE_SUMMARY.md`, `COMPLETE_SUMMARY.md`, `LSTM_OPTUNA_OPTIMIZATION_GUIDE.md`, `LESSON_LEARNED_LSTM_FAILURE.md`), never from live `.py`. This matches an **already-open action item**: `docs/reports/2026-08-02_1547_consolidated_fix_plan.md` §4 explicitly lists `train_best_lstm.py`/`optimize_lstm*.py` as needing a "confirm none of their numbers are cited anywhere before deciding whether to fix or just delete/archive" decision — this scan confirms zero citations of their numbers in any current report. | High confidence dead | grep scoped to `src baselines docs tests`, 0 real importers; `src/experiment/train_with_config.py` is explicitly excluded — already confirmed live/generic per `archive/README.md`'s VN100 pass, not part of this candidate group |
| `src/experiment/cryptomamba_baseline/` (`config.py`, `config_full.py`, `config_v2.py`, `dataset.py`, `model.py`, `model_full.py`, `model_v2.py`, `train.py`, `train_full.py`, `train_v2.py`, `__init__.py`) | Superseded: last touched 2026-06-20 (V1/V2/full iterations). The **live** version is the top-level `src/cryptomamba_baseline/` (`config_enhanced.py`, `model_enhanced.py`, `train_enhanced.py`), last touched 2026-08-02 and confirmed still in-scope by `docs/reports/2026-08-02_1056_paper_readiness_audit_report.md:100` and `2026-08-03_0020_summaryOfUpdate_report.md:182` (both discuss it as a live single-ticker pipeline that got the seeding fix in commit `fccaf6a`). No baseline/report references the `experiment/` V1/V2/full versions specifically. | High confidence superseded duplicate | import-line grep confirms these files only import each other (`src.cryptomamba_baseline.model`, `.config`, etc. — the **non**-enhanced names); the enhanced module (different filenames: `model_enhanced.py`, `config_enhanced.py`) is a separate, still-current implementation |
| `src/body_pilot/` (`extract_pilot_body.py`, `test/test_extract_pilot_integration.py`) | One-off pilot: "Quick test whether article BODY text (vs title-only) moves volatility DirAcc." Self-contained, reads from a **sibling repo path** (`../crawl_data/`), writes `unified_articles_pilot_body.csv` there — not consumed by any baseline or report found. Last touched 2026-07-18. | Worth reviewing | zero importers found in `src baselines docs tests`; no requirements.md exists (not a `baselines/` folder) so no explicit go/no-go to check — ambiguous whether the pilot's conclusion was ever acted on or is still pending, hence not "high confidence" |

**Explicitly checked and NOT flagged:** `src/lstm_har_gat_hybrid/` (last commit **2026-08-04**, i.e. touched most recently of anything in this scan; referenced as a live, still-in-scope model family in `docs/reports/2026-08-02_1056_paper_readiness_audit_report.md` and `2026-08-03_0020_summaryOfUpdate_report.md`, and got the seeding fix in the same commit as `cryptomamba_baseline`) — this is a **different** module from the already-excluded `src/lstm_gat_hybrid/` (no "har") and is itself still active, not a duplicate of it. `src/timesnet_baseline/` (last commit 2026-08-04, discussed as live/unresolved in the same audit report). `src/cryptomamba_baseline/` top-level (last commit 2026-08-02, live per above).

---

## 3. `baselines/`

No new candidates found beyond what the prior batch already logged. All 10 remaining folders were
checked against `archive/2026-08-02_2200_archive_batch_log.md`'s existing notes:

- `2026-07-15_objective_news_baseline` — already flagged in the prior log as needing a human
  decision (finish vs. formally close); still unresolved, not a new finding.
- `2026-07-25_expand_news_cache_baseline`, `2026-07-25_macro_news_baseline` — already flagged in
  the prior log as "not covered by any of the 5 reports' own conclusions, left as-is pending a
  future decision"; still unresolved, not a new finding.
- `2026-07-25_dual_group_news_embedding_baseline` — confirmed live dependency (its
  `data/external_news_embeddings/` output is read by `2026-07-25_expand_news_cache_baseline` and
  is the vendor source `2026-07-26_per_ticker_news_gate_baseline`/`2026-08-01_calendar_news_gate_baseline`
  build on); not a candidate.
- `2026-07-26_per_ticker_news_gate_baseline`, `2026-07-26_spillover_qlike_baseline`,
  `2026-08-01_calendar_news_gate_baseline`, `2026-08-01_horizon10_baseline`,
  `2026-08-01_horizon1_baseline`, `2026-08-01_horizon22_baseline` — all recent (2026-07-26 to
  2026-08-01), part of the current active lineage per `docs/claude_memory` and the latest reports;
  not candidates.

---

## 4. `docs/`

No new candidates. `docs/baseline/` is an empty directory (0 files) — trivial, not worth an
archive action. `docs/lstm/`, `docs/suggestion/`, `docs/paper/`, `docs/project/`,
`docs/report_2026-07-25/`, `docs/report_2026-08-01/` were all checked and are still cited from
current docs (see §5 below for specifics) — none are orphaned.

`data/objective_embedding/` (422K) — already noted in `archive/README.md` as deliberately **not**
archived, pending a decision on `2026-07-15_objective_news_baseline`'s own status. Not a new
finding; mentioned only as a reminder this decision is still open.

---

## 5. `results/` / `models/`

Per this project's own rule (CLAUDE.md §3.F.6 / `archive/README.md`), timestamped run outputs are
*intentionally* kept in root `results/`/`models/` regardless of whether the baseline code that
produced them later moves to `archive/` — so the ~335 timestamped folders in `results/` and ~89 in
`models/` were **not** individually audited; doing so would contradict the project's own retention
convention. Only loose, non-timestamped stray files sitting directly in `results/`'s root were
checked (these are the ones that don't fit the "one folder per run" convention at all):

| Path | Why | Confidence | Evidence |
|---|---|---|---|
| `results/vn100_evaluation_20260622_210016.csv` | Orphaned output of the VN100 track, which was dropped from project scope entirely 2026-08-02 (per `archive/README.md`'s `vn100_scripts/` entry) | High confidence dead | 0 refs anywhere in `src baselines docs tests` |
| `results/model_comparison_2026-06-20_070759.json`, `_070806.json`, `_070815.json` | Early cryptomamba/LSTM comparison runs, same 2026-06-20 vintage as the `src/experiment/` cluster above | High confidence dead | 0 refs |
| `results/MODEL_COMPARISON_FINAL_2026-06-28.json` | Same vintage as the root `MODEL_COMPARISON_FINAL_REPORT.md` (§1a) | High confidence dead | 0 refs |
| `results/all_metrics_comparison_2026-06-19_073434.json` | Stray duplicate-looking file; note its sibling `..._073515.json` **is** referenced (by `docs/project/EMBEDDING_BASELINE_REPORT_2026-07-08.md`, itself still cross-referenced from other `docs/project/` files) so was **not** flagged | High confidence dead (this specific file only) | 0 refs for `_073434`; `_073515` kept |
| `results/_latent_noise_resume.log`, `_latent_noise_resume15.log`, `_latent_noise_train.log`, `_tmp_horizon_analysis.py` | Leading-underscore scratch/temp naming, 0 refs | High confidence dead | 0 refs |

**Pre-existing, not a new finding:** `models/archive/` (2 folders: `har_baseline_2026-06-15_231300`,
`lstm_baseline_2026-06-16_000100`) and `results/archive/` (empty) are ad-hoc local "archive"
subfolders that predate the unified top-level `archive/` taxonomy and aren't mentioned in
`archive/README.md`. Not flagged as new candidates — just noting they exist in case the user wants
to eventually consolidate them into the canonical `archive/` location (a taxonomy question per the
archive-review skill's Step 2, not a "is this dead" question).

---

## 6. Not evaluated in depth (out of scope for this pass)

`_research/timesfm-google/` (a vendored external git clone, has its own `.git/`), `research/`
(planning docs for a future benchmark-dataset expansion), `_bmad-output/` (BMAD workflow
planning/implementation artifacts) — these are tooling/vendored/planning workspaces rather than
project source or experiment results, and don't fit neatly into the archive-review skill's
"code/data/docs" framing. Not assessed for dead/live status here; flagging their existence only in
case the user wants a separate pass.

---

## NOT flagged — checked and confirmed still needed

- `src/lstm_har_gat_hybrid/`, `src/cryptomamba_baseline/` (top-level enhanced), `src/timesnet_baseline/`
  — all last touched 2026-08-02 to 2026-08-04, all discussed as live/in-scope model families in the
  two most recent audit reports (`docs/reports/2026-08-02_1056_paper_readiness_audit_report.md`,
  `docs/reports/2026-08-03_0020_summaryOfUpdate_report.md`).
- `src/experiment/train_with_config.py` — explicitly confirmed generic/multi-scenario utility in
  the prior VN100 archival pass, reused outside VN100.
- `baselines/2026-07-25_dual_group_news_embedding_baseline/` — confirmed live upstream dependency
  for 3 other baselines via `_SIBLING_CODE` sys.path injection.
- `data/external_news_embeddings/`, `data/features/`, `data/processed/`, `data/raw/` — all recently
  touched (2026-07-27 to 2026-08-02) and read by current baseline code.
- `docs/report_2026-07-25/` — still directly cited (specific numbers) by
  `docs/report_2026-08-01/BAO_CAO_TONG_HOP.md`.
- `docs/lstm/`, `docs/suggestion/`, `docs/paper/` — all still linked from `docs/README.md` or
  `docs/claude_memory/`.
- `results/lstm_architecture.png` — still cited by `docs/lstm/ENHANCED_LSTM_GUIDE.md` and
  `docs/lstm/LSTM_COMPARISON_BASIC_VS_ENHANCED.md`; not an orphan despite similar vintage to other
  flagged `results/` files.
- Root `README.md`, `CLAUDE.md`, `project-context.md` — the 3 mandated root docs per CLAUDE.md
  §3.D, untouched by this scan.

---

## Summary

The prior 2026-08-02 wave covered `baselines/`, most of `data/`, `src/sentiment_baseline/`, and 3
historical `docs/report_*` folders. This pass found a **large amount of previously-unaudited
root-level clutter** (~90 files: historical reports, an abandoned GitHub-publish initiative, an
abandoned Gemini/local-LLM MCP experimentation track, and pre-`baselines/`-structure debug/training
scripts — all dated 2026-06-20 to 2026-07-11, all confirmed zero-real-reference), plus a smaller,
well-evidenced set of `src/` and `results/` candidates (`src/experiment/*.py` cluster,
`src/experiment/cryptomamba_baseline/` superseded duplicate, a handful of orphaned `results/` root
files). No new `baselines/`, `docs/`, or `data/` candidates were found beyond what the prior wave
already identified and left pending — those areas appear to have been thoroughly covered already.
