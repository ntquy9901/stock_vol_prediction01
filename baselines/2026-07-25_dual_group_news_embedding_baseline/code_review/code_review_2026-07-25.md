# Code Review — Dual-Group News Embedding Baseline (2026-07-25)

**Method:** Self-directed adversarial review (Blind Hunter + Edge Case Hunter + Acceptance
Auditor roles per CLAUDE.md §5). The `/code-review` skill's automated tooling (`review` skill)
expects a GitHub PR; `bmad-code-review` is an interactive multi-checkpoint BMAD workflow that
halts for human input at each step — neither fit this session (uncommitted local work, user
explicitly asked to proceed without waiting for approval). Reviewed by careful manual reading +
running the pipeline against real data and inspecting actual output, not just static reading.

## Findings

### HIGH — Train/test leakage in the shared PCA fit (CONFIRMED, FIXED)

**File:** `code/vendor_data_eda/news_embeddings.py` (`TRAIN_CUTOFF`)

The vendored `TRAIN_CUTOFF = "2020-01-01"` was copied unchanged from data_eda, which assumed a
single global calendar split date shared by every ticker. This project's actual dataset split
(`src/lstm_gat_hybrid/dataset_with_graph_method.py::_split_raw_data_by_date`) instead cuts every
ticker at the **same row index** (`train_ratio=0.7` of the *shortest* common ticker's length),
so each ticker's own val/test window lands on a **different, ticker-specific calendar date**.

Verified by walking each `{TICKER}_processed.csv` through the identical split arithmetic used by
the dataset: the earliest per-ticker val-start date across all 30 VN30 tickers is **2010-06-30**
(STB, VNM) — vs. the newest (SSB) at 2024-11-11. Under the original 2020-01-01 cutoff, roughly
19 of 30 tickers had their own val/test-period news rows fall *before* 2020-01-01, and were
therefore included in the "train" mask used to fit the shared PCA basis — a real leakage of
val/test-period information into a component the model consumes for every split.

**Fix:** `TRAIN_CUTOFF` changed to `"2010-06-30"` (the provably-safe minimum across all 30
tickers — see the inline comment in `news_embeddings.py` for the exact derivation). This
guarantees no ticker's val/test rows are ever included in the PCA fit, at the cost of a much
smaller PCA training set (~4 years of pooled news vs. ~14). Panel was rebuilt and the model
retrained from scratch after this fix; see the summary report for before/after numbers (the
practical impact on reported DirAcc turned out to be small, since PCA is a rotation/reduction,
not a fit to the target — but the methodology is now correct, which is what matters per
CLAUDE.md §3.A).

**Verdict:** CONFIRMED, fixed and re-verified (panel rebuilt, full retrain re-run, tests re-pass).

### MEDIUM — `_legacy_tong_hop_features()` crash after the TRAIN_CUTOFF fix (CONFIRMED, FIXED)

**File:** `code/vendor_data_eda/dual_news_features.py`

A direct consequence of the HIGH fix above: `_legacy_tong_hop_features()` fits its own,
non-shared PCA using only `tong_hop`-group rows before `TRAIN_CUTOFF`. With the corrected
(earlier) cutoff, `tong_hop`'s own pre-cutoff row count no longer supports the full
`PCA_DIM=32` components, so `_reduce()`'s honest fallback (`d = min(dim, ..., n_train-1)`)
silently produced <32 `emb_*` columns — and the legacy feature list's fixed 32-column
`EMB_FEATURES` then raised `KeyError` selecting columns that didn't exist.

**Fix:** stopped calling `_legacy_tong_hop_features()` in `build_advanced_features()` — this
baseline's `KEEP_COLS` filter (`build_dual_group_panel.py`) already discards those 39 legacy
columns (see next finding), so nothing downstream was using its output anyway.

**Verdict:** CONFIRMED, fixed (panel builds cleanly end to end now).

### LOW — Scope crept from the documented 146 cols to 185 (CONFIRMED, FIXED)

**Files:** `code/build_dual_group_panel.py`

requirements.md §3 (Simplicity Gate) explicitly scoped this baseline to `ADV_FEATURES_DUAL` (80)
+ `EWMA_FEATURES` (66) = 146 columns, deferring the richer multi-EWMA/novelty/dispersion set.
The first panel build emitted 185 feature columns (146 + 39 legacy `tong_hop`-only columns that
`build_advanced_features` also always attaches for data_eda's own backward-compat needs, which
don't apply here) — a silent scope-creep against the documented go/no-go criteria.

**Fix:** added an explicit `KEEP_COLS = ADV_FEATURES_DUAL + EWMA_FEATURES` filter before writing
the panel parquet, with a `RuntimeError` guard if any expected column is unexpectedly missing
(fail loud rather than silently write a differently-shaped panel again).

**Verdict:** CONFIRMED, fixed — panel now has exactly 146 feature + ticker + date = 148 columns.

### LOW — 316 articles silently excluded from the panel (BY DESIGN, user-confirmed)

**File:** `code/vendor_data_eda/news_embeddings.py` (`_get_article_embeddings`)

crawl_data has grown past the copied cache's snapshot time (2026-07-24 21:49); 316
ticker-mentioning articles across the 30 VN30 sources have no cached embedding. Per explicit
user decision (2026-07-25), these are skipped rather than encoded (hard constraint: never invoke
PhoBERT in this baseline). This is intentional, not a defect — documented here for traceability
since it does mean the panel is missing a small amount of the most recent news.

### Notes, not findings (checked, no action needed)

- **EWMA causality:** `_ewma_on_series`/`ewma_embedding_features` only ever look backward (decay
  applied trading-day by trading-day in order); computing them over the full calendar (spanning
  train/val/test) is standard practice for a temporal feature and is not leakage by itself — the
  PCA *fit* boundary (fixed above) was the actual leak, not the EWMA recursion.
- **Ticker universe mismatch (fixed during Story 2, not a residual finding):** this project's
  own `src/sentiment/data_collection/tickers.py::VN30_TICKERS` (45 entries) doesn't match the
  cache's actual ticker-mention filter; `vendor_config.py` now hardcodes the same 30-ticker list
  data_eda used, with a comment explaining why (see that file).
- **Dataset coverage (93.75%, not 100%):** `dataset_dual_news.py`'s date-match coverage against
  the panel is 93.75%, not 100% — investigated, not blocking: the panel is reindexed to the
  full 4989-day trading calendar built from all 30 tickers' price files, but the model's own
  per-sequence window can include a small number of dates at the margins that don't appear in
  that reindex. Below the "zero real matches" hard-fail threshold already coded in
  `_create_sequences`, and consistent with the pre-existing no-news-day = zero-fill convention
  used elsewhere in this project's news baselines.
- **Performance:** `load_news_panel`'s per-row `iterrows()` (146,700 rows) is not vectorized —
  a minor performance smell, not a correctness issue at this data size (build completes in
  minutes). Flagged for a future baseline if the panel grows substantially.

## Summary

1 HIGH (data leakage) + 1 MEDIUM (crash) + 1 LOW (scope creep) confirmed and fixed; 1 LOW
(article exclusion) is by design; remaining notes are non-blocking. All 6 project tests
(`test_build_panel_smoke.py`, `test_dataset_smoke.py`, `test_model_smoke.py`) pass after fixes.
