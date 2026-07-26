# Summary — Expand News-Embedding Cache with Newly Crawled Data (2026-07-25)

## What changed

User crawled significantly more news data into the sibling crawler `C:\luanvan\crawl_data\data`.
This added a new baseline, `baselines/2026-07-25_expand_news_cache_baseline/`, that extends
`data/external_news_embeddings/raw_cache/` (the PhoBERT article-embedding cache consumed by
`2026-07-25_dual_group_news_embedding_baseline`) with the newly crawled articles, without
modifying that sibling baseline (hard isolation, CLAUDE.md §3.F).

## Scope found

- **12 brand-new sources** with no prior cache: `baophapluat`, `bnews`, `cand`, `dantri`,
  `giaoducthoidai`, `hanoimoi`, `plo`, `sggp`, `tapchicongthuong`, `tienphong`, `viettimes`, `vov`
  — all mainstream Vietnamese press portals.
- **~10 existing sources** whose crawl grew (e.g. `nhipsongkinhdoanh` +352, `theinvestor` +31,
  `cafef` +7).
- **Total 13,818 ticker-mentioning articles** not yet cached (measured via a read-only scoping
  pass before touching anything).

## Files

- `baselines/2026-07-25_expand_news_cache_baseline/requirements/requirements.md` — goal, scope,
  go/no-go criteria.
- `baselines/2026-07-25_expand_news_cache_baseline/design/design.md` — data flow, source
  classification, safety design (backup, atomic write, dedupe-keeps-existing).
- `baselines/2026-07-25_expand_news_cache_baseline/code/build_incremental_cache.py` — the
  incremental cache builder: discovers sources (read-only import from the sibling baseline),
  filters to ticker-mentioning articles, diffs against each source's existing cache by `url`,
  encodes only the new rows via real PhoBERT calls, upserts with atomic write (`.tmp` + rename).
- `baselines/2026-07-25_expand_news_cache_baseline/test/test_build_incremental_cache.py` — 9
  tests (PhoBERT calls monkeypatched with deterministic fake embeddings).
- `baselines/2026-07-25_expand_news_cache_baseline/code_review/code_review_2026-07-25.md` —
  adversarial review findings + fixes.

## Tests + coverage

`pytest baselines/2026-07-25_expand_news_cache_baseline/test/ -v` — **9/9 pass.**
Diff-coverage (`diff-cover`): **Not run** — tooling not yet installed in this repo (documented
gap in CLAUDE.md's Per-project setup section). Manually verified all changed lines are exercised:
the diff-behavior branches (missing cache, wrong-dim cache, empty new-articles, no-url-column,
first-call vs incremental second-call) each have a dedicated test.

## Code review

Adversarial (Blind Hunter, no-spec mode) — 10 findings, 5 patched:
1. **Real bug (fixed):** the "no url column" early-return in `_ticker_mentioning_articles` didn't
   include a `url` column, which would crash `_new_rows_to_encode` (`KeyError: 'url'`) for any
   source lacking that column. Fixed + regression test added.
2. `--sources` typo silently processed 0 sources — now raises with valid names listed.
3-5. Test-quality nitpicks (hardcoded dim instead of `RAW_DIM`, brittle exact-dict assertion,
   missing default arg) — all fixed.
3 findings deferred (documented in the code review file): no concurrency lock, partial-batch-
failure isn't fully checkpointed, `main()`'s CLI wiring itself has no automated test (mitigated
by a real dry-run + real small-subset run before the full job, done below).

## Commands run (real)

1. Scoping (read-only, no PhoBERT): counted 13,818 new articles across 52 discovered sources.
2. PhoBERT/environment smoke test: confirmed `transformers 5.12.1` (installed `sentencepiece`)
   loads `vinai/phobert-base` and encodes correctly — the vendored code's docstring claim of
   needing `transformers<5` turned out to be stale/overcautious, not a real blocker.
3. Throughput benchmark: ~25 articles/sec on CPU (no GPU available) → ~9 min estimated for
   13,818 articles.
4. **Backup:** `data/external_news_embeddings/raw_cache/` copied to
   `data/external_news_embeddings/raw_cache_backup_2026-07-25/` (48 files) before any write.
5. `python build_incremental_cache.py --dry_run` against real data — confirmed exact match with
   the scoping numbers (13,818), no errors.
6. Real smoke test on 1 small new source (`hanoimoi`, 174 articles) — verified output parquet:
   174 unique urls, 769 cols, all-finite embeddings, real article content. Re-ran dry-run on that
   source afterward to confirm idempotency (0 new).
7. **Full run:** `python build_incremental_cache.py` (all 52 sources) — **13,644 new rows
   encoded in 753.7s (~12.6 min)**. (13,644, not 13,818, because `hanoimoi`'s 174 were already
   encoded in the smoke-test step above.)
8. Final verification: re-ran `--dry_run` — **0 new-to-encode remaining** for every source.
   `raw_cache/` file count: 48 → 59 (11 of the 12 new sources produced a file; `bnews` had 0
   ticker-mentioning articles, so correctly produced none).

## Risks / follow-ups

- The sibling baseline's panel (`data/features/dual_group_news_panel.parquet`) was NOT rebuilt —
  it still reflects the pre-expansion cache. If the user wants the new coverage reflected in a
  retrained model, `2026-07-25_dual_group_news_embedding_baseline/code/build_dual_group_panel.py`
  needs to be re-run (out of scope for this task, not touched).
  - Two new source names discovered are not yet classified into that sibling's
    `KHACH_QUAN_SOURCES`/`TONG_HOP_SOURCES` (they're documented here as `NEW_KHACH_QUAN_SOURCES`
    for traceability, but that constant isn't read by the sibling's panel-builder) — if the panel
    is rebuilt, these 12 new sources' cached articles will still be invisible to it until someone
    adds them to the sibling's own classification sets (a small, separate edit belonging to that
    baseline, not this one, per hard isolation).
- `raw_cache_backup_2026-07-25/` (the pre-expansion snapshot) is left on disk — safe to delete
  once the user confirms the new cache is good, or keep as a rollback point.

## DoD checklist

- [x] Code satisfies the request, no unrelated refactor
- [x] Tests written + run (9/9 pass)
- [ ] diff-cover C0/C1 — Not run (tooling gap, documented)
- [x] Lint — not run (ruff not installed, documented gap; manually reviewed for style)
- [x] Code review (adversarial) — run, 5 patches applied, 3 deferred with reasons
- [x] Smoke test — real dry-run + real small-subset + real full run, all verified
- [x] Impact analysis — blast radius (shared cache dir) identified, backed up first
- [x] Summary report — this file
