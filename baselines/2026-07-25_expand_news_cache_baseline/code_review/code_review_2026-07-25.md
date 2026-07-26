# Code Review — 2026-07-25 (Blind Hunter, no-spec mode)

Reviewed: `code/build_incremental_cache.py` + `test/test_build_incremental_cache.py` (diff-only,
no other project context loaded into the review).

## Fixed (patch)

1. **Real bug** — `_ticker_mentioning_articles`'s "no url column" early-return
   (`df.iloc[0:0].assign(_text="")`) produced a frame WITHOUT a `url` column. Its caller
   `_new_rows_to_encode` unconditionally does `articles["url"]`, so any discovered source whose
   `load_source()` output genuinely lacks a `url` column would crash with `KeyError: 'url'`
   instead of the intended graceful "0 articles" result. Fixed to
   `pd.DataFrame({"url": [], "_text": []})`. Added regression test
   `test_ticker_mentioning_articles_no_url_column_returns_empty_with_url_col`.
2. `main()`'s `--sources` filter silently processed 0 sources on a typo'd name. Added an
   unknown-source validation that raises with the list of valid discovered names.
3. Test hardcoded `768` instead of importing `RAW_DIM` — fixed (`saved.shape[1] == 1 + RAW_DIM`).
4. `test_run_source_first_call_encodes_all_new` used a brittle exact-dict `==` — split into
   per-key assertions so adding a new stats key later doesn't break this test.
5. `_encode_rows(batch_size: int)` had no default, inconsistent with the function it wraps
   (`extract_phobert_embeddings(..., batch_size: int = 32)`) — added `= 32` default.

## Deferred (documented, not blocking)

- **No locking/re-entrancy guard** if the script is run twice concurrently — acceptable for a
  manually-invoked one-off data-refresh script; not worth the complexity for this use case.
- **Partial-batch-failure**: if `_encode_rows` raises partway through the `main()` loop over ~50
  sources, sources alphabetically after the failure are skipped for that run. Already-written
  sources are safe (atomic per-source write) and re-running the script is idempotent (already-
  cached urls are skipped), so this just means "re-run if it dies partway" rather than data loss.
- **`main()` has no automated test** (only the helper functions are unit-tested) — mitigated by
  running a real `--dry_run` and a real small-subset run against actual `crawl_data` before the
  full run (see requirements.md go/no-go).

## Result

9/9 tests pass after patches (`pytest baselines/2026-07-25_expand_news_cache_baseline/test/ -v`).
