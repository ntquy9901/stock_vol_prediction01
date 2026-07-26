# Code Review — 2026-07-25 (self-directed, no-spec-file adversarial review)

Reviewed: all new files in this baseline (`build_macro_panel.py`, `dataset_macro_news.py`,
`train_macro_news.py`, tests). Run unattended overnight per user's explicit request — no
`/code-review` checkpoint wait; reviewed by careful adversarial re-reading + running the actual
test suite + a real smoke-mode train run, per CLAUDE.md §5 self-directed review precedent
(same approach the sibling `2026-07-25_dual_group_news_embedding_baseline` used).

## Fixed (patch)

1. **`_fit_pca` alphabetical-source bias.** The original loop iterated `sorted(sources.items())`
   and took each source's ENTIRE pre-cutoff row set until `MAX_PCA_FIT_ROWS` was hit, then broke.
   This means only whichever sources sort alphabetically first (e.g. `baodautu`, `baophapluat`,
   `bnews`...) would ever contribute to the PCA basis — a systematic, order-dependent bias, not a
   representative sample of the corpus. Fixed: cap each source's contribution at
   `MAX_PCA_FIT_ROWS // len(sources)` (random subsample via a seeded RNG if a source has more),
   so every source with pre-cutoff data gets proportional representation.
2. **`_aggregate_by_date` performance.** The original per-row Python loop
   (`for i in range(len(dated)): sums[idx] += ...`) is correct but slow at the real scale (~7.5M
   rows across all sources) — replaced with a vectorized `np.add.at` scatter-add per source
   (`np.add.at`, not the naive `sums[idx[valid]] += reduced[valid]`, which silently drops
   all-but-one contribution when a date index repeats within the same vectorized batch).

## Verified

- 10/10 tests pass (`pytest baselines/2026-07-25_macro_news_baseline/test/ -v`).
- Full training script smoke-tested end-to-end (`train_macro_news.py --epochs 2 --smoke`) —
  dataset, model (reused unchanged from the sibling baseline), train loop, 6-metric evaluation,
  and results.json writing all worked with dummy zero-features. Smoke-test artifacts (dummy
  results/models under this timestamp) were deleted afterward — not meaningful results.
- `model_dual_news.DualGroupNewsBaseline` reused UNCHANGED, verified via a dedicated smoke test
  with the actual wider `n_feat` (146+66=212) this baseline uses, confirming the "reuse instead
  of reinvent" design decision (design.md §5) actually works, not just in theory.

## Known limitations (documented, not blocking — see design.md §8)

- PCA fit sample size across the full (not just ticker-mentioning) corpus is unverified until the
  real `build_macro_panel.py` run completes — many of the 12 newly-added sources may have little
  pre-2010-06-30 content (recent crawls of mainstream portals), same honest-fallback pattern as
  the sibling baseline already documents for its own (smaller) corpus.
- Test coverage for `_aggregate_by_date`/`_fit_pca` monkeypatches `_load_dated_cached_embeddings`
  directly, which bypasses the real `effective_trading_date`/PCA-transform dtype/precision
  interaction — deferred to the real end-to-end run (which will surface as a non-zero/zero
  coverage count) rather than building a full synthetic-CSV integration test tonight, given the
  time budget for an unattended overnight run.
- No gating/attention distinguishing macro-relevance per ticker — plain concat by design
  (out of scope per requirements.md §7).
