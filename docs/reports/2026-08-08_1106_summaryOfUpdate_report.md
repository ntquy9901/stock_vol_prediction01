# Compact pooled manifest update

## Change

- Replaced the trainer's array-dumping `sample_manifest.json` with a version-2 compact artifact.
- The artifact contains per-split ordered sample keys, counts, canonical hashes for price, news,
  mask, raw targets, and model targets, plus the preprocessing hash.
- Checkpoint resume validation uses the compact artifact's full tensor-derived hashes; it does not
  use IDs-only identity.

## Files

- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/train.py` — compact artifact and
  tensor-content resume identity.
- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_train_smoke.py` — artifact
  size/payload regression coverage.
- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code_review/code_review_2026-08-08.md`
  — compact-manifest adversarial review record.

## Verification

- TDD red: compact-artifact regression failed against the previous 11,664-byte two-sample tensor
  dump.
- `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test -q` — 76 passed.
- `ruff check baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/train.py baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_train_smoke.py` — passed.
- `git diff --check` — passed.
- Diff coverage — Not run; `diff-cover` is not configured in this repository.

## Review and impact

The review found no critical or major remaining issue. The change is confined to Task 6 trainer
artifacts and their regression test. P0-P3 manifest/order gates and price-only/P2/P3 behavior
were exercised by the baseline suite without modification.
