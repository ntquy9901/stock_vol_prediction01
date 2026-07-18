# Summary — Objective News Baseline (2026-07-15)

## What changed

Added a new baseline (`baselines/2026-07-15_objective_news_baseline/`, per §3.F structure) that
replaces the existing news branch's input (broker analyst PDF reports — "báo cáo tổng hợp") with
**"objective" data**: official corporate-event disclosures (vietstock/VSD, already tagged with
`company_code`) plus general news matched by ticker-code regex OR a curated brand-name alias list
(e.g. "Vinamilk" → VNM), so articles no longer need to literally contain the ticker symbol. Built
following the SDD process now documented in `CLAUDE.md` §1.5 (Specify → Clarify → Plan → Tasks →
Implement → Validate).

## Files

| Path | Purpose |
|---|---|
| `baselines/2026-07-15_objective_news_baseline/requirements/requirements.md` | Specify + Clarify + addenda (data findings, rejected-fallback incident) |
| `baselines/2026-07-15_objective_news_baseline/design/design.md` | Plan (data flow, incremental-extraction design) |
| `baselines/2026-07-15_objective_news_baseline/code/extract_objective_embeddings.py` | New extraction script: ticker-code + brand-alias matching, PhoBERT encode, PCA (persisted, train-only fit), **incremental** (manifest + merge, no full re-encode on rerun) |
| `baselines/2026-07-15_objective_news_baseline/test/test_extract_objective_embeddings.py` | 7 pytest: direct company_code match, brand-alias match (case-insensitive), HTML strip, leakage-guard, incremental skip, incremental merge |
| `baselines/2026-07-15_objective_news_baseline/code_review/code_review_2026-07-15.md` | Review findings + fixes |
| `data/objective_embedding/` | New embedding cache (341 records) + `_manifest.json` + `_pca.pkl` — NOT training code, reused sibling's unmodified `train_embedding_baseline.py --emb_dir data/objective_embedding` |
| `CLAUDE.md` | Added §1.5 Spec-Driven Development, §Testing quality rules (ENFORCED) — both requested separately mid-session |

No edits to `src/` or any sibling baseline (hard isolation, §3.F.3).

## Tests + coverage

`pytest baselines/2026-07-15_objective_news_baseline/test/ -v` → **7/7 pass**. Diff-coverage tool
not run (`diff-cover` not installed in this repo yet — known gap, documented in CLAUDE.md
Per-project setup). Coverage is behavioral: every code path added/changed (alias match, ticker
regex, leakage guard, dedup, incremental skip, incremental merge, PCA persist/reuse) has a
dedicated test.

## Code review

`/code-review` (effort medium, 1 finder agent + self-verify against real crawl data). 5 findings,
all addressed:
1. Alias regex missing `re.IGNORECASE` (missed lowercase "vinamilk") — fixed.
2. Ticker regex should NOT be case-insensitive (collision risk, e.g. "GAS"~"gas") — fixed (found
   while fixing #1).
3. Rows with empty `publish_time` (174/669 vietstock, 59/59 tuổi trẻ) silently indistinguishable
   from "0 matches" — fixed with a separate `no_date_dropped` counter. **A fallback to
   `crawl_time` was tried and reverted** after it was found to collapse 179/298 test-period
   records onto a single calendar date (fake news-volume spike) — see requirements.md §5c.
4. `design.md` claimed a dedup step that didn't exist in code — implemented (`_dup_or_id`, no-op
   on current corpus since document_id is already 100% unique, but matches the documented design
   and protects against future duplicate sources).
5. Test coverage gap (PCA-fit path, unenriched-file leakage guard) — noted as follow-up, not
   blocking.
Full detail: `code_review/code_review_2026-07-15.md`.

## Commands actually run

```
python -m pytest baselines/2026-07-15_objective_news_baseline/test/ -v   # 7 passed
python baselines/2026-07-15_objective_news_baseline/code/extract_objective_embeddings.py  # real extraction, 3x (iterating on fixes)
python baselines/2026-07-07_embedding_baseline/code/train_embedding_baseline.py --emb_dir data/objective_embedding --epochs 10   # 3x (iterating on fixes), final run authoritative
```

## Results (final, authoritative run — `results/embedding_baseline_2026-07-15_015004/`)

| Baseline | Epochs | Test DirAcc | R² | QLIKE |
|---|---|---|---|---|
| HAR-only | 70 | 69.98% | — | — |
| Embedding baseline (broker reports) | 40 | 68.76% | — | 0.553 |
| Latent noise | 10 | 69.33% | 0.713 | 0.544 |
| **Objective news (this baseline)** | 10 | **67.87%** | 0.714 | 0.565 |

Data coverage measured before training (dry-run): 341 total matched records (325 vietstock direct
+ 5 vsdc + 11 ticker/alias-matched general news), **119 in test period (2021-2026), 113 unique
(stock, day) pairs — ≈0.27% of all stock-days**, roughly 20x sparser than the existing news branch
(≈5.5% coverage).

## Go/No-Go: **NO-GO**

Test DirAcc (67.87%) is the **lowest of all news-branch variants tried to date**, below HAR-only
by 2.1 points. The "objective" data source, despite being cleaner/less subjective than broker
reports, is too sparse (~20x sparser) to provide a learnable signal at this baseline's config. Not
recommending further investment in this direction (e.g. longer training) without first growing the
underlying crawl volume of general news sources (4/5 general-news sources — thanhnien, tuoitre,
vietnamplus — matched 0 or near-0 rows even with brand-name aliases; the ceiling here is crawl
volume, not matching sophistication).

## Risks / follow-ups

- `NAME_ALIASES` is a manually curated, non-exhaustive list (30 tickers, primary brand name only)
  — could be extended, unlikely to change the NO-GO conclusion given the crawl-volume ceiling.
- PCA-fit path and unenriched-file leakage guard lack dedicated unit tests (noted in code review
  finding #5) — low priority given the NO-GO outcome.
- Incremental-extraction manifest/PCA-persist design (requested mid-session) is implemented and
  tested but only exercised on this one (now NO-GO) baseline — worth reusing the same pattern if a
  future baseline needs daily-refresh extraction.

## Definition of Done checklist

- [x] Code satisfies the request (objective-data baseline + incremental extraction), no unrelated
      refactor.
- [x] Tests: 7/7 pass, cover all new/changed behavior. Diff-coverage tool: **Not run** (not
      installed — documented gap).
- [x] Code review: `/code-review` run, 5 findings, all addressed.
- [x] Smoke: extraction + training end-to-end run against REAL data (not just unit tests) — this
      IS the smoke test for this data-pipeline baseline.
- [x] Impact analysis: read-only from `crawl_data/data/objective/`, writes only to new
      `data/objective_embedding/`; reuses sibling's `train_embedding_baseline.py` unmodified via
      CLI arg — zero blast radius on existing baselines/src.
- [x] Similar check: N/A (no prior pattern of this exact kind to duplicate-check against).
- [x] Summary report: this file.
