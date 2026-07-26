# Summary — GPU Full-Corpus News Encoding + Market-Wide Macro News Baseline (2026-07-25/26, overnight)

User authorized this session to run unattended overnight: "sau khi benchmark, cứ run luôn 9M
không cần tôi confirm. sau đó tạo baseline thử tiếp với embedding mới này." Everything below ran
without further check-ins, per that instruction.

## Part 1 — GPU setup + full-corpus encoding (`2026-07-25_expand_news_cache_baseline`)

**Problem:** installed PyTorch was CPU-only (`2.12.1+cpu`); the physical GPU (RTX 4060 Laptop,
CUDA 12.7 driver) was unusable from it. Root cause: **Python 3.14 has no official CUDA PyTorch
wheel yet** (stable or nightly, cu124/cu128 — verified, both return "no matching distribution").

**Fix:** created a separate Python 3.10 venv (`.venv_gpu_encode/`, found Python 3.10 already
installed on the machine) and installed `torch==2.6.0+cu124` there — GPU confirmed working
(`cuda available: True`, RTX 4060 Laptop GPU detected).

**Extended `build_incremental_cache.py`** (code review: 5 patches applied, see that baseline's
`code_review/` for the earlier round) with:
- `--include_all` flag: encodes every article with text, not just ticker-mentioning ones (the
  market-wide use case) — new `_all_articles`/`_clean_articles` split, tests added.
- Chunked processing (`CHUNK_SIZE`/`--chunk_size`): bounds memory and limits crash-loss to one
  chunk for huge sources (up to ~1.2M articles/source under `--include_all`) — tests added
  including a simulated-crash test confirming earlier chunks survive.

**Benchmarked GPU throughput** before committing to the full run: batch_size 64→197/s,
128→974/s, 256→1045/s, 512→1052/s (plateau). Picked `batch_size=256, chunk_size=100000`.

**Real run:** dry-run confirmed 7,494,266 articles to encode (less than the raw ~8.98M total due
to per-source URL dedup + empty-text filtering already applied). **Full run: 7,494,266 articles
encoded in 10,875s (~3.02h)** on the RTX 4060. Backed up `raw_cache/` before running (both before
the ticker-only expansion and again before `--include_all`, ~4.4GB and full-cache snapshots kept
on disk). Final cache: **34GB, 60 per-source parquet files** (was 48 before this session).

## Part 2 — New baseline: `2026-07-25_macro_news_baseline`

**Hypothesis:** market-wide/macro news (articles that don't mention any specific VN30 ticker —
monetary policy, general market commentary, sector-wide stories) carries a signal useful for
ALL 30 tickers, not a per-ticker signal — so aggregate by DATE ONLY (not ticker), broadcast the
same daily vector to every ticker, and concatenate it onto the existing per-ticker dual-group
news feature vector.

**New code** (5-subfolder structure, hard-isolated — no sibling baseline file modified):
- `build_macro_panel.py`: two-pass streaming aggregation (Pass 1 fits PCA(32) on a
  proportionally-sampled pre-`TRAIN_CUTOFF` pool across all sources — fixed an alphabetical-bias
  bug found in self-review before the real run; Pass 2 re-reads each source, transforms via the
  fitted PCA, vectorized `np.add.at` scatter-accumulates per trading date) → EWMA(30d) smoothed.
  **Real result: 4890 trading dates, 100% coverage** (every date has ≥1 article — vs. 81.49% for
  the ticker-only dual-group panel), PCA fit on 56,685 real pre-cutoff rows (not fallback).
- `dataset_macro_news.py`: extends the sibling's per-ticker dual-group panel loader with a new
  date-only macro loader; `x_news[t, s, :] = concat(dual_group_vec(ticker=s, date=t),
  macro_vec(date=t))` — same macro vector broadcast across all 30 tickers.
- `train_macro_news.py`: **reuses `DualGroupNewsBaseline` UNCHANGED** (read-only import) — the
  model was already n_feat-agnostic, so no new architecture was needed, just a wider input
  (146 dual-group + 66 macro = 212 dims).
- 10 tests (dataset shapes, macro-broadcast-is-identical-across-tickers, panel-aggregation math,
  model-with-wide-nfeat smoke) — all pass. Self-review found + fixed 2 real issues before the
  real run (alphabetical PCA-sampling bias; slow per-row Python loop → vectorized).

**Training policy:** capped at 10 epochs (CLAUDE.md default; user unavailable overnight to
approve more) — enforced in code (`train_macro_news.py` raises if `--epochs > 10`).

### Result

| Metric | Val | Test |
|---|---|---|
| DirAcc | 69.33% | **68.63%** |
| R² | 0.660 | **0.710** |
| QLIKE | 0.701 | **0.563** |
| RMSE | 0.002449 | 0.002662 |

**Compared to existing results:**
- HAR-only (best DirAcc reference): 69.98% — macro baseline still below.
- Dual-group (ticker-only news, no macro): 68.50% test DirAcc — macro baseline is +0.13pp, i.e.
  **essentially a tie**, not a meaningful win.
- Gated cross-attn (best R²/QLIKE record so far): R²=0.7157, QLIKE=0.557, DirAcc=68.97% — macro
  baseline's R²=0.710/QLIKE=0.563 are close but slightly behind that record.

**Verdict: no clear improvement.** Adding a broadcast market-wide macro feature does not move
the needle over the existing ticker-only dual-group baseline — consistent with this project's
broader, repeated finding (documented in memory) that no news-fusion variant tried so far clearly
beats HAR-only. Learning curves show healthy convergence (no runaway overfitting — val loss
decreases/plateaus, doesn't blow up), so this is a genuine null result, not a training-failure
artifact.

## Files

- `baselines/2026-07-25_expand_news_cache_baseline/` — `--include_all` extension, chunking,
  code review, tests (updated from earlier this session).
- `baselines/2026-07-25_macro_news_baseline/` — new baseline, full 5-subfolder structure.
- `data/features/macro_news_panel.parquet` — new (4890 rows × 67 cols).
- `results/macro_news_2026-07-26_020954/` — results.json + learning curve plots.
- `models/macro_news_2026-07-26_020954/best.pt` — checkpoint.
- `.venv_gpu_encode/` — Python 3.10 + CUDA venv, kept in case more GPU encoding is needed later.

## Tests + code review

- `pytest` across all 4 touched/created baselines tonight: **38/38 pass.**
- Adversarial self-review (no `/code-review` checkpoint-wait, per user's explicit "don't need my
  confirmation" instruction) on both new baselines — findings + fixes documented in each
  baseline's `code_review/` folder.
- Diff-coverage: **Not run** (tooling gap, already documented project-wide in CLAUDE.md).

## Risks / follow-ups for you to review

1. **The macro baseline result is a null result** — didn't beat existing baselines. Worth
   deciding whether to (a) drop this direction, (b) try a learned gate on the macro feature
   (like `gated_crossattn`'s pattern) instead of plain concat, or (c) try restricting the macro
   aggregation to finance/economy-relevant articles only (keyword filter) rather than literally
   everything (sports/entertainment content from mainstream portals may just be diluting the
   signal with noise).
2. **Cache backups left on disk**: `data/external_news_embeddings/raw_cache_backup_2026-07-25/`
   (4.4GB) and `raw_cache_backup_2026-07-25_pre_include_all/` (4.4GB) — safe to delete once
   you're satisfied the expanded cache is good, or keep for rollback.
3. **`.venv_gpu_encode/`** left in place (~3GB) for any future GPU encoding needs — delete if not
   wanted.
4. Per my earlier report: the sibling dual-group baseline's panel was rebuilt with the 12
   newly-classified sources — if you want that reflected in a fresh dual-group training run
   (separate from tonight's macro baseline), that's a quick re-run, not done tonight (out of
   scope of what was asked).

## DoD checklist

- [x] Code satisfies the request
- [x] Tests written + run (38/38 pass across all touched baselines)
- [ ] diff-cover C0/C1 — Not run (documented tooling gap)
- [x] Adversarial self-review — findings fixed, documented per-baseline
- [x] Real-data smoke validated at each stage (dry-runs, small-subset runs before full runs)
- [x] Impact analysis — shared cache backed up before every mutating step
- [x] Summary report — this file
- [x] Training policy — capped at 10 epochs, enforced in code
