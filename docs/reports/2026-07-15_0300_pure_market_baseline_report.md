# Summary — Pure Market-Vector Baseline (2026-07-15)

## What changed

Added `baselines/2026-07-15_pure_market_baseline/`: tests the hypothesis "aggregate ALL news
each day into one market-wide vector, broadcast identically to every stock (no ticker matching,
no gating)" — user's own framing: "mỗi ngày, tin tức lấy hết xem như là các vectors của thị
trường, không phân biệt vectors nào của cổ phiếu nào."

Reuses (read-only) `data/sentiment_embedding/market_emb.npz` — already extracted by the sibling
`2026-07-08_market_fallback` baseline (PhoBERT on ALL of `unified_articles.csv`, no ticker
filter, PCA 768→64, train-only fit) — **no new extraction script needed**. New: a dataset that
drops the per-stock news branch entirely, and a model that broadcasts one market vector/day to
all 32 stocks (vs `market_fallback`'s GATED version, which still keeps a per-stock branch and
only substitutes market news when a stock has none).

## Files

| Path | Purpose |
|---|---|
| `requirements/requirements.md`, `design/design.md` | Specify + Plan (SDD, CLAUDE.md §1.5) |
| `code/dataset_pure_market.py` | `PureMarketDataset` — 5-tuple (x_har, adj, x_market, market_mask, y), market loaded by date only |
| `code/model_pure_market.py` | `PureMarketBaseline` — HAR branch + market branch (reuses `ArticleSetAttentionPooling`/`NewsTemporalEncoder` from sibling via stocks-dim=1 adapter) → broadcast → fusion |
| `code/train_pure_market.py` | Training loop (mirrors sibling structure, 5-tuple/4-arg forward) |
| `test/test_smoke.py` | 3 pytest: forward+backward, zero-market-day, market-contribution-identical-across-stocks (via forward hook on real `forward()`) |
| `code_review/code_review_2026-07-15.md` | 2 findings, both fixed |

No edits to `src/`, `2026-07-07_embedding_baseline/`, or `2026-07-08_market_fallback/`.

## Tests + code review

3/3 pytest pass. `/code-review` (1 agent): 2 findings — (1) test was checking a hand-duplicated
broadcast calculation instead of the real `forward()` path, fixed via `register_forward_pre_hook`
on `model.fusion` to inspect the actual fusion input; (2) `_pad_articles` was missing the
sibling's dim-mismatch guard, fixed with the same clear error message.

## Commands run

```
pytest baselines/2026-07-15_pure_market_baseline/test/ -v          # 3 passed
python .../train_pure_market.py --epochs 1 --smoke                 # smoke, dummy market, exit 0
python .../train_pure_market.py --epochs 10                        # real run, exit 0
```

## Results (`results/pure_market_2026-07-15_025124/results.json`)

Coverage (measured, per-day match — much denser metric than the per-cell one used in the
objective-news deep-dive, since here EVERY day in the window either has market news or not,
no per-stock breakdown):
```
train: 5127/19008 days (26.97%)
val:   3203/3608  days (88.77%)
test:  2553/3608  days (70.76%)
```

| Baseline | Epochs | Test DirAcc | R² | QLIKE |
|---|---|---|---|---|
| HAR-only | 70 | 69.98% | — | — |
| Latent noise | 10 | 69.33% | 0.713 | 0.544 |
| **Pure market (this)** | 10 | **68.95%** | 0.713 | 0.556 |
| Embedding baseline | 40 | 68.76% | — | 0.553 |
| Objective news | 10 | 67.87% | 0.714 | 0.565 |

## Go/No-Go: marginal — not conclusive, similar verdict to latent-noise

Beats both ticker-matched news variants (embedding-baseline 68.76%, objective-news 67.87%)
despite ~13-280x higher coverage, but still below latent-noise (69.33%) and HAR-only (69.98%).
**Key finding: coverage alone doesn't translate to proportional gains** — a market-wide, non-
stock-specific signal is diluted (doesn't tell the model WHICH stock is affected), so higher
volume doesn't compensate for lower specificity. Not epoch-matched vs HAR-only (10 vs 70) or vs
embedding-baseline (10 vs 40), so this is a lean/inconclusive signal like latent-noise — would
need a matched-epoch control + longer training to confirm before further investment.

## Follow-up: resumed +10 epoch (total 20, 2026-07-15 late)

`train_pure_market.py` gained a `--resume_from` flag (mirrors the sibling latent-noise
baseline's pattern) and was resumed from the 10-epoch checkpoint for 10 more epochs
(`results/pure_market_2026-07-15_182133/`). **Result: plateaued, no improvement** — test DirAcc
68.92% (vs 68.95% at 10 epoch, within noise), val DirAcc oscillating 68.9-70.6% with no upward
trend across the extra 10 epochs. Train loss still decreasing slightly (0.912→0.897) but val
loss not improving — early signs of the model having converged for this architecture/data
combination. **Recommendation: do not train further** without changing the approach (e.g. the
matched-epoch sweep suggested above, or a different fusion mechanism for the market signal).

## Risks / follow-ups

- No epoch-matched comparison yet across all 5 variants — recommend a controlled sweep (same
  epoch count, e.g. 20) across HAR-only / embedding / latent-noise / pure-market / objective-news
  before drawing a final conclusion on which news-signal design is best.
- Could combine ideas: broadcast market vector (this baseline) + per-stock ticker match when
  available (this is literally what `2026-07-08_market_fallback`'s GATE already does — its real
  result, 68.69% test @ 37 epoch, is available for comparison already).

## Definition of Done checklist

- [x] Code satisfies request, no unrelated refactor.
- [x] Tests: 3/3 pass, cover new/changed behavior. Diff-coverage: Not run (tool not installed,
      documented gap).
- [x] Code review: `/code-review` run, 2 findings, both fixed.
- [x] Smoke: `--smoke` CLI run end-to-end (exit 0) + real 10-epoch run (exit 0) — this IS the
      data-pipeline smoke test.
- [x] Impact analysis: read-only reuse of `market_emb.npz` + `model_embedding.py`'s pooling
      modules; zero edits to any sibling baseline or `src/`.
- [x] Summary report: this file.
