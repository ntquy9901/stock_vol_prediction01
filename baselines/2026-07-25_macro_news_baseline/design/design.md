# Design (Plan) — Market-Wide Macro News Baseline

## 1. Data flow

```
data/external_news_embeddings/raw_cache/news_emb_articles_{source}.parquet  (url, raw_0..767)
   x
crawl_data/data/*.csv (via discover_source_files + load_source, read-only)  (url, pub_date, ...)
   -> join on url (per source) -> (url, pub_date, raw_0..767)
   -> effective_trading_date(pub_date, trading_calendar)               [read-only, phase04_news_helpers]
   -> pool ALL sources together (no ticker filter, no khach_quan/tong_hop split — genuinely
      market-wide: every article that was cached, mentioning a ticker or not)
   -> PCA(32), fit on rows with eff_date < TRAIN_CUTOFF (pooled across ALL sources)
   -> group by eff_date -> mean-pool -> macro_emb_0..31 (one row per trading date, NOT per ticker)
   -> EWMA(30d halflife) over the date-sorted series -> ewma_macro_emb_0..31
   -> data/features/macro_news_panel.parquet: (date, macro_emb_0..31, macro_emb_norm,
      ewma_macro_emb_0..31, ewma_macro_emb_norm)  [~4890 rows x 66 cols]
```

## 2. Why pool ALL sources without a ticker filter

The whole point of this baseline is the articles the dual-group panel EXCLUDES (no ticker
mention). Re-including ticker-mentioning articles too (rather than encoding only the complement)
is deliberate and simpler: a market-wide sentiment signal should reflect ALL market news that
day, including ticker-specific stories (they're still market news) — not just the residual.
This does mean some information overlaps with the existing dual-group panel; that's fine, the
model can learn to weight it via its own fusion layer, and it avoids a fragile
"only-non-ticker" join that would need to exclude exactly the rows the OTHER baseline consumed.

## 3. Reuse map (read-only imports, hard isolation)

| Need | Source | Isolation |
|---|---|---|
| `discover_source_files`, `load_source` | `2026-07-25_dual_group_news_embedding_baseline/code/vendor_data_eda/discover_news.py` | read-only |
| `_article_cache_path`, `RAW_DIM`, `TRAIN_CUTOFF`, `PCA_DIM` | `.../vendor_data_eda/news_embeddings.py` | read-only |
| `effective_trading_date`, `_trading_calendar` | `.../vendor_data_eda/phase04_news_helpers.py` | read-only |
| `_ewma_on_series` | `.../vendor_data_eda/dual_news_features.py` | read-only |
| `load_news_panel` (dual-group per-ticker loader) | `2026-07-25_dual_group_news_embedding_baseline/code/dataset_dual_news.py` | read-only |
| `DualGroupNewsBaseline`, `build_default_model` | `.../code/model_dual_news.py` | read-only, UNCHANGED (n_feat-agnostic already) |

No file in either sibling baseline is modified.

## 4. Dataset (`dataset_macro_news.py`)

Mirrors `MultiStockDatasetWithDualNews` (established pattern in this repo: each new baseline's
dataset is a focused copy+extension of the previous one, not a shared-hook abstraction — matches
`embedding_baseline` -> `dual_news` precedent). Difference: loads a SECOND panel
(`macro_news_panel.parquet`, date-only, no ticker column) and concatenates its (broadcast) vector
onto the existing per-ticker dual-group vector before building `x_news`:

```
x_news[t, s, :] = concat(dual_group_vec(ticker=s, date=t), macro_vec(date=t))
```//same macro_vec for all 30 tickers at a given date.

Missing dual-group row -> zeros (existing convention). Missing macro row -> zeros (same
convention, consistent treatment).

## 5. Model — NO new model file

`DualGroupNewsBaseline(config, n_feat=146+macro_dims, d_news=64)` — literally the same class,
larger `n_feat`. `NewsFeatureLSTM`'s `nn.Linear(n_feat, d_news)` input proj already adapts to
any width. No architecture change needed (Anti-Abstraction Gate: reuse instead of reinventing).

## 6. Train script (`train_macro_news.py`)

Copy of `train_dual_news.py` with `create_dual_news_dataloaders` swapped for
`create_macro_news_dataloaders` (this baseline's own dataset module). Same train/validate loop,
same 6-metric reporting, same EarlyStopping/plotting reuse (read-only from
`src.lstm_gat_hybrid.train_parallel_enhanced`). **10 epochs** (CLAUDE.md training policy default;
user unavailable overnight to approve >10, so capped here regardless of results).

## 7. Gates

- **Simplicity Gate:** no new model architecture, no new fusion mechanism — literally reuses an
  existing baseline's model class with a wider input. Only new code: the macro-panel builder +
  a dataset class that concatenates two existing panels.
- **Anti-Abstraction Gate:** no wrapper/interface layer for "pluggable panels" — a single
  concatenation line in `_create_sequences`, matching how every prior baseline in this repo
  hard-codes its specific panel-loading logic rather than abstracting it.

## 8. Known limitations (documented up front, honest)

- PCA(32) fit on pre-2010-06-30 articles pooled across ALL 52 sources — many of the NEWLY added
  sources (12 from `expand_news_cache_baseline`) are unlikely to have much pre-2010 content
  (recently-crawled mainstream portals typically archive a few years back, not 15+); the actual
  achievved PCA training-sample size will be logged and the honest-fallback (full 768-dim if too
  few samples) applies exactly like the sibling baseline's own documented limitation.
- No gating/attention distinguishing "this macro news is relevant to stock X" — plain concat,
  by design (out of scope per requirements.md §7). If this baseline shows promise, a follow-up
  could add a learned gate (same pattern as `2026-07-18_gated_crossattn_baseline`).
