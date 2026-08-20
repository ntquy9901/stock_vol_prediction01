# VN100 vnstock re-crawl + clean + reprocess (persisted)

Date: 2026-08-20. Scope: re-crawl VN100 from vnstock (the 2026-08-16 vnstock crawl was staged but
NEVER committed and was lost from disk — only `data/raw/prices/vn100` = the older Yahoo crawl remained),
clean OHLC defects, reprocess to Parkinson variance, and COMMIT so it is not lost again.

## Why re-crawl
The on-disk VN100 was the Yahoo crawl: 99 rows with price movement (high≠low) but volume==0 (Yahoo's
phantom-holiday / forward-fill glitch), plus NT2 pre-listing backfill. The 2026-08-16 report
(`2026-08-20`… see `2026-08-16_vn100_vnstock_crawl_report.md`) recommended adopting vnstock (VCI) —
correct trading calendar, real reported volume — but that data was never persisted.

## Pipeline run
1. **Crawl** `.venv_vnstock` → `python -m src.data.crawl_vnstock data/raw/prices/vn100 data/raw/prices/vn100_vnstock`
   — 104/104 tickers OK via VCI, 357,852 rows, through 2026-08-14.
2. **Verify raw (pre-clean):** high<low=0; **vol==0-while-moving = 0** (Yahoo had 99 → vnstock has real
   volume, confirmed); 231 open/close-out-of-[low,high] rows across 53 tickers; 13 nonpositive; 2
   all-nonpositive rows.
3. **clean_ohlc** (`python -m src.data.clean_ohlc data/raw/prices/vn100_vnstock`): reconstruct
   high=max / low=min of positive OHLC, clamp nonpositive open/close into [low,high]. 53 files /
   239 cells corrected. After: high<low=0, out-of-range=0.
4. **Drop pre-listing backfill:** 2 all-nonpositive (all-zero OHLC) rows — both NT2 (2010-01-26,
   2010-01-27), asserted LEADING (pre-listing), dropped → cut to real listing date per project rule.
   Final raw: high<low=0, out-of-range=0, all-nonpositive=0.
5. **Reprocess** (`python -m src.common.process_parkinson_pipeline --raw data/raw/prices/vn100_vnstock
   --out data/processed/vn100_vnstock`): 104 files, 357,850 rows. Parkinson NaN=0, inf=0, neg=0.

## vnstock vs Yahoo (on disk now)
| | Yahoo `vn100` | vnstock `vn100_vnstock` |
|---|---|---|
| vol==0 while price moved (glitch) | 99 | **0** |
| high<low (after repair) | 0 | 0 |
| Trading calendar | phantom holiday rows | correct (VCI) |
| NT2 pre-listing backfill | present | trimmed (2 rows dropped) |

## Note
- 11,968 processed rows (~3.3%) have Parkinson=0 — these are high==low days (limit-up/down or
  illiquid single-price sessions), a VN-market feature, not a data defect. NaN/inf/negative = 0.
- Data COMMITTED this time (15 MB raw + 12 MB processed) so it survives (the prior vnstock crawl was
  lost by never being committed).

## Follow-ups (not done here)
- Extend the raw/processed data-quality tests to cover `vn100_vnstock` (currently they only glob the
  VN30 top-level `data/raw/prices/*_ohlcv.csv`, so VN100 is ungated).
- Repoint VN100 code (`scripts/run_vn100_ablation.py` reads `data/processed/vn100` = Yahoo) to
  `data/processed/vn100_vnstock` if adopting vnstock for VN100 results.
