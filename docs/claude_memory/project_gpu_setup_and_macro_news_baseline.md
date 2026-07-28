---
name: project-gpu-setup-and-macro-news-baseline
description: 2026-07-25/26 overnight session — Python 3.14 has no CUDA PyTorch wheel (use the 3.10 venv instead), full-corpus (7.5M article) PhoBERT encode, and a market-wide macro news baseline that did not beat existing results
metadata:
  node_type: memory
  type: project
  originSessionId: 7b3b1f97-cfdd-4b28-b9f4-b53d0110952d
  modified: 2026-07-25T19:14:44.208Z
---

**GPU blocker + fix (reusable fact):** this machine's main Python is 3.14, which has **no CUDA
PyTorch wheel** (stable or nightly, cu124/cu128 both — verified `pip install torch --index-url
.../cu124` returns "no matching distribution" for cp314). The GPU itself works fine (RTX 4060
Laptop, driver supports CUDA 12.7, confirmed via `nvidia-smi`) — it's purely a wheel-availability
gap for this Python version. **Fix:** a Python 3.10 venv at
`C:\luanvan\stock_vol_prediction01\.venv_gpu_encode\` already exists with `torch==2.6.0+cu124` +
transformers/sentencepiece/pandas/pyarrow/scikit-learn installed and GPU-verified working — reuse
this venv (`.venv_gpu_encode\Scripts\python.exe`) for any future GPU-needing task in this project
rather than re-discovering the blocker or reinstalling. Python 3.10 was already present at
`C:\Users\QUY\AppData\Local\Programs\Python\Python310\python.exe` (found via `py -0p`).

**PhoBERT GPU throughput (this RTX 4060):** plateaus around batch_size=256-512 at ~1050
articles/sec (vs. ~25/sec on CPU, ~40x speedup). Reload overhead per `extract_phobert_embeddings`
call (~3-7s, since that vendored function reloads the model from HF cache every call, not just
once) matters a lot at small batch/chunk sizes — use a large `chunk_size` (~100,000) when calling
`build_incremental_cache.py`'s `run_source`, not the ~5000 default meant for the original
CPU-only, small (13,818-article) ticker-only job.

**Full-corpus expansion done:** `2026-07-25_expand_news_cache_baseline/code/build_incremental_cache.py`
now supports `--include_all` (encode every article, not just ticker-mentioning ones — new
`_all_articles` path) with chunked, crash-resilient writes. Ran for real: **7,494,266 articles
encoded in ~3h on GPU**; `data/external_news_embeddings/raw_cache/` is now **34GB, 60 source
files** (was 48, ticker-only). Backups of the pre-expansion cache exist at
`raw_cache_backup_2026-07-25/` and `raw_cache_backup_2026-07-25_pre_include_all/` if a rollback
is ever needed.

**New baseline result — market-wide macro news (`2026-07-25_macro_news_baseline`):** hypothesis
was that non-ticker-mentioning ("macro") articles carry a signal useful for ALL 30 tickers
(broadcast a daily date-only aggregated embedding to every ticker, concatenated onto the existing
146-col dual-group per-ticker vector → 212-col input, model reused UNCHANGED from
`model_dual_news.DualGroupNewsBaseline`, which was already n_feat-agnostic). **Result: test
DirAcc=68.63%, R²=0.710, QLIKE=0.563 — essentially a TIE with the existing ticker-only dual-group
baseline (68.50% DirAcc) and still below HAR-only (69.98%).** Macro panel itself built cleanly
(4890 dates, 100% coverage, PCA fit on 56,685 real pre-cutoff rows, no fallback needed) — the
null result is about the FEATURE'S USEFULNESS to the model, not a data/pipeline problem.

**Why this matters:** this is yet another (now well-established, see
`[[project_selective_news_gate_finding]]`) instance of a news-related idea that builds cleanly and
runs without bugs, but doesn't move the needle on the actual prediction task. Combined with the
per-ticker gating findings, the accumulating picture is that this project's various attempts to
extract MORE signal from news (broader corpus, macro aggregation, learned gates, per-ticker
selection) keep landing near the HAR-only baseline rather than clearly beating it.

**How to apply:** if asked to try yet another news-fusion variant, consider surfacing this pattern
explicitly to the user before building — e.g. ask whether a fundamentally different angle (a
learned gate ON the macro feature specifically, like `gated_crossattn`'s pattern, or restricting
macro aggregation to finance/economy-keyword-filtered articles only instead of literally
everything including sports/entertainment from mainstream portals) is worth trying before another
plain-concat variant. Full writeup: `docs/reports/2026-07-26_0230_summaryOfUpdate_report.md`.
