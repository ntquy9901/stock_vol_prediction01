# Code Review — Market Fallback Baseline (2026-07-08)

**Reviewer:** code-review skill (high effort) — 4 finder agents (8 angles) + dedup/verify
**Scope:** `baselines/2026-07-08_market_fallback/code/*.py` + `test/*.py`
**Context:** baseline derive từ embedding baseline (đã review+fix), focus vào phần MỚI (market branch, gate, 7-tuple, extract_market)

## Summary
- **~24 raw candidates → 10 survived** (dedup + verify).
- Severity: **2 HIGH, 6 MEDIUM, 2 LOW**.
- 2 HIGH phải fix trước "done": market branch silent-dead (no coverage assert), emb_dim mismatch crash.

---

## Findings (most-severe first)

### [HIGH-1] Market branch không có coverage assert → silent dead branch
**File:** `code/dataset_embedding.py:197-202` · **Verdict:** CONFIRMED
**Vấn đề:** `_create_sequences` assert `if any_cache and self._matched_cells == 0: raise` — nhưng `_matched_cells` chỉ đếm **per-stock** (ticker). KHÔNG có check cho **market branch**. Nếu `market_emb.npz` tồn tại nhưng date keys không match `window_dates` (format mismatch, timezone...) → `x_market` all-zero, `market_mask` all-zero mọi window → `MarketBranch` luôn emit `no_news_token` → **toàn bộ innovation "market fallback" silently inert**, metric report nhầm là market-fallback mà thực ra chỉ HAR+per-stock.
**Fix:** Thêm `_market_total`/`_market_matched` counter trong market loop + assert nếu `self._market` non-empty nhưng 0 match (mirror HIGH-2 fix của sibling).

### [HIGH-2] emb_dim mismatch → crashMarketBranch nếu per-stock dim ≠ market dim
**File:** `code/dataset_embedding.py:120-130` (detect) + `code/model_embedding.py` · **Verdict:** CONFIRMED
**Vấn đề:** `_emb_dim` detect từ per-stock cache trước; market dim chỉ dùng nếu per-stock None. Nếu per-stock extract dim=64 (PCA) nhưng market extract `--no_pca --dim 768` → `_emb_dim`=64, model `emb_dim=64`, `MarketBranch.pool.proj=Linear(64,d_news)` nhận market vector 768-d → **RuntimeError shapes cannot be multiplied** ở forward. Không có validation.
**Fix:** Assert market array dim == `_emb_dim` khi load; fail loud với message rõ.

### [MED-3] Market PCA leakage: cutoff theo lịch, cache share across splits
**File:** `code/extract_market_embeddings.py:109` + `code/dataset_embedding.py:263` · **Verdict:** PLAUSIBLE (weak, inherited)
**Vấn đề:** PCA fit trên `date < 2020-01-01` (calendar) nhưng split thật theo row-index 0.7 → cutoff chỉ xấp xỉ. Market cache build 1 lần từ ALL articles, share cho train/val/test. Weak leakage (PCA unsupervised, raw embedding frozen thì benign).
**Fix:** Document honest (calendar cutoff ≈ row-index split, đã note trong extract). Full fix = PCA-in-dataset post-split (defer — refactor lớn). Inherited từ sibling MED-4.

### [MED-4] train_cutoff compare bằng string → hỏng nếu date không zero-pad
**File:** `code/extract_market_embeddings.py:109` · **Verdict:** PLAUSIBLE
**Vấn đề:** `r["date"] < args.train_cutoff` là string compare. Đúng CHỈ khi date `YYYY-MM-DD` zero-padded. Nếu có `"2020-1-5"` (không pad) → compare sai (`"2020-1-5" < "2020-01-01"` = True vì '-' < '0') → article test-period bị tính là train → PCA fit set sai silently.
**Fix:** Assert format `YYYY-MM-DD` (len 10, s[4]=='-', s[7]=='-') trong `_norm_date`/extract; fail loud cho date không pad. (Aggregation script đã zero-pad nên hiện safe, nhưng thêm guard.)

### [MED-5] MAX_MARKET=15 magic constant — "percentile 99" không có bằng chứng
**File:** `code/dataset_embedding.py:26`, `_pad_articles:89` · **Verdict:** PLAUSIBLE (altitude)
**Vấn đề:** Cap 15 market articles/day "keep first 15" theo CSV row order (không recency/relevance). Design doc claim "percentile 99" nhưng KHÔNG có code compute percentile. Trùng anti-pattern MED-10 của sibling. Ngày đông tin vĩ mô (>15) → bỏ bài đúng lúc market signal mạnh nhất.
**Fix:** Log khi truncation fire (count days `len > MAX_M`); compute MAX_M từ `np.percentile(article_counts, 99)` hoặc document honest trade-off.

### [MED-6] `--epochs` default 40 vs CLAUDE.md "70 epochs" rule
**File:** `code/train_market_fallback.py:110` · **Verdict:** CONFIRMED (convention)
**Vấn đề:** CLAUDE.md "Standardized Hyperparameters": `num_epochs = 70` apply ALL files. Default 40 không có waiver documented. So sánh matched-epoch (success #5) bị apples-to-oranges nếu các baseline dùng epoch khác nhau.
**Fix:** Default `--epochs 70` (rule compliance) + note matched-epoch comparison dùng `--epochs` explicit.

### [MED-7] x_market ~220MB redundant RAM (store per-window)
**File:** `code/dataset_embedding.py:182-189` · **Verdict:** PLAUSIBLE (efficiency)
**Vấn đề:** `x_market[seq=22, MAX_M=15, dim]` materialize per-window × ~864 windows × 3 splits, dù market vector 1 ngày identical across stocks + overlapping windows. ~220MB duplicate (unique content ~5MB).
**Fix (defer):** Store market per-day dict, index trong `__getitem__` theo `window_dates`. Cuts ~95% RAM. Note: efficiency, không correctness — defer.

### [MED-8] PCA crash khi train-period articles < 2
**File:** `code/extract_market_embeddings.py:112-117` · **Verdict:** PLAUSIBLE (edge robustness)
**Vấn đề:** Nếu `n_train < dim`, `args.dim = max(1, n_train-1)`. Khi `n_train==1` → `dim=1` → `PCA(n_components=1).fit(1 sample)` → sklearn `ValueError`. Crash mid-extraction thay vì degrade graceful.
**Fix:** Guard `if n_train < 2: skip PCA (use raw 768)` hoặc raise clear error.

### [LOW-9] Market cache loaded 3× (train/val/test mỗi dataset load lại npz)
**File:** `code/dataset_embedding.py:76-87, 263-265` · **Verdict:** PLAUSIBLE (efficiency)
**Vấn đề:** Cùng `market_emb.npz` load+parse+normalize 3 lần (mỗi split 1 dataset instance). 3× I/O cho object identical.
**Fix (defer):** Memo module-level hoặc load 1 lần trong `create_market_dataloaders` truyền dict in.

### [LOW-10] Heavy duplication từ sibling (~300 lines)
**File:** extract/dataset/model/train vs `baselines/2026-07-07_embedding_baseline/` · **Verdict:** PLAUSIBLE (cleanup)
**Vấn đề:** 4 file pair ~80-85% verbatim copy sibling. `ArticleSetAttentionPooling`, `NewsTemporalEncoder`, `_pad_articles`, encode loop, dataloader builder đều duplicate.
**Fix (defer):** Rule §3.F isolation justify self-contained, NHƯNG có thể subclass sibling dataset + import model classes (read-only). Note: trade-off isolation vs DRY, chấp nhận cho MVP.

## Clean (verified không bug)
- 7-tuple consistent end-to-end (dataset → collate → unpack → forward): count/order/shape đúng.
- `GatedNewsFusion` broadcast đúng: `market_daily.unsqueeze(2).expand_as(stock_daily)` + `has_news[B,T,S,1]`.
- `ArticleSetAttentionPooling` đúng cho cả per-stock `[B,T,S,MAX_A,D]` lẫn market `[B,T,MAX_M,D]` (last dim = article axis).
- Gate deterministic `g = has_news` — right-depth MVP (0 params, no collapse risk, design defer learned gate).
- Reuse `ArticleSetAttentionPooling` cho market — fit abstraction (generic leading dims).
- target clipping, per-stock coverage assert, graph builder init, train-only normalizer fit — preserved từ sibling.
- §3.B 6 metrics + Val/Test table, §3.C learning curves, §3.E weight_decay/dropout/early-stop/grad-clip — compliant.

## ✅ Remediation (2026-07-08) — DONE + re-verified
- **pytest: 8 passed** (no regression). **Smoke `--epochs 2`:** chạy sạch, market coverage log works.
| # | Fix | Verify |
|---|---|---|
| HIGH-1 | `_market_total/matched` counter + assert market coverage>0 (dataset:197-216) | ✅ smoke log "market date-match coverage" |
| HIGH-2 | dim guard trong `_pad_articles` — raise nếu cache dim ≠ _emb_dim (dataset:99-104) | code (smoke arr=None không trigger) |
| MED-4 | `DATE_RE` validate YYYY-MM-DD, skip malformed (extract:62-72) | code |
| MED-5 | `_market_truncated` counter + log khi truncation fire (dataset:182-192) | ✅ smoke log mechanism |
| MED-6 | `--epochs` default 70 (CLAUDE.md standardized) + help note (train:110) | ✅ smoke chạy (dùng --epochs explicit) |
| MED-8 | PCA skip nếu n_train<2 thay vì crash (extract:115-126) | code |
| MED-3 | Documented: calendar cutoff ≈ row-index split, weak/inherited leakage (full fix = PCA-in-dataset, defer) | doc |
| MED-7 | Documented: x_market ~220MB redundant RAM — defer (efficiency, không correctness) | doc |
| LOW-9 | Documented: market cache loaded 3× — defer (efficiency) | doc |
| LOW-10 | Documented: duplication từ sibling — accepted (§3.F isolation trade-off) | doc |

## Re-verify (sau fix) — DONE 2026-07-08
- [x] Fix 2 HIGH + 4 MED (HIGH-1, HIGH-2, MED-4, MED-5, MED-6, MED-8)
- [x] `pytest test/` → 8 passed
- [x] `--smoke --epochs 2` → chạy sạch, market coverage log đúng, Val/Test table OK
- [ ] (chưa làm) Extract market cache thật + train thật → go/no-go
