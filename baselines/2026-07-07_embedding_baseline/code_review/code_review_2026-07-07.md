# Code Review — Embedding Baseline (2026-07-07)

**Reviewer:** code-review skill (high effort, recall-biased) — 4 finder agents (8 angles) + dedup/verify
**Scope:** `baselines/2026-07-07_embedding_baseline/code/*.py` + `test/*.py`
**Method:** per CLAUDE.md mục 5 (adversarial, cynical, ≥10 issues, fix HIGH/MEDIUM)

## Summary
- **24 raw candidates → 10 survived** (dedup + verify).
- Severity: **3 HIGH, 7 MEDIUM**. 0 REFUTED (several weak ones dropped for severity cap).

## ✅ Remediation (2026-07-07) — ALL 10 FIXED + re-verified
- **pytest: 6 passed** (no regression). **Smoke `--epochs 2`:** chạy sạch, Val/Test table (§3.B) hiện đúng.
| # | Fix | Verify |
|---|---|---|
| HIGH-1 | PCA: reduce dim thay vì widen sang all (extract:107) | code-only (smoke không chạy PCA) |
| HIGH-2 | `_norm_date` cả 2 bên + assert coverage>0 khi có cache (dataset) | ✅ smoke log "date-match coverage" |
| HIGH-3 | `weight_decay=1e-5` + arg (train:136) | ✅ smoke chạy |
| MED-4 | `--train_cutoff` configurable + doc (calendar ≈ row-index split) | code+doc |
| MED-5 | ticker regex `re.IGNORECASE` (extract:39) | code-only |
| MED-6 | assert finite sau PhoBERT + `_pad_articles` skip non-finite + `allow_nan=False` JSON | ✅ smoke JSON hợp lệ |
| MED-7 | docstring note (1-article day → query no-grad, accepted trade-off) | doc |
| MED-8 | `requires_grad_(False)` trên `har.fusion` (~23K params frozen) | ✅ smoke backward OK |
| MED-9 | Val/Test comparison table + JSON keys `validation_metrics`/`val_test_diff` | ✅ smoke in bảng |
| MED-10 | `MAX_ARTICLES` 5→10 + doc trade-off | ✅ smoke |


---

## Findings (most-severe first)

### [HIGH-1] PCA fallback silently leaks val/test into the projection
**File:** `code/extract_embeddings.py:107-114` · **Verdict:** CONFIRMED
**Vấn đề:** Khi `train_mask.sum() < dim` (số bài train-period < dim), code `print [warn]` rồi fit PCA trên **ALL** articles (kể cả val/test) thay vì fail/reduce-dim. PCA đã được design.md mục 4.1 hứa "fit trên TRAIN articles ONLY (không leakage)".
**Kịch bản lỗi:** Dim=64, train ticker-matched articles ~1000 → fallback thường KHÔNG fire ở default. NHƯNG nếu user giảm data hoặc tăng `--dim`, fallback fire → val/test news variance directions leak vào projection → val metrics inflate → go/no-go (requirements §3 criterion #4) bị sai lệch. Silent (chỉ print, không flag trong output).
**Fix:** Nếu train articles < dim → auto `dim = max(1, train_count // 10)` hoặc raise hard error. KHÔNG widen scope sang all articles.

### [HIGH-2] Date-key format mismatch → silent all-zero embeddings (catastrophic)
**File:** `code/dataset_embedding.py:118-129` (window_dates) vs `code/extract_embeddings.py:93` (cache key `r["date"][:10]`) · **Verdict:** PLAUSIBLE (phụ thuộc format date HAR)
**Vấn đề:** Cache keys = `unified_articles.csv` date truncated `[ :10]` (format YYYY-MM-DD sau aggregation). Window dates = `stock_feats['date'].astype(str)` từ HAR processed CSV. Nếu HAR date có time component (`2024-01-15 00:00:00`) hoặc format khác → **zero match** → `_pad_articles(None)` → all-zero x_emb + mask=0 cho mọi (stock,day). Model vẫn train (chỉ HAR), không có warn/assert nào báo news branch = 0 đóng góp.
**Kịch bản lỗi:** HAR date parse khác format → toàn bộ nhánh news hóa rỗng → kết quả giống HAR-only mà không ai biết → baseline vô nghĩa.
**Fix:** (1) Assert `sum(mask) > 0` cho ít nhất 1 window ở _create_sequences. (2) Normalize cả 2 bên về `YYYY-MM-DD` chuẩn trước khi join. (3) Log coverage "% stock-day có tin match".

### [HIGH-3] Optimizer thiếu weight_decay — vi phạm CLAUDE.md §3.E
**File:** `code/train_embedding_baseline.py:136` · **Verdict:** CONFIRMED
**Vấn đề:** `torch.optim.Adam(model.parameters(), lr=args.lr)` — không có `weight_decay`. CLAUDE.md §3.E mandatory: "weight_decay=1e-5 cho LSTM" + checklist "Weight decay set (1e-5)".
**Kịch bản lỗi:** Với ~953 train stock-days + 33K-param news branch, không L2 → overfit train volatility → val DirAcc regress dưới scalar-sentiment baseline → kết luận sai "embedding không tín hiệu" trong khi thực ra là thiếu regularization.
**Fix:** `torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)`.

### [MEDIUM-4] PCA cutoff theo lịch, không khớp split boundary
**File:** `code/extract_embeddings.py:36` (`TRAIN_CUTOFF="2020-01-01"`) · **Verdict:** CONFIRMED
**Vấn đề:** Dataset split theo row-index ratio 0.7 (train_ratio), không phải calendar. PCA cutoff cố định 2020-01-01 → "train" của PCA ≠ "train" của supervision. Weak leakage (PCA unsupervised) nhưng cutoff sai về nguyên tắc.
**Fix:** Truyền split boundary thật (từ `_split_raw_data_by_date` return) vào extract, hoặc làm PCA fit trong dataset (biết split) thay vì offline riêng.

### [MEDIUM-5] Ticker regex case-sensitive → miss ticker viết thường
**File:** `code/extract_embeddings.py:39` · **Verdict:** CONFIRMED
**Vấn đề:** `re.compile(r"\b(...)\b")` không có `re.IGNORECASE`. Tin VN hay viết thường ("cổ phiếu ssi giảm") → không match → cache stock đó trống cho ngày đó. (Lưu ý: cùng behavior với `src/sentiment_baseline/process_news_to_sentiment.py` — consistent với prior art, nhưng vẫn là gap thật.)
**Fix:** Thêm `flags=re.IGNORECASE` (cẩn thận false-positive với từ thông thường — giữ `\b` boundary).

### [MEDIUM-6] NaN propagation không có guard → invalid JSON
**File:** `code/model_embedding.py:50-52` (softmax NaN nếu input NaN) + `code/train_embedding_baseline.py:74-86, 168` · **Verdict:** PLAUSIBLE
**Vấn đề:** Nếu PhoBERT emit NaN cho 1 article (UTF-8 lỗi...) → `_pad_articles` set `mask=1` (chỉ check `len>0`, không check finite) → proj(NaN)=NaN → softmax NaN → daily NaN → pred NaN. Trong `validate`, NaN leak vào `preds_n` → `evaluate_predictions` trả NaN → `json.dumps` ghi literal `NaN` (invalid JSON).
**Fix:** (1) `extract_embeddings.py`: `assert not np.isnan(embs).any()` sau encode. (2) `_pad_articles`: check `np.isfinite(arr)` trước khi set mask=1. (3) `train`: `json.dumps(..., allow_nan=False)` để fail rõ thay vì ghi NaN.

### [MEDIUM-7] Attention query không nhận gradient ở ngày chỉ 1 bài (common case)
**File:** `code/model_embedding.py:41, 47-52` · **Verdict:** CONFIRMED
**Vấn đề:** Ngày có đúng 1 article thật → softmax over single entry = 1.0 → `daily == h[article]` → query nhận gradient 0 cho ngày đó. Vì VN news thưa (đa số 0-1 article/stock-day), query chủ yếu train trên ít ngày đa-bài → "attention pooling" thực chất no-op cho phần lớn samples, dự đoán dominated bởi `proj(article)`.
**Fix:** Đây là design limitation của attention-pool-on-query cho sparse data. Accept và document, HOẶC dùng mean-pooling (đơn giản, không query) cho MVP — đơn giản hơn và không bị vấn đề này.

### [MEDIUM-8] Dead ParallelLSTMGNN fusion MLP ~23K params không dùng
**File:** `code/model_embedding.py:58` · **Verdict:** CONFIRMED
**Vấn đề:** `EmbeddingBaseline` instantiate `ParallelLSTMGNN(config)` (gồm fusion MLP 320→64→32→1 ≈ 23K params) nhưng chỉ gọi `get_embeddings` — fusion MLP nhận 0 gradient (dead weight), phình Adam state + checkpoint + param count báo cáo.
**Fix:** Build 1 `ParallelLSTMGNNFeatureExtractor` (chỉ LSTM+GAT, bỏ fusion) — hoặc chấp nhận (đã doc trong design, trade-off cho isolation). Ưu tiên fix nếu param budget quan trọng cho báo cáo.

### [MEDIUM-9] Console output vi phạm CLAUDE.md §3.B (thiếu Val/Test comparison table)
**File:** `code/train_embedding_baseline.py:154-177` · **Verdict:** CONFIRMED
**Vấn đề:** §3.B mandatory format: bảng `Metric / Validation / Test / Difference` + console blocks `Val <X>: ...` + `Test <X>: ...`. Hiện tại chỉ in `=== TEST ===` block + per-epoch table tự định nghĩa. JSON thiếu `validation_metrics` + `val_test_diff` keys.
**Fix:** Thêm final comparison table (Val vs Test, 6 metrics, diff) + JSON keys `validation_metrics`, `val_test_diff` theo §3.B template.

### [MEDIUM-10] MAX_ARTICLES=5 "keep first 5" — truncation tùy tiện, đánh bại design pooling
**File:** `code/dataset_embedding.py:24, 68-69` · **Verdict:** PLAUSIBLE (altitude)
**Vấn đề:** `arr[:n]` giữ 5 bài đầu theo thứ tự CSV/crawl (không phải relevance/recency). `ArticleSetAttentionPooling` đã handle variable-length qua mask — cap=5 giới hạn nó. Ngày >5 bài (tin nóng) → bỏ bài có thể quan trọng.
**Fix:** Tăng cap lên percentile 99 (vd 10-12), hoặc bỏ cap hoàn toàn (dynamic padding qua collate_fn). Ít nhất: document lý do chọn 5 bằng data (percentile), không phải assertion.

---

## Dropped (lower severity, dưới cap 10)
- Dataloader builder duplicate 3× (cleanup) — `create_embedding_dataloaders` gần verbatim `create_sentiment_dataloaders`.
- `_pad_articles` ~570K python calls + x_emb ~700MB redundant RAM (efficiency) — vectorize at cache-load.
- Dummy-emb smoke fallback scattered qua 3 file (altitude) — nên gate bằng 1 flag explicit.
- `_emb_dim` inconsistency across caches (mixed dim) → broadcast crash (low, user-error scenario).
- Default `--epochs 20` vs rule 70 (deviation justified cho matched-epoch comparison, nhưng cần comment).

## Clean (verified, không bug)
- Temporal split (§3.A): compliant — `_split_raw_data_by_date` chronological.
- DirAcc sign-of-changes (§3.B bug warning): compliant — `np.sign(np.diff(...))`.
- HAR leakage / normalizer scope: clean — fit trên train only, share by reference.
- `d_gat = gat_num_heads * gat_hidden_dim`: đúng (GAT output = heads × hidden).
- Isolation (§3.F): compliant — read-only imports, writes cô lập.

## Re-verify (sau fix) — DONE 2026-07-07
- [x] Fix tất cả 10 findings (3 HIGH + 7 MEDIUM)
- [x] `pytest test/` → 6 passed
- [x] `--smoke --epochs 2` → chạy sạch, Val/Test table đúng format §3.B, JSON hợp lệ
- [ ] (chưa làm) Chạy với PhoBERT embeddings thật → mới đánh giá go/no-go được
