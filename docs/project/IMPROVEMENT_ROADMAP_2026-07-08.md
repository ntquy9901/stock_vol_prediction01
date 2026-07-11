# Roadmap cải thiện — Post News-Direction Ceiling (2026-07-08)

**Ngày:** 08/07/2026
**Trigger:** 3/3 kiến trúc news (scalar / embedding / market-fallback) đều **không vượt HAR-only** (69.98% DirAcc). Bottleneck = DATA, không phải architecture.
**Tham chiếu:** `EMBEDDING_BASELINE_REPORT_2026-07-08.md`, `SENTIMENT_MARKET_FALLBACK_ARCHITECTURE.md`, `crawl_data/aggregated/sparsity_report.txt`

---

## 0. Strategy (rút ra từ kết quả)

```
scalar sentiment : 67.96%   ┐
embedding news   : 68.76%   ├─ tất cả < HAR-only 69.98% (70ep)
market fallback  : 68.69%   ┘
```

→ **Đã thử đủ kiến trúc sentiment.** Thêm VAE/MSGCA/learned-gate sẽ KHÔNG giúp (tín hiệu gốc yếu). **Chuyển effort: architecture → DATA.** Quy tắc: mỗi phase có **kill criteria** — nếu no-lift sau data tốt hơn → sentiment thực sự không tín hiệu cho vol 5-day VN, focus HAR.

---

## Phase 0 — Matched-epoch control (NGAY, phải làm) 🎯

**Mục tiêu:** loại confound epoch (HAR 70ep vs news 40ep). Chốt công bằng: news có HARM/HELP/NEUTRAL?

| Bước | Việc | Verify |
|---|---|---|
| 0.1 | Train HAR-only @ 40ep (cùng setup embedding/market) | results/har_only_40ep |
| 0.2 | Re-run embedding @ 40ep (đã có: 68.76%) | đã có |
| 0.3 | Re-run market @ 40ep (đã có: 68.69%) | đã có |
| 0.4 | So sánh 3 @ cùng 40ep + statistical test (skill `statistical-analysis`) | bảng + p-value |

**Effort:** ~1-2 giờ (chỉ run + compare). **KHÔNG code mới.**
**Kill criteria:** nếu HAR@40 cũng ~68-69% → news NEUTRAL (không hại không giúp) → kết luận fair. Nếu HAR@40 ≈ 70% → news HARM (gây nhiễu) → bỏ news hoàn toàn.
**Output:** 1 comparison doc `MATCHED_EPOCH_CONTROL_2026-07-XX.md`.

---

## Phase 1 — Data acquisition (LEVER THẬT, cao priority) 🚀

**Mục tiêu:** phá ceiling title-only. Đây là hướng duy nhất có khả năng lift thật (theo design doc Mục 10 + sparsity report).

### 1A. Body bài (không chỉ title) — Ưu tiên #1

> **🔄 REDIRECT (08/07/2026):** Crawling do **project khác** đảm nhiệm. Project này KHÔNG crawl — chỉ **consume** body text khi external project cung cấp (re-embed qua `extract_embeddings --use_body`). Plan crawler chi tiết (`cryptic-forging-rabin.md`) đã **DỪNG**, code partial `src/news_body/` đã xóa. Chờ body corpus từ project crawler.

**Vì sao:** title ngắn VN = ceiling embedding thấp (~99% ngữ nghĩa mất). Body = ngữ cảnh đầy đủ → embedding/sentiment chất lượng hơn bậc.
| Bước | Việc |
|---|---|
| 1A.1 | Audit `pdf/` folder (có PDF đã download chưa? data.csv có `pdf_filename`) |
| 1A.2 | Crawler body: tận dụng `pdf_url` sẵn có → download + extract text (pdfplumber/pyMuPDF) |
| 1A.3 | Re-extract embedding (PhoBERT trên title+body thay vì chỉ title) |
| 1A.4 | Re-train embedding baseline @ 40ep + so matched-epoch |

**Effort:** ~3-5 ngày (crawler + PDF extract). **Effort cao nhất nhưng lever lớn nhất.**

### 1B. More ticker-specific news — Ưu tiên #2
**Vì sao:** chỉ ~20% bài match mã VN30 → 94.5% stock-day mù tin. Tăng coverage = tăng signal trực tiếp.
| Bước | Việc |
|---|---|
| 1B.1 | Source-typed crawl: cafef ticker-tagged, broker reports có mã rõ, company filings |
| 1B.2 | Ticker extraction cải tiến (fuzzy + alias: "Vietcombank"→VCB) |
| 1B.3 | Re-run sparsity analysis xem coverage tăng bao nhiêu |

**Effort:** ~3-5 ngày. Có thể song song 1A.

**Kill criteria Phase 1:** nếu body+ticker-rich news VẪN không vượt HAR @ matched epoch → **sentiment không tín hiệu cho vol 5-day VN** → chốt, focus HAR (Phase 3).

---

## Phase 2 — Conditional architectural (CHỈ nếu Phase 1 có signal) 🔧

**Gate:** chỉ làm nếu Phase 1 (data tốt hơn) cho lift > +1% DirAcc. Nếu không → skip toàn bộ Phase 2.

| # | Hướng | Khi nào |
|---|---|---|
| 2.1 | Sector-level sentiment (banking, real-estate...) | nếu cần middle-ground giữa per-stock/market |
| 2.2 | Horizon khác (1-day returns thay 5-day vol) | nếu nghi news effect ngắn hạn |
| 2.3 | FinBERT / financial VN model thay PhoBERT | nếu embedding domain chặt hơn |
| 2.4 | Cross-attention MSGCA / learned gate | chỉ khi base signal đã có |

**Effort:** mỗi cái ~2-4 ngày. **KHÔNG làm nếu Phase 1 no-lift.**

---

## Phase 3 — HAR improvements (fallback nếu sentiment = no-signal) 📊

**Nếu Phase 1 kill:** sentiment kết luận không tín hiệu. HAR-only (69.98%) là production. Cải thiện qua HAR:
| # | Hướng |
|---|---|
| 3.1 | HAR-X variants (thêm technical features: volume, return) |
| 3.2 | Ensemble (HAR + LSTM-HAR + Parallel LSTM-GNN) |
| 3.3 | Hyperparameter sweep HAR-only |
| 3.4 | Multi-stock cross-correlation sâu hơn (GNN改进) |

---

## Decision gates (kill criteria tổng)

```
Phase 0 done → HAR@40 vs news@40公平?
   ├─ news NEUTRAL → tiếp Phase 1 (data lever)
   └─ news HARM → bỏ sentiment, jump Phase 3

Phase 1 (body+rich news) done → lift > +1% DirAcc?
   ├─ CÓ → tiếp Phase 2 (architectural refinement)
   └─ KHÔNG → sentiment = no-signal, Phase 3 (HAR)
```

## Effort / priority summary

| Phase | Effort | Kỳ vọng lift | Priority |
|---|---|---|---|
| 0 matched-epoch control | ~1-2 giờ | clarify (không lift, chỉ fair) | **NGAY** |
| 1A body crawler | ~3-5 ngày | **cao nhất** (data lever thật) | **cao** |
| 1B ticker-rich news | ~3-5 ngày | cao (coverage) | cao |
| 2 conditional arch | ~2-4 ngày mỗi cái | trung bình (chỉ nếu P1 ok) | điều kiện |
| 3 HAR improvements | ~2-3 ngày mỗi cái | trung bình (fallback) | nếu P1 kill |

## Recommend (tuần tới)

1. **Phase 0 ngay** (1-2 giờ, cheap, loại confound) — tôi có thể làm luôn.
2. **Phase 1A song song** (body crawler) — lever thật, bắt đầu audit `pdf/` + `pdf_url`.
3. Đặt **kill gate** rõ: sau Phase 1, nếu vẫn no-lift → dừng sentiment, không đầu tư Phase 2.

---

**Phiên bản:** 1.0 — 08/07/2026 · **Strategy:** architecture → data · **Principle:** mỗi phase có kill criteria, không đầu tư tiếp khi no-signal.
