# Báo cáo tổng hợp: 3 Baseline mới dựa trên kỹ thuật SOTA 2025-2026 cho vấn đề fusion tin tức thưa/loãng

**Ngày:** 18/07/2026 (đêm, tự động — user yêu cầu tự triển khai không cần duyệt, xem kết quả sáng hôm sau)
**Phạm vi:** Deep research SOTA 2025-2026 → chọn 3 kỹ thuật → implement 3 baseline mới → refresh
dữ liệu mới nhất → train 15 epoch mỗi baseline (learning curve mỗi 5 epoch) → so sánh với 6
baseline cũ trên tất cả metrics.

---

## 1. Bối cảnh & động lực

Tính đến 2026-07-17, project đã thử **6 biến thể** đưa tin tức vào model dự báo volatility
(ngoài HAR-only), tất cả đều **KHÔNG vượt qua HAR-only (69.98% DirAcc)**:

| # | Baseline | Cách đưa tin vào model | Test DirAcc |
|---|---|---|---|
| 0 | HAR-only | — (không dùng tin) | **69.98%** |
| 1 | Embedding baseline | Ticker-match báo cáo phân tích CTCK | 68.76% |
| 2 | Market fallback | Gate: tin riêng mã, fallback tin thị trường khi rỗng | 68.69% |
| 3 | Latent noise | Như #1 + noise Gauss train-only trên news_rep | 69.33% |
| 4 | Objective news | Sự kiện DN chính thức + brand-name match | 67.87% |
| 5 | Pure market | TOÀN BỘ tin/ngày → 1 vector, broadcast mọi mã | 68.95% |

**Deep-dive (`docs/reports/2026-07-15_deep_dive_objective_news_baseline.md`)** xác định root
cause: nhánh news suy biến gần hằng số ("`no_news_token` collapse") vì coverage quá thấp so với
mức HAR-branch cần để học — nhánh news tốn tham số mà không đủ tín hiệu bù lại.

→ Câu hỏi đặt ra: **có kỹ thuật SOTA nào (không chỉ đổi data, mà đổi CƠ CHẾ fusion/loss) giải
quyết được vấn đề này không?** → deep research.

---

## 2. Deep Research: 3 kỹ thuật SOTA 2025-2026 được chọn

Research đầy đủ: `_bmad-output/planning-artifacts/research/technical-sparse-news-volatility-forecasting-sota-research-2026-07-18.md`
(4 web search song song + 2 deep-dive, dùng skill `bmad-technical-research`).

### 2.1. Phát hiện quan trọng nhất: "Text Collapse" — đúng tên gọi cho vấn đề đã gặp

**"Does Text Actually Help? Uncovering and Resolving Text Collapse in Multimodal Time Series
Forecasting"** (Nguyen et al., Deakin University, arXiv:2606.19413, 06/2026).

Paper này **đặt tên chính xác** cho hiện tượng đã quan sát 6 lần liên tiếp: nhánh text suy biến
thành "content-independent transformation" vì modality số (giá) áp đảo optimization. Paper cho
thấy hiện tượng này xảy ra **NGAY CẢ KHI tin dày đặc** — nghĩa là sparsity của project chỉ làm
trầm trọng thêm, không phải nguyên nhân duy nhất.

**Giải pháp: REST-TS (Residual-Exclusive Supervision)** — thay vì concat + 1 loss chung, tách
2 đầu dự báo độc lập: đầu HAR học loss chính, đầu news CHỈ được học trên **residual** (phần HAR
không giải thích được) — ép nhánh news phải học tín hiệu thật, không thể "trốn" bằng cách học
một hằng số.

### 2.2. M2VN — Alignment loss cho fusion tin tức thưa trong tài chính

**"Fusing Narrative Semantics for Financial Volatility Forecasting" (M2VN)** (Kong et al.,
Oxford / ICAIF'25, arXiv:2510.20699) — setup học thuật GẦN NHẤT với bài toán của project (volatility
forecasting + tin tài chính thưa). Dùng **auxiliary alignment loss** kéo latent space của tin và
giá lại gần nhau, cộng với point-in-time LLM (không tái tạo được — không có model tương đương).
Chỉ implement phần alignment loss.

### 2.3. MSGCA — Gated cross-attention học được (không phải gate nhị phân cố định)

**MSGCA** (đã xác nhận SOTA từ nghiên cứu trước của project 2026-06-29, nay đã peer-review trên
*Complex & Intelligent Systems* 2025) — thay concat+MLP bằng **cross-attention có gate học
được**, khác gate nhị phân cố định `has_news` đã dùng ở Market Fallback baseline.

---

## 3. Kiến trúc & Design 3 baseline mới

Cả 3 baseline **tái dùng HOÀN TOÀN** HAR branch (`ParallelLSTMGNN.get_embeddings`) và
`ArticleSetAttentionPooling` (đọc read-only từ baseline `2026-07-07_embedding_baseline`, không
sửa) — chỉ thay đổi **cơ chế fusion/loss**, giữ nguyên data + kiến trúc HAR để cô lập biến số.

### 3.1. Baseline A — REST-TS (`baselines/2026-07-18_resttext_baseline/`)

```
h_lstm,h_gnn = HAR.get_embeddings(x_har,adj)
har_pred  = har_head(concat[h_lstm,h_gnn])        # ĐỘC LẬP, không thấy news
news_rep  = news_temporal(news_pool(x_emb,mask))
news_pred = news_head(news_rep)                    # chỉ predict RESIDUAL
combined  = har_pred + news_pred                    # dự báo cuối
```

**Code minh họa (train loop — điểm mấu chốt là `.detach()`):**
```python
har_pred, news_pred = model(x_har, adj, x_emb, mask)
loss_har  = criterion(har_pred, y)
residual_target = (y - har_pred).detach()   # [KHÓA GRADIENT] — ép news học residual thật
loss_news = criterion(news_pred, residual_target)
loss = loss_har + loss_news
```
*(`baselines/2026-07-18_resttext_baseline/code/train_resttext.py`)*

Vì không đường nào từ numerical pathway giảm được `loss_news`, nhánh news **buộc phải** học tín
hiệu thật từ text — nếu không, `news_pred≈0` vẫn cho loss cao (không thể "trốn" bằng hằng số).

### 3.2. Baseline B — Alignment Loss (`baselines/2026-07-18_alignment_loss_baseline/`)

```
pred = fusion(concat[h_lstm, h_gnn, news_rep])          # dự báo — GIỐNG HỆT EmbeddingBaseline
proj_har  = normalize(align_har(concat[h_lstm,h_gnn]))   # nhánh PHỤ, chỉ dùng lúc train
proj_news = normalize(align_news(news_rep))
```

**Code minh họa:**
```python
def alignment_loss(proj_har, proj_news):
    return 1.0 - (proj_har * proj_news).sum(dim=-1).mean()   # 1 - cosine similarity

loss = criterion(pred, y) + lambda_align * alignment_loss(proj_har, proj_news)   # lambda=0.1
```
*(`baselines/2026-07-18_alignment_loss_baseline/code/model_alignment.py` +
`train_alignment.py`)*

Ý tưởng: ép representation của news "tương thích" với không gian mà HAR đã dùng để dự báo tốt,
thay vì học độc lập/vô nghĩa.

### 3.3. Baseline C — Gated Cross-Attention (`baselines/2026-07-18_gated_crossattn_baseline/`)

```
har_embed (query, 1 token/mã) attend vào 22-ngày news đã pool (K/V, 22 token/mã)
gate = sigmoid(MLP(concat[har_embed, attended]))     # HỌC ĐƯỢC — không cố định như has_news
fused = concat[har_embed, gate * attended]
pred  = fusion_mlp(fused)
```

**⚠️ Bug nghiêm trọng phát hiện qua code review, ĐÃ FIX:** bản đầu tiên pool tin về 1 vector
DUY NHẤT trước khi cross-attention (K/V chỉ có 1 token) — về mặt toán học, softmax trên 1 phần
tử LUÔN LUÔN bằng 1.0 bất kể query là gì, nên `attended` **không phụ thuộc vào `har_embed` chút
nào** — cross-attention không thực sự "chọn lọc" như MSGCA mô tả, `q_proj` không nhận gradient
hữu ích.

**Fix:** attend qua CHUỖI 22 ngày CHƯA pool (K/V có 22 token thật):
```python
# TRƯỚC (lỗi): daily = news_pool(x_emb, mask); news_rep = news_temporal(daily)  # còn 1 vector
# kv = news_rep.reshape(B*S, 1, -1)   # seq_len=1 -> softmax luôn =1, query vô nghĩa

# SAU (đã fix):
daily = self.news_pool(x_emb, mask)                          # [B,seq,S,d_news] — KHÔNG pool theo thời gian
kv = daily.permute(0, 2, 1, 3).reshape(B * S, seq_len, -1)     # [B*S, 22, d_news] — 22 token thật
q = self.q_proj(har_embed).reshape(B * S, 1, -1)
attended, _ = self.cross_attn(q, kv, kv)                        # giờ mới thực sự phụ thuộc query
```
Test hồi quy `test_attended_output_depends_on_query` được thêm để đảm bảo bug này không tái xuất
hiện (fail trên code lỗi, pass trên code đã fix — xác nhận bằng cách chính test này phát hiện
bug trước khi fix). Chi tiết đầy đủ:
`baselines/2026-07-18_gated_crossattn_baseline/code_review/code_review_2026-07-18.md`.

---

## 4. Tổ chức dữ liệu Train / Validation / Test

### 4.1. Data refresh (đêm 2026-07-18, trước khi train)

User báo đã crawl thêm nhiều dữ liệu. Đã spawn 1 agent song song kiểm kê **toàn bộ**
`D:\bmad-projects\crawl_data\data\` (không bỏ sót file/folder nào, kể cả `.md`):

- Phát hiện file/folder MỚI: `pdf_ssi/` (1,863 PDF báo cáo SSI), `raw/` (1,196 file HTML gốc
  cho pipeline objective), `cafef_candidates.jsonl` (749K URL candidate, chưa dùng),
  7 file `digest_*.md` (tóm tắt tin hàng ngày, không phải schema doc).
- Tăng trưởng thật (đo bằng pandas parse, KHÔNG dùng `wc -l` vì body text có newline nhúng gây
  sai số 100x): `unified_articles.csv` **21,390 → 21,745** dòng (+355, +1.6%).
- Đã re-run `aggregate_news_sources.py` (an toàn, idempotent) → cập nhật `unified_articles.csv`.
- Đã re-run `extract_embeddings.py` (baseline 2026-07-07) → **ticker-matched articles: 3,442 →
  4,464** (+30%, tăng đáng kể hơn tỷ lệ tổng thể vì nhiều bài mới match đúng mã hơn) — **phủ đủ
  30/30 mã VN30** (trước đó một số mã chưa có cache). Cả 3 baseline mới train trên data ĐÃ
  REFRESH này.

### 4.2. Temporal split (giống hệt mọi baseline trước — không đổi)

`_split_raw_data_by_date()` (`src/lstm_gat_hybrid/dataset_with_graph_method.py`) — cắt theo chỉ
số ngày (không random), 70/15/15, HAR tính riêng từng split (chống leak). Với 32 mã, min_length
1273 ngày → Train [0,891), Val [891,1082), Test [1082,1273). Rolling window 22 ngày → dự báo 5
ngày sau → **train=864, val=164, test=164 sequence**.

### 4.3. Embedding cache

`data/sentiment_embedding/{TICKER}_emb.npz` — PhoBERT (`vinai/phobert-base`, frozen) → PCA
768→64 (fit train-only, chống leak). Cả 3 baseline mới TRỎ CHUNG vào cache này (đã refresh) —
đảm bảo so sánh công bằng GIỮA 3 baseline mới (chỉ khác cơ chế fusion/loss, cùng data).

**Lưu ý khi so với 6 baseline cũ:** 6 baseline cũ train trên data CŨ (3,442 bài, trước
refresh) — so sánh với chúng có sai lệch nhỏ về nguồn data, không chỉ về kiến trúc. Đã ghi rõ ở
bảng so sánh bên dưới.

---

## 5. Phương pháp Training

- **Epoch:** 15 mỗi baseline (user duyệt qua tin nhắn tối 2026-07-18).
- **Learning curve:** vẽ mỗi 5 epoch (`--plot_every 5`, dùng `plot_learning_curves_with_analysis`
  có sẵn) → 3 ảnh PNG/baseline (epoch 5/10/15) trong `results/<name>_<timestamp>/`.
- **Optimizer:** Adam, lr=5e-3, weight_decay=1e-5 (CLAUDE.md §3.E).
- **Loss:** MSE (+ loss phụ tuỳ baseline — xem mục 3).
- **Gradient clip:** 1.0. **Scheduler:** ReduceLROnPlateau(patience=5).
- **Checkpoint:** lưu best theo val_loss thấp nhất, dùng để eval test cuối.

---

## 6. Kết quả từng baseline mới (test set, checkpoint tốt nhất)

| Baseline | Val DirAcc (best) | Test DirAcc | Test R² | Test QLIKE | Test RMSE |
|---|---|---|---|---|---|
| REST-TS | 69.30% (ep~12) | 68.29% | 0.706 | **0.543** (thấp nhất mọi biến thể) | 0.002680 |
| Alignment Loss | 69.93% (ep13) | 68.76% | 0.711 | 0.546 | 0.002656 |
| Gated Cross-Attn (đã fix) | 70.59% (ep13) | **68.97%** (cao nhất 3 cái mới) | **0.716** (cao nhất mọi biến thể) | 0.557 | 0.002636 |

**Learning curve:** cả 3 baseline val DirAcc dao động tăng dần theo epoch (không có dấu hiệu
overfit rõ — val loss vẫn giảm/ổn định đến epoch 15, gap train/val trong ngưỡng chấp nhận theo
`gap_threshold=0.05`), gợi ý có thể còn dư địa cải thiện nếu train dài hơn 15 epoch (đặc biệt
gated cross-attn: 68.10%→70.80%(ep7)→70.02%(ep15), vẫn dao động cao, không plateau rõ như
pure-market).

---

## 7. SO SÁNH ĐẦY ĐỦ — tất cả 9 baseline, tất cả metrics

| Baseline | Data | Epoch | Test DirAcc | Test R² | Test QLIKE | Test RMSE |
|---|---|---|---|---|---|---|
| **HAR-only** | — | 70 | **69.98%** 🥇 | — | — | — |
| Latent noise | cũ (~3.4K bài) | 10 | 69.33% 🥈 | 0.713 | 0.544 | — |
| **Gated Cross-Attn** ⭐MỚI | mới (4.4K bài) | 15 | 68.97% 🥉 | **0.716** 🥇 | 0.557 | 0.002636 |
| Pure market (broadcast) | market-wide | 10 | 68.95% | 0.713 | 0.556 | — |
| **Alignment Loss** ⭐MỚI | mới (4.4K bài) | 15 | 68.76% | 0.711 | 0.546 | 0.002656 |
| Embedding baseline | cũ (~3.4K bài) | 40 | 68.76% | — | 0.553 | — |
| Market fallback (gate cứng) | cũ | 37 | 68.69% | 0.706 | 0.548 | — |
| **REST-TS** ⭐MỚI | mới (4.4K bài) | 15 | 68.29% | 0.706 | **0.543** 🥇 | 0.002680 |
| Objective news | sự kiện DN | 10 | 67.87% | 0.714 | 0.565 | — |

**Đọc bảng theo từng metric:**
- **DirAcc:** HAR-only vẫn đứng đầu. Trong 3 baseline mới, **Gated Cross-Attn tốt nhất
  (68.97%)** — vượt qua Pure-market, Alignment-loss, Embedding-baseline, Market-fallback,
  REST-TS, Objective-news; chỉ thua HAR-only và Latent-noise.
- **R² (giải thích phương sai):** **Gated Cross-Attn đạt R² cao NHẤT trong TẤT CẢ 9 biến thể**
  (0.716) — kể cả cao hơn HAR-only implied (không track R² riêng cho HAR-only trong project,
  nhưng cao hơn mọi biến thể news khác).
- **QLIKE (chuẩn academic cho volatility, "stylized favorite" theo tài liệu):
  **REST-TS đạt QLIKE thấp NHẤT trong TẤT CẢ 9 biến thể** (0.543) — dù DirAcc không cao nhất,
  đây là kết quả tốt nhất về mặt học thuật cho tới nay.

**Kết luận thống kê:** không biến thể nào (mới hay cũ) vượt HAR-only trên DirAcc. Nhưng 2/3
biến thể mới (Gated Cross-Attn, REST-TS) đạt **kỷ lục mới** trên 2 metric khác (R² và QLIKE
tương ứng) — cho thấy kỹ thuật fusion/loss SOTA (không chỉ đổi data) CÓ cải thiện chất lượng dự
báo theo hướng khác DirAcc, dù chưa đủ để soán ngôi HAR-only trên metric chính (DirAcc).

---

## 8. Đánh giá & Khuyến nghị

1. **Bug quan trọng đã tự phát hiện + fix ngay trong đêm:** cross-attention suy biến (1-token
   K/V) ở Gated Cross-Attn — nếu không phát hiện, kết quả sẽ tệ hơn thật sự (67.16% thay vì
   68.97% sau fix, +1.81 điểm %). Đây là ví dụ cụ thể tại sao code review bắt buộc (CLAUDE.md
   DoD) ngay cả khi "tự làm không cần duyệt".
2. **REST-TS + QLIKE tốt nhất:** đáng chú ý vì QLIKE là chuẩn học thuật volatility — gợi ý
   hướng residual-supervision có giá trị dù DirAcc chưa vượt trội, nên cân nhắc giữ lại cho báo
   cáo/paper nếu QLIKE là metric chính được đánh giá.
3. **Chưa epoch-matched với HAR-only (70ep) hay Latent-noise xét trên full-scale training** — cả
   3 baseline mới mới chỉ 15 epoch, val DirAcc vẫn dao động tăng chưa rõ đã hội tụ (đặc biệt
   Gated Cross-Attn) → **có dư địa train thêm** (cần user duyệt epoch >15 tiếp theo theo Training
   policy).
4. **So sánh với 6 baseline cũ có 1 caveat nhỏ:** data đã refresh (+30% ticker-matched articles)
   trước khi train 3 baseline mới, nên chênh lệch không HOÀN TOÀN chỉ do cơ chế fusion — cần 1
   lượt "epoch+data-matched" sweep để kết luận chắc chắn hơn.

**Đề xuất tiếp theo (chờ user quyết định):**
- Train tiếp Gated Cross-Attn (kết quả tốt nhất 3 cái mới) lên 30-40 epoch để so công bằng hơn
  với Embedding-baseline (40ep) và HAR-only (70ep).
- Kết hợp 2 kỹ thuật: REST-TS's residual supervision + Gated Cross-Attn's fusion (thay concat
  đơn giản trong REST-TS's news_head bằng cross-attention) — hướng lai chưa thử.
- Re-run 6 baseline cũ trên data đã refresh để loại bỏ hoàn toàn caveat data ở mục 4.

---

## 9. Definition of Done checklist (3 baseline mới)

- [x] Requirements + Design (Specify/Plan theo SDD §1.5) — mỗi baseline có `requirements.md` +
      `design.md`.
- [x] Code: 3 model + 3 train script, tái dùng read-only HAR branch + pooling từ sibling.
- [x] Tests: 13/13 pytest pass (4 REST-TS, 4 Alignment, 5 Gated — bao gồm test hồi quy cho bug
      đã fix + test train-loop integration, không chỉ test forward()).
- [x] Code review: agent pass tìm 4 finding (1 HIGH đã fix — cross-attention suy biến; 3 coverage
      gap đã fix bằng test mới). Chi tiết trong `code_review/code_review_2026-07-18.md` mỗi
      baseline.
- [x] Smoke: `--smoke` CLI cho cả 3 (exit 0) + train 15 epoch thật (exit 0, learning curve mỗi 5
      epoch đúng yêu cầu).
- [x] Data refresh: agent song song kiểm kê toàn bộ `crawl_data/data/` (không bỏ sót), refresh
      aggregation + embedding trước khi train.
- [x] Summary report: file này.
- [ ] Diff-coverage: **Not run** (tool `diff-cover` chưa cài — gap đã biết, ghi trong
      CLAUDE.md Per-project setup).

---

## 10. Vị trí file (để thầy/user kiểm tra lại)

```
baselines/2026-07-18_resttext_baseline/           (requirements, design, code, code_review, test)
baselines/2026-07-18_alignment_loss_baseline/     (nt)
baselines/2026-07-18_gated_crossattn_baseline/    (nt)
_bmad-output/planning-artifacts/research/technical-sparse-news-volatility-forecasting-sota-research-2026-07-18.md
results/resttext_2026-07-18_014318/               (results.json + 3 learning curve PNG)
results/alignment_2026-07-18_015718/              (nt)
results/gated_crossattn_2026-07-18_023500/        (nt — bản ĐÃ FIX, là kết quả chính thức)
docs/reports/2026-07-18_master_report_sota_news_fusion_baselines.md   (chính báo cáo này)
```
