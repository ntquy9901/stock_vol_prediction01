# Thiết kế — Kỹ thuật Latent Space cho nhánh Sentiment (Phase 3)

**Ngày:** 05/07/2026
**Tác giả:** ntquy99
**Trạng thái:** Thiết kế + research, chưa triển khai code
**Liên quan:** `SENTIMENT_ANALYSIS_DESIGN.md` (Phase 3, Mục 5.3), `SENTIMENT_NEWS_EMBEDDING_ARCHITECTURE.md`

---

## 0. Bối cảnh & nguồn gốc

Trong quá trình thảo luận Phase 3 (Late Fusion + Cross-Attention MSGCA), **thầy góp ý**:

> *"Có thể thêm 1 layer random các vector để chống việc thiếu dữ liệu semantic lúc train. Keyword: latent space / latent vectors."*

Vấn đề cốt lõi (xem `SENTIMENT_ANALYSIS_DESIGN.md` Mục 2.5, 5.3): dữ liệu sentiment **rất thưa** — chỉ ~16% stock-day có tin matched VN30, 84% stock-day `sentiment_1d=0`. Nhánh sentiment encoder gần như chỉ thấy input = 0 → **undertrained**, latent representation "trống".

Tài liệu này research kỹ thuật thầy gợi ý, xác định thuật ngữ chính thức, và đề xuất 2 cấp độ code.

---

## 1. Bản đồ thuật ngữ — "đó là kỹ thuật gì?"

Mô tả của thầy (*"thêm 1 layer random các vector trong latent space để chống thiếu data semantic lúc train"*) khớp **5 thuật ngữ** trong literature:

| # | Thuật ngữ chính thức | Tiếng Việt thường gọi | Khớp vì sao | Khả năng là thứ thầy nói |
|---|---|---|---|---|
| **1** | **Variational Information Bottleneck (VIB)** — Alemi et al. ICLR 2017 | "Thông tin cổ chai biến phân" / "stochastic latent layer" | **Khớp nhất.** "Thêm 1 layer ngẫu nhiên hóa latent vector trong mô hình phân biệt (dự báo)". VIB = VAE cho model dự báo, cope "low-resource" là use-case nổi tiếng. | ⭐⭐⭐ Rất cao |
| **2** | **VAE Reparameterization Trick** — Kingma 2013 | "Reparameterization" / "lấy mẫu không gian ẩn" | Cùng công thức `z = μ + σ·ε`, nhưng gốc là mô hình sinh. Keyword "latent space" gợi ý nhất về VAE. | ⭐⭐ Cao |
| **3** | **Latent Noise Injection / Additive Latent Noise** | "Thêm nhiễu vào latent" / "GaussianNoise layer" | Nghĩa đen đúng nhất "thêm 1 layer random vector". Biến thể nhẹ của Dropout. | ⭐⭐ Cao |
| **4** | **Latent Space Data Augmentation** (MODALS, LSDA, Latent Mixup) | "Tăng cường dữ liệu trong không gian ẩn" | Tạo vector mới bằng interpolate/replace latent. | ⭐ Trung bình |
| **5** | **Stochastic Forward Pass / MC Dropout** | "Forward ngẫu nhiên" | Đặt Dropout ON cả lúc test, lấy mẫu nhiều lần. Liên quan nhưng không trọng tâm. | ⭐ Thấp |

**Câu hỏi xác nhận với thầy:** *"Thầy có đang nói về Variational Information Bottleneck (VIB) — thêm 1 stochastic layer `z = μ + σ·ε` vào latent representation để regularize khi data thưa không?"* — nếu đúng, đó là kỹ thuật #1 (VIB); về code implement y hệt VAE.

---

## 2. Vì sao kỹ thuật này giải quyết được bài toán

2 cơ chế:

1. **Reparameterization** `z = μ + σ·ε` — mỗi forward pass, latent vector **rung quanh μ** một lượng ngẫu nhiên → model không bao giờ thấy cùng 1 vector z 2 lần → **không overfit** vào vài example có tin thật.
2. **KL regularization** `KL(N(μ,σ²) ‖ N(0,I))` — kéo phân phối về prior chuẩn → **khi không có tin (input=0), latent space vẫn được "điền" bởi các mẫu ngẫu nhiên hợp lý** quanh 0 → "chống thiếu data semantic".

---

## 3. Cấp độ A — Latent Noise Injection (5 dòng, làm đầu tiên)

Áp dụng ngay cho encoder sentiment hiện tại, không đổi loss function. Đây là "VIB ≈ 0%":

```python
# Trong forward của sentiment encoder, SAU khi ra embedding, TRƯỚC khi fusion:
class SentimentEncoder(nn.Module):
    def __init__(self, ..., noise_std: float = 0.1):
        super().__init__()
        ...
        self.noise_std = noise_std

    def forward(self, x_sent):
        emb = self.encode(x_sent)          # [batch, stocks, emb_dim]
        if self.training and self.noise_std > 0:
            emb = emb + torch.randn_like(emb) * self.noise_std   # ← "layer random vector"
        return emb
```

**Lý thuyết nền:** additive noise ≈ một dạng Tikhonov/KL regularization.

→ **Recommend bắt đầu từ đây** (luật Simplicity First).

---

## 4. Cấp độ B — VAE/VIB đầy đủ (khi A có tín hiệu)

Thêm μ-head, logvar-head, reparameterize, KL term vào loss. Code khớp kiến trúc thật của Parallel LSTM-GNN (`forward(x, adj)`, `x=[batch, seq, stocks, feat]`, per-stock loop như LSTM stream):

```python
import torch
import torch.nn as nn

class SentimentVariationalEncoder(nn.Module):
    """
    Nhánh sentiment Phase 3 — VAE/VIB.
    Input : [batch, seq_len=22, num_stocks=30, 2]  (sentiment_1d, news_count_1d)
    Output: z   [batch, num_stocks, latent_dim]   — random latent (train) / mu (eval)
            mu, logvar  (dùng để tính KL)
    """
    def __init__(self, num_sent_features=2, hidden=32, latent_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(num_sent_features, hidden, num_layers=1, batch_first=True)
        self.fc_mu     = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def reparameterize(self, mu, logvar):
        # Train: lấy mẫu ngẫu nhiên. Eval: dùng mu (xác định, ổn định).
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x_sent):
        batch, seq, stocks, _ = x_sent.shape
        mu_list, logvar_list, z_list = [], [], []
        for s in range(stocks):                       # per-stock (giống LSTM stream)
            _, (h, _) = self.encoder(x_sent[:, :, s, :])   # h: [1, batch, hidden]
            h = h[-1]                                       # [batch, hidden]
            mu_list.append(self.fc_mu(h))
            logvar_list.append(self.fc_logvar(h))
            z_list.append(self.reparameterize(mu_list[-1], logvar_list[-1]))
        mu     = torch.stack(mu_list, dim=1)            # [batch, stocks, latent]
        logvar = torch.stack(logvar_list, dim=1)
        z      = torch.stack(z_list, dim=1)
        return z, mu, logvar


def kl_divergence(mu, logvar):
    """KL(N(mu,sigma) || N(0,I)): sum theo latent dim, mean theo batch+stocks."""
    return -0.5 * torch.mean(torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
```

Tích hợp vào training loop:

```python
# --- forward ---
z_sent, mu, logvar = model.sentiment_encoder(x_sent)   # x_sent tách ra từ x ở __getitem__
predictions = model(x_har, adj_matrix, z_sent)          # fusion dùng z_sent (random lúc train)

# --- loss: MSE như cũ + KL term ---
mse = criterion(predictions, y)
kl  = kl_divergence(mu, logvar)
loss = mse + beta * kl                                   # beta = trọng số KL

loss.backward()
```

### 4.1. 3 chi tiết quan trọng để không bị bug

1. **Tách `x_sent` khỏi `x` ở `__getitem__`** (như design Phase 2 đã làm): hiện `dataset_sentiment.py` đang concat 5 feature chung. Để nhánh sentiment riêng, phải tách `FEATURE_COLS` thành `HAR_COLS` (3) + `SENTIMENT_COLS` (2) ở nguồn data.

2. **β (KL weight) — dùng β-warmup để tránh "posterior collapse"** (lỗi VAE kinh điển: model lười, bỏ qua data, z ≈ prior N(0,I) hoàn toàn → sentiment vô dụng):
   ```python
   beta = min(1.0, epoch / warmup_epochs) * beta_max   # beta_max ~ 1e-3..1e-2
   ```
   Bắt đầu β≈0 (chỉ MSE), tăng dần. Điểm hay bị bug nhất khi implement VAE.

3. **Inference dùng μ, không sample** (đã xử lý trong `reparameterize`): test phải xác định để metric không noise.

---

## 5. Đánh giá trung thực

| Điểm | Nhận xét |
|------|----------|
| ✅ Hợp lý cho Phase 3 | VIB đúng triết lý — nhánh sentiment riêng + stochastic latent → giảm overfit khi data thưa. Cite được cho báo cáo thầy. |
| ⚠️ Không phải phép màu | Nếu bản thân tín hiệu sentiment yếu (lexicon thô + title-only như Mục 6.4 design), VIB chỉ giúp latent mịn hơn, **không tạo tín hiệu mới**. Ceiling do chất lượng scorer quyết định. |
| ⚠️ Phức tạp hơn A | Cấp B thêm ~40 dòng + KL loss + β-tuning + posterior-collapse risk. **Chỉ làm nếu A đã cho thấy tín hiệu.** |
| 🔴 Vẫn không thay crawl tin test | Thiếu tin test 2021-2025 là ưu tiên #1 (Mục 10 design). VIB giúp train/val, **không nâng metric test** nếu test mù tin. |

---

## 6. Quy trình đề xuất (verify từng bước)

```
1. Chạy baseline HAR-only tại 10/15/20 epoch          → verify: có so sánh công bằng chưa
2. Cấp A (Latent Noise Injection, noise_std=0.1)       → verify: val Dir Acc có ổn hơn không
3. Nếu A không regression + có hint cải thiện → Cấp B (VAE/VIB + β-warmup)
                                                       → verify: val loss giảm, không collapse
                                                         (check: var(mu) > 0, không ≈ 0)
4. Chỉ khi crawl đủ tin test → đánh giá test metric    → verify: Dir Acc test có lift không
```

---

## 7. Sources

- [Alemi et al. — Deep Variational Information Bottleneck (arXiv 1612.00410)](https://arxiv.org/abs/1612.00410)
- [Baeldung — VAE Reparameterization](https://www.baeldung.com/cs/vae-reparameterization)
- [abhik.ai — VAE Latent Space](https://www.abhik.ai/concepts/deep-learning/vae-latent-space)
- [NeurIPS — Regularizing DNN by Noise](http://papers.neurips.cc/paper/7096-regularizing-deep-neural-networks-by-noise-its-interpretation-and-optimization.pdf)
- [NINR — Noise Injection Node Regularization (arXiv 2210.15764)](https://arxiv.org/abs/2210.15764)
- [MODALS — Latent Space Augmentation (OpenReview)](https://openreview.net/pdf?id=XjYgR6gbCEc)
- [MissModal — Missing Modality in MSA (TACL)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00628/118797/)
- [LVAT — Latent Space Virtual Adversarial Training (ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460545.pdf)

---

**Phiên bản:** 1.0 — 05/07/2026
