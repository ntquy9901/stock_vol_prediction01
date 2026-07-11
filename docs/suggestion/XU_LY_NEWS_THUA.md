# Xử lý dữ liệu tin tức thưa (Sparse News Data) cho dự báo độ biến động cổ phiếu

## Vấn đề
Trung bình mỗi cổ phiếu chỉ có 1-3 tin/ngày, đa số ngày không có tin (đặc biệt là cổ phiếu vốn hóa nhỏ). Làm thế nào để tận dụng sentiment khi news rất thưa?

---

## 1. Sentiment Decay State

Ý tưởng: Sentiment phai dần theo thời gian nếu không có tin mới.

```
t0: news → s = +0.8
t1: none → s = +0.8 × 0.9 = +0.72
t2: none → s = +0.72 × 0.9 = +0.648
t3: news → s = FinBERT(new_news)
```

```python
s_t = mask_t * FinBERT(news_t) + (1 - mask_t) * s_{t-1} * decay
```

- `decay_rate` học được hoặc fixed (VD: 0.9)
- Kèm binary mask feature `[1/0]` cho model biết có/không news

## 2. Rolling Window Aggregation

Lấy sentiment trung bình trong N ngày gần nhất.

```python
s_t = mean(FinBERT(news_{t-N+1}), ..., FinBERT(news_t))
```

- Neutral sentiment (0.0) nếu window không có news
- Kèm mask token

## 3. Sector/Peer Propagation (Novelty cao)

Sentiment từ cổ phiếu cùng ngành lây lan sang cổ phiếu không có tin.

```python
s_it = α · s_i(t-1) · γ + (1-α) · mean(s_jt for j ∈ sector(i))
```

- Tận dụng graph industry có sẵn
- α và γ learnable

## 4. Attention over Sparse Events

Dùng learned attention để gộp K events gần nhất.

```python
events = [e_1, ..., e_K]  # buffer K events gần nhất
weights = softmax(MLP(events))
s = Σ(weights · events)
```

## So sánh

| Method | Code | Novelty | Hiệu quả |
|--------|------|---------|----------|
| Decay State | 5 dòng | TB | Tốt |
| Rolling Window | 3 dòng | Thấp | TB |
| Sector Propagation | 20 dòng | Cao | Tốt |
| Attention Buffer | 30 dòng | Cao | Rất tốt |

---

## Khuyến nghị: Decay State + Sector Propagation

Kết hợp (1) sentiment decay theo thời gian + (2) propagation từ peer cùng ngành → Module "Sentiment Propagation" duy nhất, vừa xử lý sparsity vừa tận dụng graph structure.

Đầu vào model: concat(s_it, mask_it) — sentiment scalar + binary flag để model biết tin thật hay ước lượng.
