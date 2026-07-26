# Design (Plan) — Directed Volatility-Spillover Graph + QLIKE-Augmented Loss

## 1. Nguồn nghiên cứu (2026-07-26 web search, deep research theo yêu cầu user)

1. Zhang, Pu, Cucuringu & Dong (2025), *"Forecasting realized volatility with spillover effects:
   Perspectives from graph neural networks"*, International Journal of Forecasting 41(1), 377–397
   (https://web.media.mit.edu/~xdong/paper/ijf25.pdf). Thay neighborhood-aggregation TUYẾN TÍNH
   trong GHAR (Graph-HAR) bằng GNN phi tuyến; đồ thị = spillover network (có hướng, mỗi cạnh biểu
   diễn hiệu ứng lan truyền). Kết luận quan trọng: (a) multi-hop neighbor aggregation một mình
   KHÔNG cho lợi thế rõ rệt; (b) mô hình hoá spillover **phi tuyến** cải thiện dự báo, đặc biệt
   horizon ngắn (≤1 tuần); (c) train với **Quasi-likelihood (QLIKE) loss** cải thiện đáng kể so
   với MSE thông thường.
2. Chi et al. (2026), *"Global Stock Market Volatility Forecasting Incorporating Dynamic Graphs
   and All Trading Days"*, Journal of Forecasting (Wiley). Dùng spatial-temporal GNN với đồ thị
   xây từ **volatility spillover index** (kiểu Diebold-Yilmaz) thay vì tương quan giá thô — DCRNN-HAR
   thắng baseline trên toàn bộ 8 thị trường/kịch bản test.
3. (Bối cảnh, không phải nguồn mới) Regime-Dependent TemporalGAT (MDPI, Jan 2026) — cùng họ ý
   tưởng: LSTM (temporal) + GCN/GAT (spatial) trên đồ thị Diebold-Yilmaz, khớp với kiến trúc
   `LSTMGATHybrid` đã có sẵn trong project (`src/lstm_gat_hybrid/model.py`) — xác nhận kiến trúc
   nền hiện tại đã đúng hướng SOTA, đồ thị + loss là 2 điểm cần sửa.

**Kết luận nghiên cứu → 2 thay đổi cụ thể, ít rủi ro, tái dùng tối đa code sẵn có** (xem §2).

## 2. Kiến trúc / Data flow

```
data/features/dual_group_news_panel.parquet   (đã rebuild, 12 nguồn mới — task #1 phiên này)
                    │
                    ▼
dataset_spillover_news.py (copy sửa của dataset_dual_news.py)
  - graph_method='spillover' (MỚI) thay vì 'correlation'/'knn'
  - mỗi sequence window (22 ngày): gọi graph_spillover.construct_directed_spillover_graph(...)
    thay vì graph_correlation.construct_correlation_graph(...) — CÙNG chỗ gọi, cùng data slice
    (sequence_volatility = all_volatility[i:i+seq_length]) → KHÔNG có leakage mới (đồ thị build
    lại mỗi window, y hệt cơ chế cũ, chỉ đổi công thức).
                    │
                    ▼
model_dual_news.DualGroupNewsBaseline (import read-only, KHÔNG đổi)
  - forward(x_har, adj_matrix, x_news) — adj_matrix giờ KHÔNG đối xứng; GAT layer
    (src/lstm_gat_hybrid/model.py dòng ~177-191) đã hỗ trợ sẵn adjacency bất kỳ (mask theo
    adj_matrix_expanded == 0, không giả định đối xứng) → không cần sửa model.
                    │
                    ▼
train_spillover_qlike.py (copy sửa của train_dual_news.py)
  - loss = MSE(pred_norm, y_norm) + qlike_weight * QLIKE(denorm(pred), denorm(y), clamp>0)
  - metrics/report: giữ nguyên 6 metric bắt buộc + val/test comparison table
```

### 2.1 Đồ thị có hướng (directed spillover graph)

**Hàm mới, KHÔNG sửa `graph_correlation.py`:**

```python
def construct_directed_spillover_graph(volatility_window: np.ndarray, k: int = 8) -> np.ndarray:
    """adj[i, j] > 0  <=>  mã j (transmitter, ngày t) lan truyền cú sốc biến động sang
    mã i (receiver, ngày t+1) — top-k cạnh VÀO mạnh nhất mỗi node i, KHÔNG symmetrize."""
```

- Input: `volatility_window` [seq_length, num_stocks] — **CÙNG data slice** dataset đã truyền cho
  đồ thị cũ (`sequence_volatility`), không cần nguồn dữ liệu mới.
- `lead = volatility_window[:-1]` (ngày t), `lag = volatility_window[1:]` (ngày t+1).
- Với mỗi receiver `i`: tính `pearsonr(lead[:, j], lag[:, i])` cho mọi transmitter `j != i`, giữ
  top-k (theo trị tuyệt đối) → `adj[i, j] = abs(corr)`. KHÔNG set `adj[j, i]`.
- Lý do khớp với GAT layer hiện có: `attention_scores[b, i, j, h] = LeakyReLU(a1·h_i + a2·h_j)`,
  softmax theo chiều `j` (dim=2) — tức node `i` (query/receiver) tổng hợp thông tin từ các `j`
  (key/transmitter) mà `adj[i,j] != 0`. Vậy đặt cạnh vào theo `adj[i,j]` là ĐÚNG semantics để `i`
  "nghe" tín hiệu biến động từ `j` đã xảy ra trước đó (lag-1) — không cần sửa `model.py`.
- Degenerate case (zero-variance window, window quá ngắn): trả `adj` toàn 0 (giống hệt behavior
  hiện có của `construct_correlation_graph`/`construct_knn_graph` khi `np.std==0`) — self-loop
  vẫn được `GraphAttentionLayer.forward` tự thêm (`fill_diagonal_(1.0)`, dòng 143 `model.py`), nên
  node vẫn nhận thông tin của chính nó dù không có cạnh nào.

### 2.2 QLIKE-augmented loss

**Vấn đề đã biết (CLAUDE.md §"LSTM-GNN Normalization Failure"): dùng Softplus/activation ép dương
lên OUTPUT của model đã từng làm predictions collapse về 0.** Tránh lặp lại: KHÔNG đổi output
layer (giữ linear, scale chuẩn hoá, đúng pattern đã chứng minh 67.90% Dir Acc). Thay vào đó QLIKE
chỉ dùng làm **thành phần loss phụ**, tính trên giá trị đã **inverse-transform về scale gốc**
(dương tự nhiên vì volatility ≥ 0 trong data thật) và **clamp** để tránh chia-cho-0/log(0):

```python
def combined_loss(pred_norm, y_norm, target_scaler, qlike_weight=0.1, eps=1e-6):
    mse = F.mse_loss(pred_norm, y_norm)
    pred_denorm = target_scaler.inverse_transform_torch(pred_norm)  # affine, differentiable
    y_denorm = target_scaler.inverse_transform_torch(y_norm)
    pred_clamped = torch.clamp(pred_denorm, min=eps)
    y_clamped = torch.clamp(y_denorm, min=eps)
    ratio = y_clamped / pred_clamped
    qlike = (ratio - torch.log(ratio) - 1).mean()
    return mse + qlike_weight * qlike
```

- `target_scaler.inverse_transform_torch`: affine (mean/std), differentiable, không cần rời khỏi
  autograd graph (không như sklearn `.inverse_transform` — cần viết bản torch tương đương, 2 dòng).
- `qlike_weight=0.1` mặc định (chưa tune, §"Out of scope" trong requirements.md) — giữ MSE làm
  loss chính (ổn định huấn luyện đã biết), QLIKE là regularizer nhẹ hướng model theo metric mục
  tiêu thật.
- Early predictions (random init) có thể cho `pred_denorm` gần 0 hoặc âm nhẹ trên scale chuẩn hoá
  → `clamp(min=eps)` chặn NaN/Inf; nếu QLIKE quá lớn ở vài batch đầu, `mse` vẫn dominate loss vì
  `qlike_weight` nhỏ.

## 3. Simplicity Gate / Anti-Abstraction Gate (CLAUDE.md §1.5 bước 4)

- **Simplicity Gate: PASS.** Không thêm project/abstraction mới — 1 hàm graph, 1 hàm loss, tái
  dùng 100% model/dataset-loader/eval pipeline sẵn có. Không thêm config file, không thêm CLI
  framework mới.
- **Anti-Abstraction Gate: PASS.** Dùng thẳng `pearsonr` (scipy, đã dùng sẵn trong
  `graph_correlation.py`), `torch.clamp`/`torch.log` built-in — không tự viết wrapper thừa.
- Không có gate nào bị phá — không cần complexity tracking note.

## 4. File list

```
baselines/2026-07-26_spillover_qlike_baseline/
├── requirements/requirements.md
├── design/design.md                       (this file)
├── code/
│   ├── __init__.py
│   ├── graph_spillover.py                 # construct_directed_spillover_graph()
│   ├── dataset_spillover_news.py          # copy-sửa dataset_dual_news.py (graph_method='spillover')
│   ├── losses.py                          # combined_loss() + TorchTargetScaler helper
│   └── train_spillover_qlike.py           # copy-sửa train_dual_news.py
├── code_review/code_review_2026-07-26.md
└── test/
    ├── __init__.py
    ├── test_graph_spillover.py            # asymmetry, top-k, degenerate window
    ├── test_losses.py                     # combined_loss no-NaN, QLIKE-only sanity vs manual calc
    └── test_train_smoke.py                # dataset+model+train_epoch smoke (dummy data)
```

## 5. Risks

- Lag-1 correlation trên window ngắn (22 ngày → 21 cặp) có phương sai thống kê cao (ít mẫu hơn cả
  đồ thị cùng-ngày vốn đã dùng cùng 22 điểm) — chấp nhận được vì đây là baseline THỬ so sánh 2
  cách xây đồ thị trên CÙNG lượng dữ liệu, không phải đề xuất production.
- QLIKE loss có thể không ổn định nếu model dự báo âm nhiều ở epoch đầu — đã có `clamp` + `qlike_weight`
  nhỏ để giảm rủi ro; theo dõi learning curve (bắt buộc mỗi 5 epoch, CLAUDE.md §3.C) để phát hiện
  sớm nếu QLIKE làm loss instability thay vì cải thiện.
