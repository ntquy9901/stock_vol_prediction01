# Kiến trúc: Track-A-style GAT + node features tốt + edge volume→PK (tham khảo)

> File lưu để tham khảo và dùng lại. Mô tả kiến trúc mô hình dùng trong baseline
> `2026-08-15_trackA_gat_edge`. Mục tiêu: kiểm tra xem một GNN với **GAT thật (attention học
> được)** kiểu Track A, dùng **node features đã thắng HAR** (MarketPK + volume_zscore_20) và
> **edge directed volume→PK Top-5**, có thêm giá trị out-of-sample so với chính nó khi tắt graph
> hay không (graph-on vs graph-off), và so với HAR.

## Sơ đồ

```
INPUT: VN30 (N mã) × 22 ngày × 5 NODE FEATURES            +      VN30 × 22 ngày × 146 news (PhoBERT)
   [pk_daily, har_weekly, har_monthly, MarketPK, volume_zscore_20]                  │
              │                                                                      │
   ┌──────────┴───────────────────────────────────┐                     ┌──────────┴──────────┐
   │  LSTM temporal (shared weights, per-mã)       │                     │ News encoder         │
   │  LSTM(5 → 64, 2 lớp, dropout 0.2)             │                     │ Linear(146→64)+LSTM  │
   │  → h_lstm [B,N,64]  (last hidden)             │                     │ (2 lớp) causal-mask  │
   └──────────┬──────────────────┬─────────────────┘                     │ → news_rep [B,N,64]  │
              │                   │ node rep = h_lstm                     └──────────┬──────────┘
              │            ┌──────┴────────────────────────────┐          gate = sigmoid(gate_logits[mã])
              │            │  GAT spatial (multi-head, THẬT)    │          (1 tham số/mã, KHÔNG phụ
              │            │  2 lớp, 4 head × 64  → 256         │           thuộc input; đóng băng khi
              │            │  đồ thị = directed volume→PK Top-5 │           chuyển từ NODE→GNN)
              │            │  → h_gnn [B,N,256]                 │                     │
              │            │  ┌ GRAPH-OFF: adjacency = identity │          gated_news = gate ⊙ news_rep
              │            │  └ (mỗi node chỉ self-attend →     │                     │
              │            │     KHÔNG lan truyền chéo mã)      │                     │
              │            └──────┬────────────────────────────┘                     │
              └──── concat ───────┴──  h_lstm(64) ⊕ h_gnn(256) = har_embed [B,N,320] ┤
                                                                                     │
                        concat( har_embed 320 , gated_news 64 )  =  h [B,N,384]
                                              │
                    Head per-mã: Linear(384→64) → ReLU → Dropout → Linear(64→1)
                                              │  + positivity floor (eps·softplus)
                                       pred [B,N]  (Parkinson VARIANCE 5-ngày ahead)
```

## Mô tả từng thành phần

### 1. Đầu vào
- **Node features (5):** `pk_daily` (Parkinson variance ngày), `har_weekly` (TB 5 ngày), `har_monthly`
  (TB 22 ngày), `MarketPK` (median chéo-mã của √PK tại t — nhân tố thị trường), `volume_zscore_20`
  (z-score rolling-20 của log1p(volume), nhân quả). Đây là bộ E2 đã **DM-thắng HAR trên QLIKE**
  (p=0.012). Chuẩn hoá per-mã, fit trên TRAIN.
- **News (146):** panel PhoBERT/PCA/EWMA (`dual_group_news_panel.parquet`), gắn theo cutoff nhân quả
  per-mã (chống rò rỉ). Chuỗi 22 ngày.

### 2. Nhánh LSTM temporal (per-mã, chia sẻ trọng số)
`LSTM(input=5, hidden=64, num_layers=2, dropout=0.2)`; lấy hidden cuối → `h_lstm [B,N,64]`. Mỗi mã
xử lý độc lập theo thời gian; đây cũng là node representation cấp cho GAT.

### 3. Nhánh GAT spatial (multi-head, tự cài — không PyG)
- Đầu vào: `h_lstm [B,N,64]` (node rep). Đồ thị: **directed volume→PK Top-5** (mỗi target j nhận
  Top-5 source theo lead-lag `corr(vshock_i(t), √PK_j(t+1))`, ước lượng trên TRAIN, **đóng băng**).
- 2 lớp Graph Attention, **4 head × 64 = 256**. Với mỗi head, hệ số attention
  `α_ij = softmax_j( LeakyReLU( aᵀ[W h_i ‖ W h_j] ) )` chỉ trên các cạnh của adjacency (masked),
  rồi `h_i' = Σ_j α_ij W h_j`. Concat các head → `h_gnn [B,N,256]`. Attention **học được** từ node
  features (khác residual-MP của Track B dùng adjacency cố định làm trọng số).
- **Cơ chế graph-on/off (nested):** graph-off = thay adjacency bằng **identity** (mỗi node chỉ
  self-attend) → `h_gnn` chỉ chứa thông tin bản thân, KHÔNG lan truyền chéo mã. Graph-on = adjacency
  vol→PK. Cùng một checkpoint → so sánh sạch "message-passing chéo mã có giá trị không".

### 4. Nhánh news + gate
- `Linear(146→64) → ReLU → LSTM(64→64, 2 lớp)` với mask nhân quả → `news_rep [B,N,64]` (cô lập
  per-mã như Track A).
- **Gate:** `gate_logits ∈ ℝ^N` là `nn.Parameter`; `gate = sigmoid(gate_logits[ticker_id])` — MỘT
  số vô hướng học được PER MÃ, **không phụ thuộc input**. `gated_news = gate ⊙ news_rep`.

### 5. Fusion + head
- `har_embed = concat(h_lstm[64], h_gnn[256]) = 320`; `h = concat(har_embed[320], gated_news[64]) =
  384`.
- Head per-mã: `Linear(384→64) → ReLU → Dropout(0.2) → Linear(64→1)`; positivity floor
  `eps·softplus(raw/eps)+eps`, eps=1e-6 (đồng nhất với các rung khác để QLIKE so sánh công bằng — bài
  học H2).
- Output: `pred [B,N]` = Parkinson variance 5-ngày ahead.

## Ablation (cùng checkpoint) — trả lời "GNN có helpful không"
| Rung | Adjacency | Ý nghĩa |
|---|---|---|
| **HAR** | — | baseline ngoài (hồi quy tuyến tính 3 HAR feature) |
| **NODE** (graph-off) | identity | node features (5) + news, KHÔNG lan truyền chéo mã |
| **GNN** (graph-on) | vol→PK Top-5 | GAT lan truyền chéo mã |

- **GNN vs NODE** = graph có thêm giá trị không (câu hỏi chính, nested trên cùng checkpoint).
- **GNN/NODE vs HAR** = Diebold-Mariano (QLIKE + squared-error, HLN, h=5).

## Basis dữ liệu (dùng chung để so sánh công bằng)
Dùng masked manifest leakage-safe của pipeline hiện có: **train 73026 / val 14418 / test 14464** —
giống hệt HAR/E2 canonical, nên DM so trực tiếp được. Node features (5) + edge vol→PK tái dùng code
từ `combo`/`eda_gnn`. Bất biến một-basis được assert (obs của graph == obs pooled).

## Huấn luyện + cơ chế RESUME
- MSE loss (train), Adam (weight_decay 1e-5), grad-clip 1.0, dropout 0.2.
- **Checkpoint:** sau khi train, lưu `{model_state, optimizer_state, epoch, best_val_loss,
  best_state, rng_state}` → `models/trackA_gat_seed{seed}.pt`.
- **Resume:** load checkpoint, tiếp tục train thêm `--extra-epochs` (optimizer + epoch counter +
  best-val được khôi phục), lưu lại. Cho phép: 15 epochs → xem → resume +5/10 → xem → quyết định.
- **Lịch:** 1 seed × 15 epochs trước (báo cáo val metrics) → resume nếu cần → 3 seeds (42/123/2026)
  khi chốt cấu hình. (>10 epochs: user đã đồng ý rõ 2026-08-15.)

## Chống rò rỉ
Per-mã chronological split; scaler/edge/gate/regime fit TRAIN-only; edge vol→PK đóng băng theo train
per-mã; news cutoff nhân quả; positivity floor đồng nhất mọi rung; bất biến một-basis assert.

## Kỳ vọng (trung thực)
Node features sẽ thắng HAR (đã biết). Câu hỏi thực là **GNN (graph-on) > NODE (graph-off)** không —
mọi bằng chứng trước (combo DM null, edge-discovery linear+nonlinear STOP) nghiêng mạnh về **null**.
GAT-attention là cơ chế GNN duy nhất chưa chạy (các test trước dùng residual-MP adjacency cố định),
nên chạy để đóng câu hỏi. Kết quả (thắng/hoà/thua) đều là phát hiện hợp lệ.
```
