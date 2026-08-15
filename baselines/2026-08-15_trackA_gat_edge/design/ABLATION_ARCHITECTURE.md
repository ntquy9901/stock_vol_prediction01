# Kiến trúc model Track-A GAT + bảng ablation ladder (tài liệu review)

> Tài liệu tổng hợp cho baseline `baselines/2026-08-15_trackA_gat_edge`, dùng để review kiến trúc
> model (`code/model.py`, `code/gat.py`) và cấu hình ablation ladder (`code/run_ablation.py`).
> Không sửa code; đây là tài liệu mô tả lại — mọi chiều/tham số lấy trực tiếp từ code, không suy
> đoán. File `design/ARCHITECTURE.md` gốc mô tả kiến trúc ở mức thiết kế ban đầu; file này bổ
> sung: (a) sơ đồ đầy đủ hơn kèm chú thích cơ chế graph-on/off tại từng rung, (b) ghi chú chi tiết
> từng node feature, (c) bảng ablation ladder 5 rung đúng định nghĩa hiện có trong
> `run_ablation.py`, (d) ma trận thành phần × rung.

## 1. Sơ đồ kiến trúc đầy đủ

```
INPUT
 ├─ Price node features: [B, N=VN30, T=22 ngày, 5]
 │     thứ tự cột: [pk_daily, har_weekly, har_monthly, market_pk, volume_zscore_20]
 └─ News panel:          [B, N, T=22 ngày, 146]  (PhoBERT/PCA/EWMA, cắt nhân quả per-mã)
                                    +  news_mask [B, N, T]  (1 nếu có tin tại t, 0 nếu không)
                                    +  ticker_ids [B, N]  (id per-mã, tra gate + scaler)
                                    +  adjacency  [B, N, N]  (edge directed vol→PK Top-5, train-only, đóng băng)

┌────────────────────────────────────────────┐        ┌──────────────────────────────────────────┐
│  NHÁNH GIÁ (price_lstm, chia sẻ trọng số)   │        │  NHÁNH NEWS (news_proj + news_lstm)       │
│  reshape [B*N, T, 5]                        │        │  news_masked = news ⊙ news_mask           │
│  LSTM(input=5, hidden=64, layers=2,         │        │  reshape [B*N, T, 146]                    │
│       dropout=0.2, batch_first=True)        │        │  proj = ReLU(Linear(146→64))              │
│  lấy hidden state cuối (t=T)                │        │  LSTM(input=64, hidden=64, layers=2,      │
│  → h_lstm  [B, N, 64]                       │        │       dropout=0.2)                        │
└──────────────────┬───────────────────────────┘        │  lấy hidden cuối                          │
                    │ node rep = h_lstm                  │  → news_hidden  [B, N, 64]                │
                    │                                     └───────────────────┬────────────────────┘
                    │                                                         │
     ┌──────────────┴───────────────────────────────┐         gate_logits ∈ ℝ^num_tickers (nn.Parameter,
     │  NHÁNH GAT SPATIAL (tự cài, không PyG)        │         KHÔNG phụ thuộc input, học per-mã)
     │  gat1: GATLayer(in=64,  out=64, heads=4)      │         gate = sigmoid(gate_logits[ticker_ids])
     │        64 → 4×64 = 256                        │            nếu use_gate=True, ngược lại gate = 1.0
     │  gat2: GATLayer(in=256, out=64, heads=4)      │         gated_news = gate ⊙ news_hidden   [B,N,64]
     │        256 → 4×64 = 256                       │            (nếu use_news=False: gated_news = 0)
     │  mỗi layer, mỗi head:                         └────────────────────────┬──────────────────────┘
     │   e_ij = LeakyReLU(a_dstᵀ·Wh_i + a_srcᵀ·Wh_j)                          │
     │   α_ij = softmax_j(e_ij) trên các cạnh adjacency>0 (mask -inf chỗ      │
     │          không có cạnh; softmax theo trục nguồn j)                    │
     │   h_i' = ELU( concat_head Σ_j α_ij · W h_j )                          │
     │  adjacency = adj thật (vol→PK)   nếu apply_graph=True                 │
     │  adjacency = identity (I_N)      nếu apply_graph=False                │
     │    → identity: mỗi hàng chỉ có 1 cạnh (tự-tới-tự) → softmax=1 tại     │
     │      chính nó → h_gnn CHỈ còn là 1 phép biến đổi phi tuyến của         │
     │      h_lstm (không mixing chéo mã), nhưng W/a của GAT vẫn được HỌC     │
     │      và áp dụng — đây KHÔNG phải "bỏ nhánh GAT", mà là "tắt lan        │
     │      truyền chéo mã trên cùng nhánh GAT"                              │
     │  → h_gnn  [B, N, 256]                                                 │
     └──────────────────┬─────────────────────────────────────────────────────┘
                         │
                         └── concat( h_lstm[64] , h_gnn[256] ) = har_embed  [B, N, 320]
                                              │
                    concat( har_embed[320] , gated_news[64] ) = h  [B, N, 384]
                                              │
        Head per-mã: Linear(384 → 64) → ReLU → Dropout(0.2) → Linear(64 → 1)
                                              │  raw  [B, N]
                    denorm = raw · scaler_std[ticker_ids] + scaler_mean[ticker_ids]
                    floored = eps·softplus(denorm / eps) + eps      (eps = 1e-6, positivity floor)
                    output  = (floored − mean) / std                (trả về trên thang chuẩn hoá)
                                              │
                    pred [B, N] = Parkinson VARIANCE dự báo h-ngày tới (h ∈ {1,5,10,22})
```

Tham chiếu code: `TrackAGatModel.forward()` (`code/model.py:47-68`), `GATLayer.forward()`
(`code/gat.py:20-32`). Positivity floor `eps·softplus(x/eps)+eps` với `eps=1e-6` giống nhau ở mọi
rung (đồng nhất để so QLIKE công bằng — bài học H2 ghi trong MEMORY dự án).

### Cờ điều khiển kiến trúc (đọc từ `run_ablation.py::_train` / `TrackAGatModel.__init__`)
- `use_news: bool` — bật/tắt toàn bộ nhánh news ở forward (`False` → `gated_news` = tensor 0, các
  module `news_proj`/`news_lstm`/`gate_logits` vẫn tồn tại trong model nhưng không được gọi).
- `use_gate: bool` — chỉ có ý nghĩa khi `use_news=True`; `False` → `gated_news = 1.0 · news_hidden`
  (news cộng thẳng, không qua gate per-mã); `True` → nhân với `sigmoid(gate_logits[ticker_ids])`.
- `apply_graph: bool` (tham số của `forward()`, KHÔNG phải của `__init__`) — chọn adjacency thật
  (vol→PK) hay identity cho cả `gat1` và `gat2`. Vì đây là tham số forward-time, **một checkpoint
  đã train có thể được đọc ở cả hai chế độ graph-on/graph-off** mà không cần train lại — đây chính
  là cơ chế NODE→GNN dùng chung 1 checkpoint.

## 2. Node features (5) — định nghĩa chi tiết

Nguồn: `baselines/2026-08-11_eda_gnn_baseline/code/features.py` (hàm `market_pk_series`,
`volume_zscore_series`, `ExtendedTickerPreprocessor._stack`) và `_har_features()`
(`baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/scaling.py:199-207`). Thứ tự cột cố
định, đúng thứ tự nested của ablation ladder gốc (E0→E1→E2):

| # | Tên cột | Công thức | Nhân quả? | Lý do chọn |
|---|---|---|---|---|
| 1 | `pk_daily` (= `parkinson_volatility`) | Parkinson variance ngày `t` (đã clip ±3σ trên TRAIN) | Dùng giá trị TẠI `t` (contemporaneous với input window, không rò rỉ target `t+h`) | Đặc trưng HAR gốc — mức biến động tức thời |
| 2 | `har_weekly` | `rolling(5).mean()` của `pk_daily` (đã clip), tại `t` | Trailing window (chỉ dùng `t-4..t`) | HAR chuẩn — trung bình tuần |
| 3 | `har_monthly` | `rolling(22).mean()` của `pk_daily` (đã clip), tại `t` | Trailing window (chỉ dùng `t-21..t`) | HAR chuẩn — trung bình tháng, cũng là ràng buộc warm-up chặt nhất (quyết định tập quan sát hợp lệ) |
| 4 | `market_pk` | Median chéo-mã của `sqrt(parkinson_volatility)` tại `t`, tính trên toàn bộ mã có mặt (`market_pk_series()`) | Contemporaneous tại `t`, không dùng thông tin tương lai | Nhân tố thị trường — bộ node feature E2 dùng cột này đã **DM-thắng HAR trên QLIKE** (p=0.012, xem `docs/reports/2026-08-11..._eda_gnn_result` / MEMORY `project_eda_gnn_result`) |
| 5 | `volume_zscore_20` | `(log1p(volume) − rolling(20).mean) / rolling(20).std`, trailing, tại `t`; mã không có OHLCV volume (vd LPB) → 0.0 trung tính | Trailing window (`t-19..t`), std==0/NaN → 0.0 (không suy diễn dữ liệu) | Cú sốc khối lượng bất thường — tín hiệu thanh khoản độc lập với HAR |

Ghi chú:
- **Target = Parkinson VARIANCE** (không phải độ lệch chuẩn) tại `t+h`, `h ∈ {1, 5, 10, 22}` — đơn
  vị bẫy đã ghi nhận trong MEMORY dự án (`project_parkinson_target_is_variance`); mọi rung dùng
  cùng định nghĩa target và cùng positivity floor nên so sánh QLIKE hợp lệ.
- Chuẩn hoá: `ArrayStandardizer` fit **trên TRAIN per-mã** rồi áp cho val/test (leakage-safe).
- Bất biến tập quan sát: thêm `market_pk`/`volume_zscore_20` không được làm đổi tập hàng hợp lệ so
  với pipeline HAR(3) gốc — `ExtendedTickerPreprocessor.transform_frame()` assert
  `valid_rows == har_valid`, nếu lệch thì raise lỗi (giữ E0 == HAR P0, so sánh trên cùng basis).

## 3. Bảng ablation ladder (5 rung)

Nguồn: `code/run_ablation.py::run_horizon()` (dòng 64-76). Mỗi rung sâu hơn **thêm đúng 1 thành
phần**; rung nào đổi trọng số cần học thì **train riêng** (checkpoint riêng); rung GNN đọc lại
checkpoint của NODE (không train thêm) vì `apply_graph` chỉ là cờ forward-time.

| Rung | Thêm gì so với rung trước | Train riêng hay tái dùng checkpoint | Cấu hình cờ (`use_news`, `use_gate`, `apply_graph`) | Câu hỏi trả lời |
|---|---|---|---|---|
| **HAR** | — (baseline ngoài, không phải nhánh của model này) | Hồi quy tuyến tính pooled trên 3 HAR feature (`har`/`day/week/month`), tách biệt hoàn toàn khỏi `TrackAGatModel` | — | Điểm neo cổ điển để so sánh mọi rung deep-learning |
| **LSTM** | Nhánh LSTM giá trên 5 node feature | **Train riêng** (`lstm.pt`) | `use_news=False, use_gate=False`; eval `apply_graph=False` | 5 node feature (HAR mở rộng + market_pk + volume_zscore) tự nó đóng góp bao nhiêu so với HAR tuyến tính |
| **NEWS** | + nhánh news (chưa gate per-mã) | **Train riêng** (`news.pt`) | `use_news=True, use_gate=False`; eval `apply_graph=False` | News cộng thẳng (không gate) có thêm giá trị so với LSTM giá thuần không |
| **NODE** | + gate per-mã học được cho news, đọc **graph-off** (nested) | `full.pt` — **train VỚI graph-on** (`apply_graph_train=True`), đọc lại graph-off | `use_news=True, use_gate=True`; TRAIN `apply_graph=True`, EVAL `apply_graph=False` | Gate per-mã có thêm giá trị không (lưu ý: NODE là `full.pt` train-với-graph đọc graph-off — combo P3-style) |
| **GNN** | + lan truyền chéo mã qua GAT trên đồ thị directed vol→PK Top-5 (đọc **graph-on**) | **Cùng `full.pt`** — đây là ĐÚNG cấu hình train (train+eval đều graph-on) | `use_news=True, use_gate=True`; TRAIN `apply_graph=True`, EVAL `apply_graph=True` | Câu hỏi chính: graph (message-passing chéo mã) có thêm giá trị OOS không (GNN vs NODE, nested cùng trọng số — combo G1-vs-P3 style) |

Hai phép so sánh chính đọc trực tiếp từ `ladder_metrics.json`:
- **`graph_effect` = QLIKE(GNN, test) − QLIKE(NODE, test)`** (đã tính sẵn trong `run_horizon()`,
  dòng 78) — nested so sánh sạch, cùng 1 bộ trọng số, chỉ khác adjacency lúc forward.
- **Mọi rung deep-learning vs HAR** — dùng Diebold-Mariano (QLIKE + squared-error, hiệu chỉnh
  HLN) theo đúng basis (không thực hiện trong `run_ablation.py`, chạy riêng ở bước phân tích).

## 4. Ma trận thành phần × rung

| Rung | LSTM giá (5 feature) | Nhánh news | Gate per-mã | GAT lan truyền chéo mã (adjacency vol→PK) |
|---|:---:|:---:|:---:|:---:|
| HAR | ✗ (hồi quy tuyến tính, không phải LSTM) | ✗ | ✗ | ✗ |
| LSTM | ✓ | ✗ | ✗ | ✗ (identity — GAT layer vẫn chạy nhưng chỉ self-attend) |
| NEWS | ✓ | ✓ (không gate) | ✗ | ✗ (identity) |
| NODE | ✓ | ✓ | ✓ | ✗ (identity — "graph-off") |
| GNN | ✓ | ✓ | ✓ | ✓ (adjacency thật — "graph-on") |

Ghi chú bảng: ở các rung LSTM/NEWS/NODE, hai lớp `GATLayer` trong model vẫn được gọi (không bị
loại bỏ khỏi kiến trúc) nhưng với `adjacency = identity`, nên về mặt chức năng KHÔNG có lan truyền
thông tin chéo mã — cột "GAT lan truyền chéo mã" trong bảng phản ánh đúng hiệu ứng chức năng đó,
không phải sự tồn tại của module.

## 5. Basis dữ liệu + horizon

- **Manifest leakage-safe (masked), pooled toàn VN30:** train = 73026 quan sát, val = 14418, test =
  14464 — giống hệt basis HAR/E2 canonical nên Diebold-Mariano so trực tiếp được (không cần khớp
  lại tập quan sát).
- **Horizon:** `h ∈ {1, 5, 10, 22}` ngày, chạy tuần tự trong `run_ablation.py::main()`
  (`horizons=(1,5,10,22)` mặc định).
- **Seed:** 1 seed (mặc định 42) chạy trước để lấy kết quả sơ bộ trên cả 4 horizon; mở rộng sang 3
  seed (42/123/2026) khi cấu hình được chốt (theo lịch trong `design/ARCHITECTURE.md` — mở resume
  15 → +5/+10 epoch nếu cần trước khi nhân seed).
- **Edge vol→PK:** directed, Top-5 nguồn cho mỗi mã đích, ước lượng lead-lag correlation
  `corr(vshock_i(t), sqrt(PK_j(t+1)))` **trên TRAIN**, đóng băng (không cập nhật lại trên val/test).
- **Chống rò rỉ:** chia theo thời gian per-mã; scaler/edge/gate/regime fit TRAIN-only; news cắt
  theo cutoff nhân quả per-mã; positivity floor đồng nhất mọi rung (bài học H2); bất biến một-basis
  được assert trong `features.py` (số quan sát đồ thị == số quan sát pooled).

## 6. Điểm cần review

- Xác nhận cách đọc `graph_effect`: âm nghĩa là graph giúp giảm QLIKE (tốt hơn), dương nghĩa là
  graph làm QLIKE tệ hơn so với graph-off — cần nêu rõ dấu này khi báo cáo kết quả để tránh đọc
  ngược.
- Rung "LSTM"/"NEWS"/"NODE" đều chạy qua 2 lớp GAT với `adjacency=identity` (không phải bỏ hẳn
  nhánh GAT khỏi kiến trúc) — nếu mục tiêu ablation là "không có tham số GAT nào cả" thì rung này
  chưa đúng nghĩa đó; nếu mục tiêu là "không có lan truyền chéo mã" (đúng như định nghĩa hiện tại
  trong `run_ablation.py`) thì cấu hình hiện tại đã khớp.
- `full.pt` được **TRAIN với `apply_graph=True`** (đồ thị vol→PK thật), nên **GNN (đọc graph-on) là
  đúng cấu hình train — GAT ĐÃ học trên đồ thị thật** (train+eval nhất quán). **NODE** là cùng
  `full.pt` nhưng đọc graph-off (nested "gỡ graph residual"), tức NODE có train(on)/eval(off) —
  đúng kiểu combo P3 (= trained-G1 đọc không graph). `graph_effect = QLIKE(GNN) − QLIKE(NODE)` do
  đó đo tác động bật/tắt lan truyền chéo mã trên **cùng một model đã học đồ thị** (nested, combo
  G1-vs-P3), KHÔNG phải "graph chỉ áp lúc suy luận".
- **Lưu ý confound nhỏ cho báo cáo:** LSTM và NEWS train graph-off thuần (train=eval=off), còn NODE
  là `full.pt` train-với-graph đọc-off. Nên bước LSTM→NEWS sạch (đều graph-off), bước NODE→GNN
  nested sạch (cùng trọng số), riêng bước NEWS→NODE (thêm gate) có confound nhỏ vì NODE được train
  cùng đồ thị. Nếu cần tách gate hoàn toàn khỏi graph-training, phải train NODE riêng graph-off —
  đây là đánh đổi có chủ đích để giữ GNN nested-sạch với NODE (theo định nghĩa ladder user chọn).
- Dấu `graph_effect`: ÂM = graph giúp giảm QLIKE (tốt hơn); DƯƠNG = graph làm QLIKE tệ hơn.
