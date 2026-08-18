# Snapshot GNN: tổ chức dữ liệu train (target và quá khứ) — ví dụ cụ thể

Tài liệu mô tả một **snapshot đồ thị** được tổ chức thế nào để huấn luyện: quá khứ (đầu vào) và
target (đích) sắp xếp ra sao, kèm ví dụ có mốc thời gian cụ thể. Cấu trúc lấy theo code
`baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py`
(`build_pooled_manifest`, `build_masked_graph_manifest`, lớp `GraphSnapshot`).

---

## 1. Một snapshot gồm gì

Một snapshot = **một ngày mục tiêu `target_date` T** + **một split** (train / val / test). Các node là
các mã (từ điển cố định 33 mã). Mỗi node **có mặt** mang:

| Thành phần | Kích thước | Nội dung |
|---|---|---|
| `x_price` | `[22, 5]` | 5 node-feature của **cửa sổ 22 ngày quá khứ** của chính mã đó |
| `x_news`, `news_mask` | `[22, 146]`, `[22]` | news 22 ngày quá khứ + cờ có/không có tin ngày đó |
| `y` (target) | 1 số | Parkinson **variance một ngày** của mã đó tại T |
| `presence` | 0 / 1 | mã không có dữ liệu ngày đó → 0 (bị **mask**, không dùng) |
| `adjacency` | `[N, N]` | cạnh có hướng vol→PK Top-5 (đóng băng trên train), giữ self-loop |

5 node-feature: `pk_daily`, `har_weekly` (TB 5 ngày), `har_monthly` (TB 22 ngày), `market_pk`
(median chéo-mã của √PK tại t), `volume_zscore` (z-score rolling-22 của log khối lượng).

## 2. Cách sắp xếp quá khứ và target (chỉ số thời gian)

Theo `build_pooled_manifest`:
```
cửa sổ (đầu vào) = [start .. start+21]          # 22 ngày quá khứ
origin            = start + 21                    # ngày cuối cửa sổ = gốc dự báo
target_index      = start + 22 + h − 1 = origin + h   # target cách origin đúng h phiên
```
Nghĩa là: **quá khứ = 22 ngày kết thúc tại `origin`; target = giá trị MỘT ngày, cách `origin` đúng `h`
phiên** (h ∈ {1, 5, 10, 22}). Target KHÔNG phải trung bình h ngày.

## 3. Ví dụ cụ thể (TRAIN snapshot, h = 5, seq = 22)

Chọn `target_date` **T = 2015-03-27** (giai đoạn TRAIN của các mã lịch sử dài):
- **origin** = 5 phiên giao dịch trước T ≈ **2015-03-20**.
- **cửa sổ 22 ngày** (đầu vào) ≈ **2015-02-13 → 2015-03-20**.

Các node trong snapshot này:
- **VNM** (present): `x_price` = 5 node-feature của **VNM** trên 22 ngày 2015-02-13→2015-03-20;
  `y` = Parkinson variance của **VNM** tại **2015-03-27**.
- **ACB** (present): cùng cấu trúc, dữ liệu của **ACB**, target ACB tại 2015-03-27.
- **SSB** (niêm yết 2021): `presence = 0` → **masked** (không có dữ liệu 2015; không tính loss, không
  tham gia message-passing; không bịa/không dời ngày).
- `adjacency`: các cạnh vol→PK giữa các mã present; GAT đọc **`node_raw` = feature tại origin
  2015-03-20** (ngày cuối cửa sổ) để chú ý chéo-mã.

Mốc thời gian (mọi node CÙNG một ngày T; mỗi mã dùng lịch sử của chính nó):
```
2015-02-13 ............. 2015-03-20 |  ← 5 phiên →  | 2015-03-27
   |____ cửa sổ 22 ngày (đầu vào) ____| origin        |  = target T
   VNM: 22 ngày của VNM  ─────────────► dự báo PK(VNM, T)
   ACB: 22 ngày của ACB  ─────────────► dự báo PK(ACB, T)
   SSB: (chưa niêm yết) → presence = 0, masked
```

## 4. Huấn luyện trên snapshot

- 1 snapshot = **1 bước gradient** (batch = 1 đồ thị). Model chạy 3 nhánh (LSTM giá, GAT trên đồ thị,
  news có gate) → dự báo `y` cho **mỗi node present** → tính **MSE** giữa `(dự báo − y_chuẩn_hoá)`
  **chỉ trên các node present** → lan truyền ngược.
- Mỗi epoch duyệt **tất cả snapshot train** theo thứ tự xáo trộn (RNG theo seed).
- Snapshot train **chỉ chứa node thuộc train của chính mã** (nhóm bên trong từng split) → **không rò rỉ**;
  mọi node trong snapshot **cùng một ngày T** → **không lệch ngày chéo-mã**.

## 5. Khác biệt với hai nhánh còn lại

- **LSTM và News**: cùng cửa sổ 22 ngày / cùng target như trên, nhưng xử lý **từng mã độc lập** (mỗi mẫu
  = 1 mã); "pooled" chỉ là gộp mọi mẫu `(mã, ngày)` vào một tập train.
- **GAT (graph)**: đọc **feature thô tại origin** (ngày cuối cửa sổ) làm biểu diễn node, rồi trộn
  chéo-mã theo cạnh vol→PK bằng trọng số chú ý học được.

*Tham chiếu code:* `data.py` (`build_pooled_manifest`, `build_masked_graph_manifest`, `GraphSnapshot`);
`gat.py` (multi-head GAT); `model.py` (`node_raw = price[:, :, -1, :]`, ghép 3 nhánh + head + sàn dương).
Xem thêm `data_organization_example.md` (chia train/val/test theo từng mã, ví dụ ngày thật).
