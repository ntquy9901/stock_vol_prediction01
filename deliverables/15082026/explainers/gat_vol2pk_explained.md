# Giải thích: "multi-head GAT trên cạnh có hướng volume→volatility (vol→PK) lead-lag"

Tài liệu giải thích ý nghĩa và **cách tính** của cụm từ *"a real multi-head Graph Attention Network
(GAT) over a directed volume-to-volatility (vol→PK) lead-lag edge"*. Công thức lấy theo code:
`baselines/2026-08-15_trackA_gat_edge/code/gat.py` (GAT) và
`baselines/2026-08-11_eda_gnn_baseline/code/edges.py` (cạnh vol→PK).

Cụm từ gồm 3 phần: (A) **cạnh có hướng vol→PK lead-lag** (đồ thị nối mã nào với mã nào), (B) **GAT
đa đầu (multi-head)** (cách lan truyền thông tin trên đồ thị đó), (C) chữ **"real"** (GAT tự viết đầy đủ,
không phải xấp xỉ).

---

## A. Cạnh có hướng volume→volatility (vol→PK) lead-lag

**Ý tưởng:** cú sốc **khối lượng** của mã nguồn `i` hôm nay có thể **báo trước** biến động (Parkinson)
của mã đích `j` **ngày mai**. Đây là quan hệ **có hướng** (i → j) và **dẫn-trễ** (lead-lag, lệch 1 ngày),
khác với tương quan **đồng thời** (contemporaneous) vốn chủ yếu phản ánh nhân tố thị trường chung.

**Cách tính (ước lượng trên TRAIN, rồi đóng băng):**

1. Với mỗi mã, lấy 2 chuỗi trên các ngày **train**:
   - `volume_shock_i(t)` = z-score rolling-22 của `log(1+volume_i(t))` (cú sốc khối lượng bất thường).
   - `√PK_j(t)` = căn bậc hai của Parkinson variance của mã `j`.
2. Tính **tương quan dẫn-trễ 1 ngày** giữa cú sốc khối lượng của `i` (tại `t`) và biến động của `j`
   (tại `t+1`):
   $$\rho_{i\to j} \;=\; \operatorname{corr}\big(\text{volume\_shock}_i(t),\ \sqrt{PK_j}(t+1)\big),$$
   chỉ trên các ngày train chung của cặp `(i, j)`.
3. Với **mỗi mã đích `j`**, chọn **Top-5 mã nguồn `i`** có $|\rho_{i\to j}|$ lớn nhất → đó là 5 cạnh
   đi vào `j`. Ma trận kề `A` là **có hướng** (i → j), **không đối xứng**, và giữ **self-loop** (mỗi
   node nối chính nó).
4. `A` được **đóng băng** (frozen): dùng y nguyên cho validation/test (không ước lượng lại) → **không
   leakage**. Mã không có chuỗi khối lượng (vd LPB) không làm nguồn.

**Ví dụ:** nếu trên train, cú sốc khối lượng của HPG hôm nay tương quan mạnh với biến động của HSG ngày
mai, thì có cạnh **HPG → HSG**; khi dự báo HSG, mô hình được "nhìn" thông tin của HPG qua cạnh này.

> Vì sao chọn cạnh này thay vì k-NN tương quan trên PK: tương quan PK–PK **đồng thời** ≈ nhân tố thị
> trường chung (đã được HAR bắt gián tiếp), không có tính **dự báo**; còn vol→PK là **có hướng + dẫn
> trễ** nên về lý thuyết mang thông tin dùng được ngoài mẫu.

---

## B. Mạng chú ý đồ thị đa đầu (multi-head GAT)

GAT lan truyền thông tin giữa các mã **theo trọng số chú ý (attention)** học được, chỉ dọc các cạnh của
`A`. Trong mô hình này, **đầu vào node là feature THÔ tại ngày cuối** `node_raw = price[:, :, -1, :]`
(5 feature), **không phải** trạng thái ẩn của LSTM.

Ký hiệu: `h_i` = vector feature của node `i`; `H` = số đầu (heads) = 4; `W` = ma trận biến đổi tuyến
tính (mỗi head). Một lớp GAT tính như sau (theo `gat.py`):

1. **Biến đổi tuyến tính:** $z_i = W h_i$ (chiếu mỗi node sang không gian ẩn, mỗi head riêng).
2. **Điểm chú ý thô** giữa đích `i` và nguồn `j`:
   $$e_{ij} \;=\; \text{LeakyReLU}\big(a_{\text{dst}}^\top z_i \;+\; a_{\text{src}}^\top z_j\big),$$
   với $a_{\text{dst}}, a_{\text{src}}$ là vector chú ý học được (mỗi head).
3. **Che theo đồ thị:** chỉ giữ $e_{ij}$ khi `A[i,j] > 0` (có cạnh `j → i`); còn lại đặt $-\infty$.
4. **Chuẩn hoá softmax trên các NGUỒN `j`:**
   $$\alpha_{ij} \;=\; \frac{\exp(e_{ij})}{\sum_{k:\,A[i,k]>0}\exp(e_{ik})}.$$
   (Node cô lập → cả hàng $-\infty$ → `nan_to_num` đưa về 0.)
5. **Tổng hợp có trọng số + phi tuyến:**
   $$h_i' \;=\; \text{ELU}\Big(\sum_{j} \alpha_{ij}\, z_j\Big).$$
6. **Đa đầu (multi-head):** làm song song `H=4` head rồi **nối (concat)** kết quả.

**Kích thước thực tế (2 lớp):**
- Lớp 1: `GATLayer(5 → 64, heads=4)` → concat = `[B, N, 256]`.
- Lớp 2: `GATLayer(256 → 64, heads=4)` → `h_gnn = [B, N, 256]`.

Ý nghĩa: mỗi mã đích `j` **tổng hợp có chọn lọc** (qua trọng số $\alpha$) feature khối lượng/biến động
của Top-5 mã nguồn nối tới nó — tức "học xem nên nghe mã nguồn nào nhiều hơn".

---

## C. "real" GAT nghĩa là gì

"real" = **GAT đa đầu tự viết đầy đủ** kiểu Veličković (có ma trận `W`, vector chú ý `a`, softmax theo
cạnh, ELU, multi-head) — **không** phải một xấp xỉ tuyến tính hay lớp giả. Do repo không cài
`torch_geometric`, lớp GAT được hiện thực trực tiếp bằng PyTorch trong `gat.py`.

## D. Ghép vào mô hình

$$h_{\text{gnn}} = \text{GAT}_2\big(\text{GAT}_1(\text{node\_raw},\,A),\,A\big),$$
rồi nối 3 nhánh song song: $h = [\,h_{\text{lstm}}(64)\ \|\ h_{\text{gnn}}(256)\ \|\ \text{gated\_news}(64)\,]
\in \mathbb{R}^{384}$, đưa qua head (Linear→ReLU→Dropout→Linear) + sàn dương (softplus) ra dự báo
Parkinson tại `t+h`. Biến thể **minus_graph** = bỏ hẳn nhánh GAT này (head còn 128).

## E. Trực giác và kết quả thực nghiệm

Cạnh vol→PK cho GAT **cơ hội tốt nhất** để bắt spillover có hướng (đầu vào thô chứa thẳng
`volume_zscore`, đúng đại lượng cạnh dựa vào). Tuy nhiên, đo trên test (Diebold-Mariano đa-metric,
3 seed): **bỏ nhánh GAT không làm mô hình tệ đi có ý nghĩa trên MSE/RMSE/R²** ở hầu hết horizon, và
trên QLIKE việc bỏ graph còn **giảm** QLIKE ở h1/h22 — tức graph **không thêm giá trị ngoài mẫu ổn định**.
Kết quả này nhất quán với phân tích EDA trước đó (nhân tố thị trường chung chi phối đồng biến động
chéo-mã) và với văn liệu (GNNHAR trên DJIA-30: spillover đa-bước không cho lợi thế rõ dưới Model
Confidence Set).

---

*Tham chiếu code:* `gat.py` (lớp `GATLayer`), `model.py` (ghép nhánh + head + sàn dương),
`edges.py` (`build_vol2pk_adjacency`: cạnh vol→PK Top-5 train-only, đóng băng).
