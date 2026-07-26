# Design (Plan) — Per-Ticker Isolated-Gradient News Gate

## 1. Kiến trúc

```
DualGroupNewsBaseline (sibling, KHÔNG đổi):
  har_embed = concat(h_lstm, h_gnn)         # [B,S,320], từ ParallelLSTMGNN.get_embeddings
  news_rep  = NewsFeatureLSTM(x_news)        # [B,S,64]
  h = concat(har_embed, news_rep)            # [B,S,384]
  pred = fusion(h).squeeze(-1)               # [B,S]

PerTickerGatedNewsBaseline (MỚI, kế thừa/copy cấu trúc trên, CHỈ thêm gate):
  har_embed = concat(h_lstm, h_gnn)                       # [B,S,320] — GIỐNG HỆT
  news_rep  = NewsFeatureLSTM(x_news)                      # [B,S,64]  — GIỐNG HỆT
  gate = sigmoid(gate_logits)                               # [S] — MỚI, 1 số/mã, KHÔNG phụ thuộc input
  gated_news = gate.view(1, S, 1) * news_rep                # [B,S,64] — nhân theo-mã
  h = concat(har_embed, gated_news)                          # [B,S,384]
  pred = fusion(h).squeeze(-1)                               # [B,S]  — fusion là Linear áp theo
                                                              #          từng (b,s) ĐỘC LẬP (không
                                                              #          trộn giữa các mã)
```

**Vì sao gradient của `gate_logits[i]` cô lập hoàn toàn với mã khác (chứng minh, không chỉ khẳng
định):**

1. `news_rep[:, i, :]` chỉ phụ thuộc `x_news[:, :, i, :]` (tin của MÃ i) — `NewsFeatureLSTM`
   reshape `[B,T,S,F] -> [B*S,T,F]` rồi chạy LSTM theo batch `B*S`, không có phép trộn giữa các
   `s` khác nhau ở bước này.
2. `gate_logits[i]` chỉ nhân vào `news_rep[:, i, :]` (qua `gate.view(1,S,1)` broadcast
   elementwise theo đúng vị trí `s=i`) — không đụng tới `news_rep[:, j, :]` với `j≠i`.
3. `self.fusion` là `nn.Sequential(nn.Linear, ...)` áp lên chiều cuối (feature dim) của tensor
   `[B,S,384]` — PyTorch's `nn.Linear` broadcast áp dụng ĐỘC LẬP theo từng vị trí `(b,s)`, không
   có tầng nào trộn thông tin giữa các `s` khác nhau (khác với GAT — nhưng GAT chỉ nằm ở
   `har_embed`, TRƯỚC gate, gate không đi qua GAT).
4. ⇒ `pred[:, i]` là hàm của `(har_embed[:,i,:], gate_logits[i], x_news[:,:,i,:])` **duy nhất**,
   không phụ thuộc `gate_logits[j]`, `x_news[:,:,j,:]`, hay `y[:,j]` với `j≠i`.
5. ⇒ `∂loss/∂gate_logits[i]` (loss = MSE trung bình cả batch) chỉ có 1 thành phần khác 0: đạo hàm
   qua `pred[:,i]` so với `y[:,i]`. Thay đổi `y[:,j]` (j≠i) không làm đổi giá trị này.

**Kiểm chứng bằng test (không chỉ suy luận):** `test_gate_gradient_isolated_per_ticker` — chạy
forward+backward 2 lần với CÙNG input, CHỈ đổi `y[:, j]` (j≠i), so `gate_logits.grad[i]` — phải
BẰNG NHAU giữa 2 lần chạy. Đây là property test trực tiếp cho đúng claim ở §1, không phải test
shape thông thường.

## 2. Learning rate riêng cho gate (không phải speculative tuning)

`gate_logits` là 30 số vô hướng (rất ít tham số, rất "nhẹ" so với hàng chục nghìn tham số của
`fusion`/`NewsFeatureLSTM`/HAR branch). Với LR chung mặc định (5e-3, đã tối ưu cho phần network
lớn), 30 số này có thể gần như không di chuyển trong 10 epoch (cap training policy) — khiến
"quá trình học" mà user muốn quan sát (debug log) không có gì để xem. Dùng optimizer 2 param-group
(Adam hỗ trợ native), `gate_lr` mặc định 0.05 (10x base) — kỹ thuật chuẩn (differential learning
rate), không phải regularization/abstraction thừa (Anti-Abstraction Gate: dùng thẳng
`torch.optim.Adam([{...}, {...}])`, không tự viết optimizer riêng).

## 3. Debug output (yêu cầu user, bắt buộc)

1. **Console mỗi epoch:** bảng 30 mã sắp giảm dần theo `sigmoid(gate_logits)`, in kèm giá trị
   epoch trước để thấy delta (mã nào đang tăng/giảm gate qua epoch).
2. **`gate_history.json`** (trong `results/<run>/`): `{"0": {"AAA": 0.5, ...}, "1": {...}, ...}` —
   ghi sau MỖI epoch (không chỉ epoch cuối) để xem lại toàn bộ quá trình học sau khi train xong.
3. **`gate_evolution_epoch_{N}.png`** mỗi 5 epoch (cùng nhịp với loss learning curve,
   CLAUDE.md §3.C): 1 line chart, 1 đường/mã (30 đường, legend rút gọn hoặc colormap liên tục +
   colorbar để tránh legend quá rối), trục X = epoch, trục Y = gate value (0-1).
4. Loss learning curve mỗi 5 epoch: TÁI DÙNG `plot_learning_curves_with_analysis` — không viết
   lại (Anti-Abstraction Gate).

## 4. Simplicity Gate / Anti-Abstraction Gate

- **Simplicity Gate: PASS.** 1 tham số mới (`gate_logits`, 30 số), không thêm layer/module phức
  tạp. Debug logging dùng `print`/`json.dump`/`matplotlib` built-in, không thêm dependency.
- **Anti-Abstraction Gate: PASS.** `torch.optim.Adam` với 2 param-group là API sẵn có, không tự
  viết wrapper. Tái dùng `plot_learning_curves_with_analysis`, `evaluate_predictions`,
  `EarlyStopping`, `create_dual_news_dataloaders` — không viết lại bất kỳ hàm nào đã có.

## 5. File list

```
baselines/2026-07-26_per_ticker_news_gate_baseline/
├── requirements/requirements.md
├── design/design.md
├── code/
│   ├── __init__.py
│   ├── model_per_ticker_gate.py     # PerTickerGatedNewsBaseline
│   └── train_per_ticker_gate.py     # debug print + gate_history.json + gate-evolution plot
├── code_review/code_review_2026-07-26.md
└── test/
    ├── __init__.py
    ├── test_model_per_ticker_gate.py   # shapes + CORE gradient-isolation property test
    └── test_train_smoke.py             # dataset+train_epoch smoke, gate_history file write
```

## 6. Risks

- **Dying gate (sigmoid saturation):** nếu `theta_i` trôi nhanh về cực trị (±∞), gradient
  `gate_i*(1-gate_i)` → 0, gate "đông cứng" sớm. Debug log (console + `gate_history.json`) đủ để
  PHÁT HIỆN nếu xảy ra (nhìn gate value dừng thay đổi từ epoch nào) — không tự thêm fix trước khi
  có bằng chứng cần (per Out of scope, requirements.md §6).
- **Gate là hằng số theo mã, không theo ngày:** giới hạn có chủ đích (đúng yêu cầu user), không
  phải thiếu sót — nêu rõ trong requirements.md.
- **10 epoch có thể chưa đủ để 30 số hội tụ ổn định** — `gate_lr` cao hơn giúp giảm rủi ro này
  nhưng không loại bỏ hoàn toàn; ghi nhận trung thực trong summary report nếu gate chưa "chốt".
