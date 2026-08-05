# Ghi chú kỹ thuật — vấn đề công thức DirAcc (phát hiện 01/08/2026)

**Trạng thái:** DirAcc đã được gỡ khỏi `BAO_CAO_TONG_HOP.md` (mọi bảng, mọi kết luận) cho tới khi
công thức được kiểm tra lại. File này giữ nguyên phần phân tích kỹ thuật đầy đủ (công thức, code,
số liệu đối chiếu) để tham khảo khi kiểm tra lại.

**Đang có 2 công thức DirAcc khác nhau trong dự án, cho kết quả rất khác nhau.** Mọi con số DirAcc
"headline" từng xuất hiện trong các báo cáo trước đây (68.42%, 69.51%, 72.35%...) dùng công thức
(A) — công thức (B) mới là con số đo đúng nghĩa "dự báo đúng chiều biến động của 1 mã qua thời
gian".

## 1. Công thức (A) — công thức "headline" dùng trong mọi báo cáo trước đây

**File:** `src/common/evaluation.py`
```python
def directional_accuracy(y_true, y_pred):
    actual_changes = np.sign(np.diff(y_true))   # dấu của (y_true[i+1] - y_true[i])
    pred_changes = np.sign(np.diff(y_pred))     # dấu của (y_pred[i+1] - y_pred[i])
    accuracy = np.mean(actual_changes == pred_changes)
    return accuracy * 100
```
**Phụ thuộc 2 biến đầu vào** (`y_true`, `y_pred` — mảng 1 chiều) nhưng mỗi phép so sánh cụ thể
phụ thuộc **4 số vô hướng**: `y_true[i]`, `y_true[i+1]`, `y_pred[i]`, `y_pred[i+1]`.

## 2. Công thức (B) — đúng theo từng mã, chỉ có ở script per-ticker-gate

```python
# vd baselines/2026-07-26_per_ticker_news_gate_baseline/code/train_per_ticker_gate.py
p2 = preds_d.reshape(n_windows, n_stocks)   # sắp lại đúng theo (thời gian, mã)
t2 = targs_d.reshape(n_windows, n_stocks)
dir_per = [np.mean(np.sign(np.diff(t2[:, s])) == np.sign(np.diff(p2[:, s]))) * 100
           for s in range(n_stocks)]
directional_accuracy_per_stock = np.mean(dir_per)
```

## 3. Vì sao (A) và (B) khác nhau — thứ tự mảng quyết định ý nghĩa phép tính

`y_true`/`y_pred` truyền vào công thức (A) là mảng **đã làm phẳng theo thứ tự `[window, mã]`**
(mọi mã ở cùng 1 ngày liệt kê liên tiếp, rồi mới sang ngày kế tiếp) — xem `validate()`:
```python
preds_n.append(pred.cpu().numpy().reshape(-1))   # [B,S] -> phẳng: mã0,mã1,...,mãS-1, mã0(ngày sau),...
```
→ `np.diff` trên mảng này **chủ yếu so sánh 2 MÃ KHÁC NHAU Ở CÙNG 1 NGÀY** (S-1 trong mỗi S phép
so sánh), chỉ 1/S phép so sánh rơi vào ranh giới 2 ngày — và ngay cả phép đó cũng lệch mã (mã cuối
ngày i vs mã đầu ngày i+1). **Không có phép so sánh nào trong công thức (A) thực sự là "cùng 1 mã,
2 ngày liên tiếp"** trừ khi số mã = 1.

Công thức (B) sửa đúng: `reshape(n_windows, n_stocks)` rồi `np.diff` theo trục THỜI GIAN cho TỪNG
CỘT (từng mã) riêng — đây mới là phép so sánh "cùng mã, ngày kế tiếp" đúng nghĩa.

## 4. Bằng chứng số liệu thật — chênh lệch rất lớn

| Run | DirAcc công thức (A) — "headline" | DirAcc công thức (B) — đúng theo mã |
|---|---:|---:|
| Per-ticker-gate, epoch 20 (5-ngày) | 69.51% | **48.52%** (~ngẫu nhiên) |
| Per-ticker-gate, 1-ngày | 72.39% | **33.16%** (DƯỚI CẢ ngẫu nhiên 50%) |

*(Trích trực tiếp từ `results/per_ticker_gate_2026-08-01_094139/results.json` và
`results/per_ticker_gate_h1_2026-08-01_104140/results.json` — trường `directional_accuracy` vs
`directional_accuracy_per_stock`.)*

## 5. Ý nghĩa

Toàn bộ con số DirAcc 66-74% từng trích dẫn trong các báo cáo trước (Bảng A, B, mọi bảng kết quả)
dùng công thức (A) — **không đo đúng** "model dự báo đúng chiều biến động của 1 mã qua thời gian".
Khả năng cao công thức (A) cho kết quả cao một cách giả tạo vì tận dụng được đồng biến động thị
trường chung (nhiều mã tăng/giảm volatility cùng lúc do yếu tố vĩ mô/thị trường chung — so 2 mã
khác nhau cùng ngày dễ "trùng dấu" hơn ngẫu nhiên thuần tuý, dù không phải do model dự báo đúng cho
từng mã). Con số công thức (B) — thấp, gần hoặc dưới mức ngẫu nhiên — cho thấy khả năng dự báo
ĐÚNG CHIỀU theo thời gian cho từng mã cụ thể **có thể yếu hơn nhiều** so với những gì các bảng so
sánh DirAcc từng thể hiện. Đây là phát hiện chưa được sửa trong code — chỉ mới ghi nhận ở đây.

**Không phải mọi baseline đều có công thức (B):** tất cả script `train_per_ticker_gate*.py` (5,
10, 22, 1-ngày) đều tính `directional_accuracy_per_stock`; các script
`train_har_only_reference*.py` (HAR-only) THÌ KHÔNG — trường này là `None` trong `results.json`
của mọi run HAR-only (vd `har_only_h1_2026-08-01_103548/results.json`), nên hiện tại KHÔNG có cách
đối chiếu công thức (B) cho riêng kiến trúc HAR-only.

## 6. Việc cần làm trước khi đưa DirAcc trở lại báo cáo

1. Xác nhận công thức (B) (`directional_accuracy_per_stock`) là công thức đúng cần dùng thay thế.
2. Thêm tính toán công thức (B) vào các script `train_har_only_reference*.py` (hiện đang thiếu) để
   có đối chiếu ngang giữa HAR-only và gated-news.
3. Retrain/đánh giá lại tất cả biến thể đã so sánh trong `BAO_CAO_TONG_HOP.md` bằng công thức (B),
   rồi mới đưa DirAcc trở lại các bảng so sánh.
