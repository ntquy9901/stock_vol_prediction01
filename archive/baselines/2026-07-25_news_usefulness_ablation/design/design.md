# Design (Plan) — News Usefulness Ablation

## 1. File list

| File | Trách nhiệm |
|---|---|
| `code/train_har_only_reference.py` | Train `ParallelLSTMGNN` (fusion riêng, không freeze) qua `create_dual_news_dataloaders(news_panel_path=None)` — x_har/adj/y giống hệt dual-group, bỏ qua x_news. Per-ticker MSE/QLIKE/DirAcc → `results/har_only_ablation_ref_<ts>/results.json` |
| `code/eval_checkpoint_per_ticker.py` | Load checkpoint có sẵn (dual-group all-ON, 40ep), chạy per-ticker eval trên test set — KHÔNG train |
| `code/compute_ablation_deltas.py` | Đọc 2 file results.json trên, tính delta từng mã (QLIKE, MSE, DirAcc), in ra danh sách ON/OFF theo ngưỡng |

## 2. Vì sao dùng chung `create_dual_news_dataloaders`

Hàm này xây `train_ds/val_ds/test_ds` từ `_load_raw_stock_data` + `_split_raw_data_by_date` +
`_generate_har_for_split` — HOÀN TOÀN không phụ thuộc `news_panel_path` cho tới bước cuối cùng
(`x_news`). Gọi 2 lần (1 lần `news_panel_path=None`, 1 lần path thật) cho **CÙNG** `common_stocks`
(32 mã, cùng thứ tự `sorted(set(...))`), **CÙNG** `train_end_idx`/`val_end_idx` (deterministic,
không random) → windows/split giữa Model A và Model B thẳng hàng tuyệt đối, so sánh per-ticker
hợp lệ.

## 3. Isolation

Import read-only: `ParallelLSTMGNN`, `LSTMGATConfig` (`src/lstm_gat_hybrid/`, đã dùng ở mọi
baseline khác), `create_dual_news_dataloaders`/`SelectiveGateNewsBaseline` (2 sibling baseline
hôm nay). Không sửa checkpoint cũ, không sửa baseline khác. Output:
`results/har_only_ablation_ref_<ts>/`, KHÔNG tạo model mới cho phần eval-only (dùng checkpoint có sẵn).
