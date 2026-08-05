# Design (Plan) — Selective News Gate Baseline

**Simplicity Gate:** mask cố định (buffer, không phải `nn.Parameter`) — không học, không thêm
abstraction (không cần 1 module "gate" riêng, chỉ 1 tensor 0/1 nhân trực tiếp).
**Anti-Abstraction Gate:** tái dùng thẳng `DualGroupNewsBaseline`/`NewsFeatureLSTM`
(`2026-07-25_dual_group_news_embedding_baseline/code/model_dual_news.py`, đọc read-only) — subclass
thay vì viết lại.

## 1. Data flow

```
data/features/dual_group_news_panel.parquet   (CÓ SẴN — không rebuild)
                    │
   dataset_dual_news.MultiStockDatasetWithDualNews  (sibling, KHÔNG sửa, import read-only)
                    │
   x_har[B,22,30,3]  adj[B,30,30]  x_news[B,22,30,146]  y[B,30]
                    │
   SelectiveGateNewsBaseline (subclass DualGroupNewsBaseline)
   h_lstm,h_gnn = har.get_embeddings(x_har,adj)        # tái dùng nguyên, không đổi
   news_rep     = news_branch(x_news)                   # tái dùng nguyên NewsFeatureLSTM
   news_rep_masked = news_rep * stock_mask.view(1,S,1)   # MỚI — mask cố định theo ticker index
   h = concat([h_lstm, h_gnn, news_rep_masked])
   pred = fusion(h)                                      # tái dùng nguyên MLP
```

`stock_mask` là buffer `[S]` (0/1 float), xây từ danh sách 22 mã NEWS-ON so khớp với
`stock_names` THẬT của dataset (không giả định thứ tự cố định — `common_stocks = sorted(set(...))`
trong `create_dual_news_dataloaders`, lấy đúng list này khi build model).

## 2. File list

| File | Trách nhiệm |
|---|---|
| `code/model_selective_gate.py` | `SelectiveGateNewsBaseline(DualGroupNewsBaseline)` — override `forward()` để mask `news_rep`; `NEWS_ON_TICKERS`/`NEWS_OFF_TICKERS` constants (từ requirements.md §2); `build_stock_mask(stock_names)` helper |
| `code/train_selective_gate.py` | Train loop — tái dùng `create_dual_news_dataloaders` (sibling, read-only), thêm breakdown per-stock DirAcc theo nhóm NEWS-ON/NEWS-OFF |
| `test/test_mask_correctness.py` | Test then chốt: mask=0 cho 1 mã → xáo trộn `x_news` của MÃ ĐÓ không đổi `pred` của mã đó (numerically exact, không xấp xỉ); mask=1 cho mã khác → xáo trộn CÓ đổi `pred` |
| `test/test_model_smoke.py` | Forward/backward, shape, không NaN (như baseline gốc) |

## 3. Per-stock breakdown (bổ sung so với train_dual_news.py gốc)

`validate()` tái dùng logic denormalize hiện có, thêm bước: sau khi có `preds_d`/`targs_d` reshape
`[n_windows, n_stocks]`, tính DirAcc riêng cho tập con cột ứng với NEWS-ON vs NEWS-OFF, in ra
console + lưu JSON — phục vụ đúng success criterion "kỳ vọng NEWS-ON cải thiện, NEWS-OFF không đổi".

## 4. Isolation

Import read-only: `dataset_dual_news.py`, `model_dual_news.py` (từ baseline sibling
`2026-07-25_dual_group_news_embedding_baseline/code/`, thêm vào `sys.path`). KHÔNG sửa file nào ở
đó. KHÔNG rebuild panel. Output: `results/selective_gate_<timestamp>/`, `models/selective_gate_<timestamp>/`.

## 5. Risk

- Mask cứng dựa trên EDA từ model KHÁC (HGB/XGBoost) — có thể không khớp với cách
  `NewsFeatureLSTM` thực sự dùng tín hiệu. Nếu kết quả NEWS-ON không cải thiện rõ so với baseline
  không mask, đây là finding hợp lệ (bác bỏ giả thuyết), không phải bug.
- SHB đã loại theo yêu cầu user — không cần xử lý thêm trong baseline này.
