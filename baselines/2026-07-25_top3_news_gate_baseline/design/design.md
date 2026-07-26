# Design (Plan) — Top-3 News Gate Baseline

**Simplicity Gate:** không viết lại mask mechanism — subclass `SelectiveGateNewsBaseline`
(sibling `2026-07-25_selective_news_gate_baseline/code/model_selective_gate.py`), chỉ override
2 hằng số ticker list. **Anti-Abstraction Gate:** không tạo thêm config file/abstraction cho
"danh sách ticker" — giữ nguyên dạng hardcoded set như baseline gốc (nhất quán, đơn giản).

## 1. Data flow

Giống hệt `2026-07-25_selective_news_gate_baseline` (dùng chung `dual_group_news_panel.parquet`,
`dataset_dual_news.create_dual_news_dataloaders`) — chỉ khác `NEWS_ON_TICKERS`/`NEWS_OFF_TICKERS`.

```
NEWS_ON_TICKERS  = {"VIB", "ACB", "MWG"}                         # 3 mã
NEWS_OFF_TICKERS = {tất cả 29 mã còn lại trong 32-mã pipeline}   # bao gồm SHB, VPB, VRE
```

## 2. File list

| File | Trách nhiệm |
|---|---|
| `code/model_top3_gate.py` | `Top3NewsGateBaseline(SelectiveGateNewsBaseline)` — chỉ đổi ticker set qua class attribute override |
| `code/train_top3_gate.py` | Train loop — copy từ `train_selective_gate.py`, đổi import model |
| `test/test_mask_correctness.py` | Test mask đúng (kế thừa cấu trúc test cũ, số liệu 3-vs-29) |

## 3. Isolation

Import read-only từ `2026-07-25_selective_news_gate_baseline/code/` (model_selective_gate.py) VÀ
`2026-07-25_dual_group_news_embedding_baseline/code/` (dataset_dual_news.py). Không sửa 2
baseline sibling đó. Output: `results/top3_gate_<timestamp>/`.
