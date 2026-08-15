# Requirements — Track-A GAT + node features + volume→PK edge

## Mục tiêu
Kiểm tra bằng một GNN **GAT thật (attention học được) kiểu Track A** xem cross-stock graph có thêm
giá trị out-of-sample so với chính nó khi tắt graph hay không (graph-on vs graph-off), dùng node
features đã DM-thắng HAR (HAR + MarketPK + volume_zscore_20) + news + edge directed volume→PK Top-5.
Đây là cơ chế GNN duy nhất chưa chạy (các test trước dùng residual-MP adjacency cố định).

## Input / Output
- Input: masked manifest leakage-safe hiện có — 5 node features × 22 ngày + news 146 + đồ thị vol→PK
  Top-5 (snapshot chéo-mã), train 73026 / val 14418 / test 14464 (giống HAR/E2 canonical).
- Output: per-seed `results/trackA_gat_seed{seed}_<TS>/ladder_metrics.json` (6 metric val+test cho
  HAR/NODE/GNN) + per-obs prediction dumps + checkpoint `models/trackA_gat_seed{seed}_<TS>.pt`;
  DM aggregate: GNN vs NODE (graph helpful?), GNN vs HAR, NODE vs HAR.

## Rung / Ablation
- HAR = hồi quy tuyến tính 3 HAR feature (baseline ngoài).
- NODE = model đầy đủ đọc với `apply_graph=False` (adjacency identity → không lan chéo mã).
- GNN = model đầy đủ đọc với `apply_graph=True` (adjacency vol→PK) — nested trên cùng checkpoint.

## Success criteria / go-no-go
- Leakage-safe (assert): scaler/edge/gate/news-cutoff/positivity-floor TRAIN-only; bất biến một-basis
  (obs graph == obs pooled); positivity floor 1e-6 đồng nhất mọi rung (bài học H2). **Gate cứng.**
- Cơ chế RESUME hoạt động: train 15 epoch → checkpoint → resume +5/10 epoch (epoch counter +
  optimizer + best-val khôi phục đúng), có test chứng minh.
- Báo cáo DM trung thực theo từng metric; graph thắng/hoà/thua đều là kết quả hợp lệ. KHÔNG bịa win,
  không cherry-pick seed, không overfit test.

## Lịch chạy (user duyệt 2026-08-15)
1 seed × 15 epoch trước (báo cáo val metrics) → resume +5/10 nếu cần → 3 seeds (42/123/2026) + DM khi
chốt. (>10 epoch: user đã đồng ý rõ.)

## Non-goals
Không dùng common-date panel riêng (sẽ không DM-so được với HAR canonical); không torch_geometric
(GAT tự viết); không tìm hyperparameter.
