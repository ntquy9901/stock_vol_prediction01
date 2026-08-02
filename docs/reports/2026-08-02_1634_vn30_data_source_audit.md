# VN30 Data Source Audit — 2026-08-02

## Mục đích

Xác định: (1) `data/raw/` có những folder nào, folder nào là nguồn thật sự tạo ra
`data/processed/`, (2) vì sao project hiện có 32 mã thay vì 30, (3) đối chiếu chính xác với
danh sách VN30 hiệu lực đến hết 2026-08-02 do user cung cấp, (4) `data/processed/` có bao nhiêu
biến thể và khác nhau thế nào.

## 1. Các folder trong `data/raw/`

| Folder | Số file CSV mã cổ phiếu | Ghi chú |
|---|---|---|
| `data/raw/prices/` | 30 | Nguồn mặc định của `process_parkinson_pipeline.py` (`raw_dir` hardcode) |
| `data/raw/vn30/` | 28 (+`stock_summary`) | Có `VPB`, `VRE` (2 mã `prices/` không có); thiếu `NVL`, `PDR`, `SHB`, `SSB` |
| `data/raw/vn30_enhanced/` | 28 (+`stock_summary`) | Danh sách mã giống hệt `vn30/` |
| `data/raw/vn100/` | 102 (+`stock_summary`) | Universe rộng hơn nhiều (không chỉ VN30), có `LPB` |
| `data/raw/vn100_enhanced/` | 102 (+`stock_summary`) | Giống `vn100/`, có `LPB` |
| `data/raw/all_available/` | 135 (+`stock_summary`) | Universe rộng nhất, có `LPB` |
| `data/raw/hnx/`, `hnx_enhanced/` | 72 / 69 | Sàn HNX, không liên quan VN30 |
| `data/raw/news/`, `sentiment/`, `vn30_sentiment/` | — | Dữ liệu tin tức/sentiment, không phải giá |
| `data/raw/test/`, `test_combined/` | 10 / 11 | Dữ liệu test, không dùng cho pipeline chính |

**Không folder nào chứa `BSR` hay `VPL`** (đã `find` toàn bộ `data/`, 0 kết quả).

## 2. Tại sao `data/processed/` (root) có 32 mã

Truy vết bằng cách so khớp tập mã:

- `data/processed/vn30_only/` (30 mã) = **đúng bằng** `data/raw/prices/` (30 mã) đã qua
  `process_parkinson_pipeline.py`. Đây là kết quả "thuần" của pipeline mặc định.
- `data/processed/` (root, 32 mã) = `data/processed/vn30_only/` (30 mã) **cộng thêm** `VPB` và
  `VRE` — 2 file này khớp chính xác với `data/raw/vn30/VPB_ohlcv.csv` và
  `data/raw/vn30/VRE_ohlcv.csv` (đã xác nhận qua lỗi format ngày `+07:00` chỉ có ở `vn30/`, đã
  fix trong phiên này, xem commit `e434b1a`).

→ **Kết luận: `data/processed/` (32 mã) = `data/raw/prices/` (30 mã, nguồn chính) + `VPB`/`VRE`
lấy thủ công/riêng lẻ từ `data/raw/vn30/`** — không phải lỗi ngẫu nhiên, mà là 1 lần merge thủ
công tại thời điểm nào đó trong lịch sử project (không tìm thấy script nào tự động hoá việc
merge này — nghi ngờ làm tay hoặc bằng script đã bị archive/xoá).

## 3. Đối chiếu với danh sách VN30 chính thức (hiệu lực đến hết 2026-08-02)

Danh sách chính thức do user cung cấp (30 mã): ACB, BID, BSR, CTG, FPT, GAS, GVR, HDB, HPG, LPB,
MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VPL, VRE.

Project hiện tại (`data/processed/`, 32 mã): ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG,
MBB, MSN, MWG, NVL, PDR, PLX, POW, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC,
VNM, VPB, VRE.

**Trùng nhau: 27/30 mã chính thức.**

**5 mã DƯ trong project (KHÔNG thuộc VN30 hiện tại):**

| Mã | Trong `data/processed/`? | Raw source |
|---|---|---|
| BCM | Có | `data/raw/prices/` |
| BVH | Có | `data/raw/prices/` |
| NVL | Có | `data/raw/prices/` |
| PDR | Có | `data/raw/prices/` |
| POW | Có | `data/raw/prices/` |

**3 mã THIẾU trong project (thuộc VN30 hiện tại nhưng project không có):**

| Mã | Có raw data ở đâu không? |
|---|---|
| LPB | **Có** — `data/raw/vn100/LPB_ohlcv.csv`, `vn100_enhanced/`, `all_available/` (1456 dòng, 2020-11-09 → 2026-06-19). **Đã có sẵn bản processed**: `data/processed/vn100_only/LPB_ohlcv_processed.csv` (102 mã trong `vn100_only/`, format tên file hơi khác: `LPB_ohlcv_processed.csv` thay vì `LPB_processed.csv` như các mã khác — cần chuẩn hoá tên nếu dùng). Ngày cũng có dạng `+07:00` — cần qua fix `parkinson_utils.py` (đã sửa trong phiên này) nếu re-process. |
| BSR | **Không có** ở bất kỳ folder nào trong repo — cần crawl mới. |
| VPL | **Không có** ở bất kỳ folder nào trong repo — cần crawl mới. |

## 4. Các biến thể `data/processed/` hiện có

| Folder | Số mã | Nguồn |
|---|---|---|
| `data/processed/` (root) | 32 | `vn30_only` + VPB/VRE (mục 2) — **folder chính, được hầu hết baseline dùng** |
| `data/processed/vn30_only/` | 30 | = `data/raw/prices/` qua `process_parkinson_pipeline.py`, không có VPB/VRE |
| `data/processed/vn100_only/` | 102 | = `data/raw/vn100/` (hoặc tương đương) qua cùng pipeline, universe rộng hơn nhiều VN30 |
| `data/processed/vn30_sentiment/` | — | Dữ liệu sentiment, không phải giá |

## 5. Kết luận / khuyến nghị (chưa thực hiện, chờ quyết định)

1. **LPB có thể thêm ngay** — raw data đã có sẵn (`vn100_only`), chỉ cần: chuẩn hoá tên file
   (`LPB_ohlcv_processed.csv` → `LPB_processed.csv`), copy/re-process vào `data/processed/`
   (root), qua đúng `parkinson_utils.py` đã fix format ngày.
2. **BSR, VPL cần crawl mới** — không có trong repo, ngoài phạm vi 1 lần sửa code, cần chạy lại
   crawler (không rõ crawler nằm ở project này hay ở repo `crawl_data` sibling).
3. **5 mã dư (BCM, BVH, NVL, PDR, POW)** — cần quyết định: loại bỏ khỏi universe khi build lại
   dataset date-aligned (mục tiêu hiện tại: fix P1.2), hay giữ lại làm universe riêng (không
   khớp tên "VN30" chính thức, cần đổi tên/caveat nếu vậy).
4. Việc đóng băng universe 30-mã-chính-xác chỉ khả thi đầy đủ SAU KHI có BSR + VPL (crawl mới) —
   nếu cần fix P1.2 (date alignment) ngay bây giờ mà chưa có 2 mã này, có 2 lựa chọn tạm thời:
   (a) build trên 27 mã giao nhau (chính xác nhưng thiếu 3 mã chính thức), hoặc (b) build trên
   29 mã (27 giao nhau + LPB có sẵn data), chờ crawl BSR/VPL sau.
