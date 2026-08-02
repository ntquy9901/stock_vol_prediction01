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

## 5. Kết luận / khuyến nghị

1. ~~LPB có thể thêm ngay~~ — **[ĐÃ XONG 2026-08-02]** Processed qua đúng `parkinson_utils.py`
   (đã fix format ngày) → `data/processed/LPB_processed.csv` (1456 dòng, 2020-11-09 → 2026-06-19,
   format ngày plain `YYYY-MM-DD` nhất quán với các mã khác).
2. ~~BSR, VPL cần crawl mới~~ — **[ĐÃ CRAWL, QUYẾT ĐỊNH: LOẠI KHỎI UNIVERSE]** Crawl qua
   `src/data/crawl_vietnam_stocks.py` (Yahoo Finance, `.VN` suffix) thành công:
   - BSR: 401 dòng, 2025-01-17 → 2026-07-31 (~1.4 năm)
   - VPL: 99 dòng, 2026-03-17 → 2026-07-31 (~4.5 tháng — mới niêm yết)

   Nếu bắt buộc dùng chung khung ngày (intersection) cho cả 30 mã chính thức, khung chung chỉ còn
   **~84 ngày lịch (~58 ngày giao dịch)** do VPL — quá ít để train LSTM+GNN (chỉ ~31 window trước
   khi chia train/val/test). **Quyết định (2026-08-02): loại BSR và VPL khỏi universe** do dữ liệu
   quá ít. File raw đã crawl (`data/raw/vn30/BSR_ohlcv.csv`, `VPL_ohlcv.csv`) **giữ lại, không xoá**
   — không dùng trong pipeline hiện tại, nhưng có sẵn nếu sau này đủ lịch sử để cân nhắc lại.
3. **5 mã dư (BCM, BVH, NVL, PDR, POW) — CHƯA quyết định**, vẫn còn trong `data/processed/`.
   Universe hiện tại (sau khi thêm LPB, chưa loại 5 mã dư): 33 mã = 28/30 mã chính thức (thiếu
   đúng BSR, VPL đã loại có chủ đích) + 5 mã dư nói trên.
4. ~~5 mã dư — CHƯA quyết định~~ — **[ĐÃ CHỐT 2026-08-02]** Giữ cả 5 mã (BCM/BVH/NVL/PDR/POW):
   toàn bộ đều có ≥8 năm dữ liệu (dài hơn mã ngắn nhất trong universe, SSB ~5 năm), không thu hẹp
   thêm khung ngày intersection. Universe cuối cùng: **33 mã** (xem mục 6).

---

## 6. Universe dữ liệu cuối cùng (dùng cho paper)

Chốt ngày 2026-08-02. 33 mã, nguồn từng mã như sau:

| Mã | Nguồn raw | Số dòng | Khoảng ngày | Trong VN30 hiệu lực 2026-08-02? |
|---|---|---|---|---|
| ACB | `data/raw/prices/` | 4868 | 2006-11-21 → 2026-06-09 | ✅ |
| BCM | `data/raw/prices/` | 2065 | 2018-02-21 → 2026-06-09 | ❌ (giữ lại, đủ data) |
| BID | `data/raw/prices/` | 3083 | 2014-01-24 → 2026-06-09 | ✅ |
| BVH | `data/raw/prices/` | 4232 | 2009-06-25 → 2026-06-09 | ❌ (giữ lại, đủ data) |
| CTG | `data/raw/prices/` | 4217 | 2009-07-16 → 2026-06-09 | ✅ |
| FPT | `data/raw/prices/` | 4854 | 2006-12-13 → 2026-06-09 | ✅ |
| GAS | `data/raw/prices/` | 3508 | 2012-05-21 → 2026-06-09 | ✅ |
| GVR | `data/raw/prices/` | 2046 | 2018-03-21 → 2026-06-09 | ✅ |
| HDB | `data/raw/prices/` | 2100 | 2018-01-05 → 2026-06-09 | ✅ |
| HPG | `data/raw/prices/` | 4625 | 2007-11-15 → 2026-06-09 | ✅ |
| LPB | `data/raw/vn100/` | 1456 | 2020-11-09 → 2026-06-19 | ✅ (thêm 2026-08-02) |
| MBB | `data/raw/prices/` | 3643 | 2011-11-01 → 2026-06-09 | ✅ |
| MSN | `data/raw/prices/` | 4138 | 2009-11-05 → 2026-06-09 | ✅ |
| MWG | `data/raw/prices/` | 2973 | 2014-07-14 → 2026-06-09 | ✅ |
| NVL | `data/raw/prices/` | 2356 | 2016-12-28 → 2026-06-09 | ❌ (giữ lại, đủ data) |
| PDR | `data/raw/prices/` | 3956 | 2010-07-30 → 2026-06-09 | ❌ (giữ lại, đủ data) |
| PLX | `data/raw/prices/` | 2281 | 2017-04-21 → 2026-06-09 | ✅ |
| POW | `data/raw/prices/` | 2054 | 2018-03-06 → 2026-06-09 | ❌ (giữ lại, đủ data) |
| SAB | `data/raw/prices/` | 2372 | 2016-12-06 → 2026-06-09 | ✅ |
| SHB | `data/raw/prices/` | 4275 | 2009-04-20 → 2026-06-09 | ✅ |
| SSB | `data/raw/prices/` | 1299 | 2021-03-24 → 2026-06-09 | ✅ |
| SSI | `data/raw/prices/` | 4841 | 2006-12-18 → 2026-06-09 | ✅ |
| STB | `data/raw/prices/` | 4887 | 2006-10-27 → 2026-06-09 | ✅ |
| TCB | `data/raw/prices/` | 2002 | 2018-06-04 → 2026-06-09 | ✅ |
| TPB | `data/raw/prices/` | 2031 | 2018-04-19 → 2026-06-09 | ✅ |
| VCB | `data/raw/prices/` | 4229 | 2009-06-30 → 2026-06-09 | ✅ |
| VHM | `data/raw/prices/` | 3426 | 2011-11-10 → 2026-06-09 | ✅ |
| VIB | `data/raw/prices/` | 2342 | 2017-01-09 → 2026-06-09 | ✅ |
| VIC | `data/raw/prices/` | 4666 | 2007-09-19 → 2026-06-09 | ✅ |
| VJC | `data/raw/prices/` | 2318 | 2017-02-28 → 2026-06-09 | ✅ |
| VNM | `data/raw/prices/` | 4887 | 2006-10-27 → 2026-06-09 | ✅ |
| VPB | `data/raw/vn30/` | 2294 | 2017-08-17 → 2026-06-19 | ✅ |
| VRE | `data/raw/vn30/` | 2237 | 2017-11-06 → 2026-06-19 | ✅ |

**Tổng hợp:** 28/30 mã VN30 chính thức (hiệu lực đến 2026-08-02) + 5 mã ngoài danh sách hiện tại
(giữ lại vì đủ dữ liệu lịch sử, không phải sai sót). Tất cả đã qua `src/common/parkinson_utils.py`
(đã fix format ngày 2026-08-02), format `date` thống nhất `YYYY-MM-DD` cho toàn bộ 33 mã.

## 7. Giới hạn dữ liệu — lý do loại 2 mã khỏi VN30 (dùng cho mục Limitations của paper)

**BSR và VPL bị loại khỏi universe huấn luyện**, mặc dù cả hai đều thuộc danh sách VN30 hiệu lực
đến 2026-08-02. Lý do: dữ liệu lịch sử quá ngắn để tham gia vào 1 dataset yêu cầu chung 1 khung
ngày giữa các mã (kiến trúc LSTM-GAT hybrid xây đồ thị quan hệ chéo-mã theo từng timestep, đòi
hỏi mọi mã phải có dữ liệu tại cùng ngày).

- **VPL** (Vinpearl): niêm yết 2026-03-17, chỉ 99 phiên giao dịch tính đến thời điểm audit
  (2026-08-02). Đưa VPL vào sẽ ép khung ngày dùng chung của toàn bộ 33 mã co lại còn ~84 ngày
  lịch (~58 ngày giao dịch) — không đủ tạo ra dù chỉ vài chục sequence 22-ngày cho train/val/test.
- **BSR** (Binh Son Refining): niêm yết 2025-01-17, 401 phiên giao dịch tính đến thời điểm audit —
  đủ dài hơn VPL nhiều, nhưng vẫn là mã có lịch sử ngắn thứ nhì trong toàn bộ universe, thu hẹp
  đáng kể khung ngày nếu đưa vào cùng VPL.

Dữ liệu 2 mã này (`data/raw/vn30/BSR_ohlcv.csv`, `VPL_ohlcv.csv`) đã crawl và lưu lại (không xoá)
— có thể tái xét khi có đủ lịch sử giao dịch trong tương lai. Tại thời điểm hiện tại, việc loại
trừ này là quyết định có chủ đích, có định lượng cụ thể, không phải một thiếu sót — nên nêu rõ
trong phần Limitations/Dataset Description của paper, kèm 2 con số ở trên.
