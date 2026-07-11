"""Create a TIMESTAMPED copy of SENTIMENT_ANALYSIS_DESIGN.md with section 2.3 updated
to reflect the 9-source aggregated dataset + per-year sparsity analysis.

Does NOT modify the original file. Output: SENTIMENT_ANALYSIS_DESIGN_YYYY-MM-DD.md.

Usage: python -m src.data_aggregation.update_design_23
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ORIG = ROOT / "docs" / "project" / "SENTIMENT_ANALYSIS_DESIGN.md"
OUT = ROOT / "docs" / "project" / "SENTIMENT_ANALYSIS_DESIGN_2026-07-06.md"

# Per-year data from analyze_news_sparsity.py (run 06/07/2026 on 21,107 unified articles)
# (year: articles, match_full, stockdays_news, stockdays_total)
ROWS = [
    ("2008", 183, 4, 4, 1718),
    ("2009", 203, 4, 4, 2366),
    ("2010", 392, 19, 18, 3109),
    ("2011", 758, 33, 32, 3305),
    ("2012", 387, 27, 27, 3908),
    ("2013", 504, 39, 40, 4000),
    ("2014", 513, 43, 43, 4304),
    ("2015", 513, 58, 58, 4464),
    ("2016", 391, 43, 42, 4540),
    ("2017", 369, 37, 37, 5519),
    ("2018", 389, 84, 82, 6862),
    ("2019", 725, 229, 566, 7242),
    ("2020", 1352, 365, 517, 7283),
    ("2021", 1683, 386, 387, 7445),
    ("2022", 1594, 353, 323, 7470),
    ("2023", 1484, 305, 292, 7470),
    ("2024", 1601, 396, 363, 7500),
    ("2025", 1685, 411, 389, 7470),
    ("2026", 2486, 584, 483, 3120),
]
SPLIT = [("Train (<2020)", 5335, 953, 51337),
         ("Val (2020-2021)", 3035, 904, 14728),
         ("Test (>=2021)", 10533, 2237, 40475)]


def bar(n, unit=50):
    return "█" * max(1, round(n / unit))


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "0.0%"


def build_section():
    L = []
    L.append("### 2.3 Phân bố thời gian & khoảng trống (gaps)")
    L.append("")
    L.append("> **Cập nhật 06/07/2026** — bảng dưới tính trên **dataset gộp 9 nguồn** (21.107 bài unique sau dedup, script `src/data_aggregation/aggregate_news_sources.py`), KHÔNG phải 3 file cũ (~12.212). Sparsity per-year: `src/data_aggregation/analyze_news_sparsity.py` → `crawl_data/aggregated/sparsity_report.txt`. (Các số ở 2.4-2.5 dưới vẫn là của 3 file cũ — chưa cập nhật.)")
    L.append("")
    L.append("**2 góc nhìn: (a) số bài/năm — phân bố đều chưa; (b) mật độ per-stock-day — còn thưa không.**")
    L.append("")
    L.append("**(a) Số bài/năm (sau dedup) — phân bố ĐÃ ĐỀU, gap đã lấp:**")
    L.append("")
    L.append("| Năm | Số bài | Biểu đồ (1 █ = ~50 bài) |")
    L.append("|-----|--------|--------------------------|")
    for y, a, _mf, _sdn, _sdt in ROWS:
        L.append(f"| {y} | {a:,} | {bar(a)} |")
    L.append("")
    L.append("→ **Gap cũ đã hết**: 2016 từng chỉ 18 bài (LOW) → giờ 391. 2021-2026 đều 1.484-2.486 — **đều giữa các năm**. Ở cấp số bài, KHÔNG còn thưa.")
    L.append("")
    L.append("**(b) Mật độ per-stock-day (metric thật cho model) — VẪN RẤT THƯA:**")
    L.append("")
    L.append("| Năm | bài | match ticker¹ | stock-day có tin² | tổng stock-day³ | **coverage** |")
    L.append("|-----|-----|---------------|-------------------|-----------------|--------------|")
    for y, a, mf, sdn, sdt in ROWS:
        note = " ⁴" if y == "2026" else ""
        L.append(f"| {y}{note} | {a:,} | {mf} | {sdn} | {sdt:,} | **{pct(sdn, sdt)}** |")
    L.append("")
    L.append("¹ bài match ≥1 mã VN30 (title+lead). Toàn dataset: **19.9%** (title-only: 16.7%)  ")
    L.append("² unique (ticker, date) có tin — đơn vị model dùng  ")
    L.append("³ tổng (ticker, ngày giao dịch) từ lịch giá VN30  ")
    L.append("⁴ 2026 đang partway (mới ~3.120 stock-day đầu năm) → coverage 15.5% là ước lượng thiêu biện")
    L.append("")
    L.append("| Split | bài | stock-day có tin | tổng stock-day | **coverage** |")
    L.append("|-------|-----|------------------|----------------|--------------|")
    for name, a, sdn, sdt in SPLIT:
        L.append(f"| {name} | {a:,} | {sdn:,} | {sdt:,} | **{pct(sdn, sdt)}** |")
    L.append("")
    L.append("**Verdict trung thực:**")
    L.append("- ✅ **Cấp số bài**: gap đã lấp, phân bố đều 2008-2026. Vấn đề 'test mù tin' (Mục 10 #1) **đã giải quyết ở cấp article** — có tin trong test period.")
    L.append("- \U0001f534 **Cấp per-stock-day**: **VẪN CỰC THƯA** — chỉ **5.5%** stock-day trong test có tin ticker-specific (94.5% = 0 tin). Nguyên nhân gốc: chỉ ~20% bài match mã VN30 cụ thể (đa số tin vĩ mô/market-wide).")
    L.append("- → Thêm bài (`vnstock` 14.825 raw → chỉ 432 unique) **không đặc lên**: bottleneck là match-rate ~20%, không phải số bài.")
    L.append("- → Đây là lý do **các kĩ thuật thầy gợi ý vẫn cần thiết**: embedding (rút thêm tín hiệu từ ít bài có match) + latent/missing-modality (vì 94.5% stock-day vẫn mù tin) + market-level fallback. Xem `SENTIMENT_LATENT_SPACE_TECHNIQUES.md` & `SENTIMENT_NEWS_EMBEDDING_ARCHITECTURE.md`.")
    L.append("")
    return "\n".join(L)


def main():
    text = ORIG.read_text(encoding="utf-8-sig")
    start = text.index("### 2.3")
    end = text.index("### 2.4", start)
    new_text = text[:start] + build_section() + "\n" + text[end:]
    OUT.write_text(new_text, encoding="utf-8")
    print(f"[write] {OUT}")
    print(f"[ok] original untouched: {ORIG}")
    # sanity: confirm section markers present in copy
    chk = OUT.read_text(encoding="utf-8")
    assert "### 2.3" in chk and "### 2.4" in chk and "Cập nhật 06/07/2026" in chk
    print("[verify] 2.3 updated, 2.4 intact, marker present.")


if __name__ == "__main__":
    main()
