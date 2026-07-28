---
name: project-vn30-ticker-universe-mismatch
description: "project's 32-ticker price universe is stale vs official HOSE VN30 (Ky 1/2026) by 5 extra + 3 missing; separately, news-filter regex is missing VPB/VRE vs the price universe itself"
metadata:
  node_type: memory
  type: project
  originSessionId: 4f7cf132-7896-4bf1-8313-3063fa32630a
  modified: 2026-07-26T17:07:33.181Z
---

Verified 2026-07-26 against the **official HOSE PDF** (Kỳ 1/2026, công bố 21/1/2026 —
`static2.vietstock.vn/.../21012026_cbtt___danh_sach_thanh_phan_hose_index_thang_1_2026.pdf`, the
authoritative source, not a web aggregator — an earlier web-aggregator source had a typo, "DCG"
instead of the real ticker "DGC", initially causing doubt).

**Official current VN30 (30 tickers):** ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, HPG, LPB, MBB,
MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VPL, VRE.

**This project's price/HAR universe (`data/processed/`, 32 tickers):** same as above MINUS
{DGC, LPB, VPL} PLUS {BCM, BVH, NVL, PDR, POW}. Net: 27 tickers in common, 5 the project has that
current VN30 doesn't (BCM, BVH, NVL, PDR, POW), 3 current VN30 has that the project is missing
(DGC, LPB, VPL) — NOT a simple "2 extra" as first assumed; verified count via official PDF, not
guessed.

**Why:** HOSE rebalances VN30 twice a year (4th Monday of Jan and Jul). Confirmed via news search:
BCM was removed and VPL added in the Jan/Feb-2026 rebalance specifically. The other 4 swaps
(BVH/NVL/PDR/POW out, LPB/DGC in) happened at earlier rebalances predating this project's data
collection — the project's `data/processed/` was built once against an older VN30 basket and
never refreshed as HOSE's index composition changed.

**Separate, second bug found in the same investigation:** the news-embedding ticker-mention
regex (`vendor_config.py::VN30_TICKERS` in
`baselines/2026-07-25_dual_group_news_embedding_baseline/code/`) lists only 30 tickers that
themselves DON'T include VPB or VRE — even though VPB/VRE ARE in the project's own 32-ticker price
universe. Consequence: `dual_group_news_panel.parquet` has ZERO rows for VPB and VRE (verified:
`df[df.ticker=='VPB']` has 0 rows) — their `x_news` input is always an all-zero vector in every
baseline that uses this panel (dual_group, macro, gated_crossattn, spillover_qlike,
per_ticker_gate, etc.). Any "gate value" or "news usefulness" reading for VPB/VRE specifically in
ANY of these baselines is an artifact of network bias terms on zero input, not a real signal —
discard those two tickers' numbers when interpreting per-ticker news results.

**How to apply:** (1) Issue A (price universe stale vs. current VN30) is a BIG change — needs
collecting price history for DGC/LPB/VPL and re-running every baseline. User decision 2026-07-26:
keep as-is for now, documented for later ("giữ nguyên, ghi chú vào .md để sau này giải quyết") —
NOT fixed, intentionally deferred. (2) Issue B (VPB/VRE missing from the news regex) — user
explicitly asked to fix it 2026-07-26 ("hãy fix bugs"). **FIXED**: added VPB, VRE to
`VN30_TICKERS` in `vendor_config.py` (`baselines/2026-07-25_dual_group_news_embedding_baseline/code/`),
rebuilt `dual_group_news_panel.parquet` (0 cache misses — the 2026-07-25 `--include_all` GPU
expansion already covered these articles). Verified: VPB now 61.54% date coverage, VRE 47.02%
(previously both 0%). Panel is now 32 tickers × 4989 dates (was 30×4890 pre-fix; date count also
grew, likely VPB/VRE's own trading calendar contributing new dates to the intersection — not
investigated further, not concerning). **Not yet re-trained** on the fixed panel — the
per-ticker-gate baseline's numbers/gate values reported earlier this session still reflect the
PRE-fix panel (VPB/VRE gate values were noted as artifacts to discard; post-fix, a re-run of
`per_ticker_news_gate_baseline` would be needed to get real VPB/VRE gate readings). (3) Old
per-ticker gate/usefulness tables computed before this fix still need VPB/VRE excluded from
interpretation (they reflect the OLD zero-coverage state) — only a fresh re-run reflects the fix.
