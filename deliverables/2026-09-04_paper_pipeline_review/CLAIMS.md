# Claims the paper would make — each with an evidence pointer

Every claim is checkable against `RESULTS_SUMMARY.md` and the result JSONs. The headline is a
**parsimony / partly-negative** result stated honestly; reviewer should reject any claim that
overstates a non-significant difference.

## Primary claims
1. **The HAR/HAR-X econometric baseline is not beaten by the deep models on QLIKE at any horizon**
   on either VN30 or VN100. Evidence: DM VolGA-vs-HAR-X n.s. at all 8 market×horizon cells
   (`RESULTS_SUMMARY.md`; `results/walkforward_volga/*`). HAR/HAR-X hold the point-QLIKE best at
   7 of 8 cells.

2. **The dynamic vol→PK graph adds significant marginal value only on the larger, more liquid
   market at short horizons.** VolGA − no-graph LSTM is significant on VN100 at h1 (p=0.008) and
   h5 (p=0.011), and not significant at h10/h22 or at any VN30 horizon. Evidence: `dm_date_clustered.
   VolGA_vs_LSTM` in the VN100/VN30 JSONs.

3. **Graph value scales with node breadth × liquidity, not correlation magnitude.** VN30 has higher
   mean |ρ| than VN100 yet shows a null graph effect, while the 102-node VN100 shows a significant
   one. Evidence: claim 2 + the cross-market report `docs/reports/2026-08-31_volga_walkforward_vn30_vs_vn100.md`.

4. **The marginal-value verdict is loss-function dependent.** On squared/absolute error the graph
   reaches significance at some short horizons where QLIKE does not; therefore DM is reported on all
   three bases (QLIKE/SE/AE). Evidence: the 3-basis DM tables in the dashboards.

5. **Parkinson variance is the appropriate range-based target for this market.** Rogers–Satchell,
   Yang–Zhang and close-to-close give worse forecast QLIKE; the loss comes from non-persistent
   overnight components inflated by unadjusted corporate actions. Evidence:
   `docs/reports/2026-08-23_1600_volatility_estimator_research.md`;
   `docs/reports/appendix/2026-09-03_overnight_tail_appendix.md`.

## Ablation claim (COMPLETE — all 4 horizons)
6. **Widening the deep model's training universe (VN30→VN100) does not overcome the QLIKE ceiling on
   VN30, but reduces absolute error at short-to-medium horizons.** Paired DM Arm1-vs-Arm0: QLIKE not
   significant for VolGA at any horizon (LSTM significant only at h10), so the QLIKE ceiling holds
   (matches the prior Track B A1 null); absolute error is significantly lower under the wider universe
   at h1/h5/h10 for both deep models, fading by h22. The benefit of pooling is therefore loss- and
   horizon-dependent. Evidence: `results/pooled_transfer_vn30/pooled_vn30_h{1,5,10,22}.json`;
   `docs/reports/2026-09-04_pooled_transfer_vn30_report.md`; dashboard
   `docs/reports/2026-09-04_pooled_transfer_vn30_dashboard.html`.

## Baseline definition (must be cited)
7. **HAR-X is a published baseline**, not an ad-hoc construction: HAR (Corsi, 2009) + exogenous
   regressors ("-X" convention; Clements, Preve & Tee, 2024, "Harvesting the HAR-X", which uses
   low-frequency OHLC — validating the range-based target); network+HAR-X analogue GNAR-HARX
   (arXiv:2510.24443, 2025). Disclosed deviations: (a) range-based Parkinson target, (b) direct
   single-day t+h target. Evidence: `docs/papers/README.md`.

## Disclosed limitations (paper must include)
- Raw VN prices not split/dividend-adjusted → overnight tail; mitigated by using the intraday
  Parkinson estimator + winsorization; root fix = adjusted prices.
- Target is variance (σ²), not σ. QLIKE positivity floor from config; floor sensitivity on
  high-zero-range markets (HNX) documented.
- VN30 is small (31 nodes) — deep/graph methods are data-limited there.
- Ablation Arm 0 is on the VN100 grid (VN100 `market_pk`), not the standalone-VN30 setup; only
  Arm0-vs-Arm1 is a valid comparison.

## What is NOT claimed (guard against overreach)
- No claim that any deep model beats HAR-X on the primary loss.
- No claim that the graph helps on small or illiquid markets.
- No directional-accuracy claim (structural ~48% ceiling; excluded).
