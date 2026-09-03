# Requirements — Pooled/transfer ablation for VN30

**Objective:** measure whether widening the deep model's training universe from 31 (VN30) to 102
(VN100) stocks improves VN30 volatility forecasts, as a single-variable walk-forward ablation.

**Inputs:** `data/processed_enriched/vn100/*.csv` (102-node enriched panel; VN30 ⊂ VN100),
`data/processed_enriched/vn30/*.csv` (VN30 universe screen). 5 node features
`[parkinson_variance, har_weekly, har_monthly, market_pk, volume_zscore_22]`; target = Parkinson
variance at t+h.

**Design (one variable):** ONE VN100 panel + ONE fold set. Arm 0 trains on the 31 VN30 nodes,
Arm 1 trains on all 102 nodes (training loss + vol→PK graph restricted per arm). Both arms score
exactly the 31 VN30 nodes on the identical OOS grid → perfect paired alignment.

**Success criteria (a priori decision rule):**
- Headline (A): paired date-clustered DM, Arm 1 vs Arm 0, for VolGA and LSTM, on 3 loss bases
  (QLIKE / SE / AE), per horizon. "Pooling helps deep" ⟺ Arm 1 significantly better (p < 0.05).
- Secondary (B): difference-in-differences of gap(deep − HAR) between arms.
- Report H0/H1 honestly regardless of sign (prior Track B A1 2026-08-08 found a null; stated).

**Go/no-go:** run h1 first; extend to h5/h10/h22 if a clear signal (either sign).

**Leakage:** graph + scalers train-only per fold; `assert_no_leakage`; non-VN30 stocks are training
context only, never scored.

See `design/design.md` (full spec) and `docs/superpowers/plans/2026-09-04-pooled-transfer-vn30.md`.
