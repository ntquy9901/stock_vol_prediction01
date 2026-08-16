# VN100 quick ablation — does more pooled training data change the ablation?

Date: 2026-08-16
Scope: exploratory, single-seed, few-epoch leave-one-out ablation on the VN100 universe (104
tickers) vs the VN30 universe (33 tickers), to check whether ~3x more pooled training data changes
absolute metrics or component contributions. Indicative only.

## Configuration

| Item | VN100 run (this report) | VN30 reference |
|---|---|---|
| Universe / ticker files | 104 (`data/processed/vn100/*_processed.csv`) | 33 (`data/processed/*_processed.csv`) |
| Lookback SEQ | 22 | 22 |
| Split ratios | 70/15/15 (default `load_and_split_price_data`) | 70/15/15 |
| Seed | 42 (single) | 42 (single seed of the 5-seed loo) |
| Epoch cap | 8 (early-stop patience 3, min 6) | 12 (early-stop patience 3, min 6) |
| Horizons | h1, h5, h10 (h22 not run) | h1, h5, h10 |
| Node features | 5 (parkinson_volatility, har_weekly, har_monthly, market_pk, volume_zscore_20) | same |
| News dim | 146 (PhoBERT panel) | 146 |
| Model / rungs | FULL, minus_graph, minus_gate, minus_news, lstm_only + HAR | same (loo run has no lstm_only rung) |
| Results TS | `2026-08-16_vn100quick` | `2026-08-15_085544_loo` |

Basis sizes (leakage-safe pooled manifest, warm-up + horizon binding):

| | VN100 | VN30 | ratio |
|---|---|---|---|
| train observations | 218,462 | 73,158 | 3.0x |
| val observations | 43,264 | 14,550 | 3.0x |
| test observations (h1) | 43,413 | 14,596 | 3.0x |
| directed vol->PK edges | 520 | 165 | 3.2x |
| graph snapshots | 5,799 | 6,482 | 0.9x |

The pipeline ran on VN100 without error. News is present only for the ~33 VN30-era tickers; the 71
new VN100 tickers have no rows in `data/features/dual_group_news_panel.parquet` and are zero-filled
at attach time (mask = 0), so the news/gate branches run but news is effectively absent for ~68% of
tickers.

## How the run was pointed at VN100

Wrapper script: `scripts/run_vn100_ablation.py`. It inserts the Track-A code dir on `sys.path`,
imports `run_ablation` (which sets up paths and imports `combo_ladder`), overrides the module global
`combo_ladder._PROCESSED = <repo>/data/processed/vn100` before any basis is built (SEQ stays 22,
ratios stay default), then calls `run_ablation.main(ts, device, seed, epochs, horizons)`.

Command run (GPU venv, RTX 4060):
```
PYTHONIOENCODING=utf-8 .venv_gpu_encode/Scripts/python.exe \
    scripts/run_vn100_ablation.py 2026-08-16_vn100quick cuda 42 8 1 5
```
Results: `results/trackA_ablation_h{1,5}_seed42_2026-08-16_vn100quick/` (per-rung dumps +
`ladder_metrics.json`). DM: `baselines/2026-08-15_trackA_gat_edge/code/dm_report.py`.

## VN100 per-rung held-out TEST metrics

### h1 (test n = 43,413)
| rung | MSE | RMSE | MAE | R2 | QLIKE |
|---|---|---|---|---|---|
| HAR | 6.959e-07 | 0.000834 | 0.000288 | 0.0895 | 0.8945 |
| FULL | 6.929e-07 | 0.000832 | 0.000283 | 0.0934 | 0.8818 |
| minus_graph | 6.928e-07 | 0.000832 | 0.000289 | 0.0935 | 0.8811 |
| minus_gate | 6.928e-07 | 0.000832 | 0.000285 | 0.0935 | 0.8816 |
| minus_news | 6.931e-07 | 0.000833 | 0.000284 | 0.0931 | 0.8825 |
| lstm_only | 6.930e-07 | 0.000832 | 0.000290 | 0.0933 | 0.8817 |

Leave-one-out QLIKE effect (FULL − minus_X; negative ⇒ removing X hurt ⇒ X helped):
graph +0.00062, gate +0.00019, news −0.00072. All near zero.

### h5 (test n = 42,997)
| rung | MSE | RMSE | MAE | R2 | QLIKE |
|---|---|---|---|---|---|
| HAR | 7.354e-07 | 0.000858 | 0.000315 | 0.0451 | 0.9727 |
| FULL | 7.322e-07 | 0.000856 | 0.000313 | 0.0492 | 0.9609 |
| minus_graph | 7.330e-07 | 0.000856 | 0.000313 | 0.0482 | 0.9632 |
| minus_gate | 7.322e-07 | 0.000856 | 0.000313 | 0.0492 | 0.9608 |
| minus_news | 7.299e-07 | 0.000854 | 0.000319 | 0.0521 | 0.9563 |
| lstm_only | 7.306e-07 | 0.000855 | 0.000320 | 0.0513 | 0.9580 |

Leave-one-out QLIKE effect (FULL − minus_X): graph −0.00231 (graph helped), gate +0.00012,
news +0.00457 (news hurt). 

### h10 (test n = 42,477)
| rung | MSE | RMSE | MAE | R2 | QLIKE |
|---|---|---|---|---|---|
| HAR | 7.515e-07 | 0.000867 | 0.000324 | 0.0338 | 0.9948 |
| FULL | 7.556e-07 | 0.000869 | 0.000340 | 0.0284 | 1.0007 |
| minus_graph | 7.549e-07 | 0.000869 | 0.000333 | 0.0293 | 1.0023 |
| minus_gate | 7.567e-07 | 0.000870 | 0.000340 | 0.0270 | 1.0040 |
| minus_news | 7.564e-07 | 0.000870 | 0.000338 | 0.0275 | 1.0043 |
| lstm_only | 7.549e-07 | 0.000869 | 0.000333 | 0.0294 | 1.0022 |

Leave-one-out QLIKE effect (FULL − minus_X): graph −0.00168 (helped), gate −0.00336 (helped),
news −0.00358 (helped). At h10 all three components help FULL, but FULL as a whole is worse than HAR.

## VN100 Diebold-Mariano (HLN) verdicts — FULL vs each comparator
dm_hln(p); negative dm favors FULL; `*` = p<0.05.

### h1
| A vs B | QLIKE | SE | AE | n |
|---|---|---|---|---|
| FULL vs HAR | −3.41(0.00)* | −7.37(0.00)* | −16.05(0.00)* | 43,413 |
| FULL vs minus_graph | +0.21(0.83) | +0.59(0.55) | −54.65(0.00)* | 43,413 |
| FULL vs minus_gate | +0.30(0.76) | +1.06(0.29) | −30.41(0.00)* | 43,413 |
| FULL vs minus_news | −1.63(0.10) | −2.78(0.01)* | −18.90(0.00)* | 43,413 |
| FULL vs LSTM_only | +0.02(0.98) | −0.56(0.58) | −59.13(0.00)* | 43,413 |

### h5
| A vs B | QLIKE | SE | AE | n |
|---|---|---|---|---|
| FULL vs HAR | −2.69(0.01)* | −4.50(0.00)* | −7.42(0.00)* | 42,997 |
| FULL vs minus_graph | −6.10(0.00)* | −7.09(0.00)* | −0.86(0.39) | 42,997 |
| FULL vs minus_gate | +0.71(0.48) | +0.56(0.57) | −9.08(0.00)* | 42,997 |
| FULL vs minus_news | +4.39(0.00)* | +7.48(0.00)* | −28.37(0.00)* | 42,997 |
| FULL vs LSTM_only | +2.69(0.01)* | +5.71(0.00)* | −29.32(0.00)* | 42,997 |

### h10
| A vs B | QLIKE | SE | AE | n |
|---|---|---|---|---|
| FULL vs HAR | +2.48(0.01)* | +6.30(0.00)* | +28.00(0.00)* | 42,477 |
| FULL vs minus_graph | −1.05(0.29) | +1.51(0.13) | +16.71(0.00)* | 42,477 |
| FULL vs minus_gate | −5.57(0.00)* | −7.86(0.00)* | −3.42(0.00)* | 42,477 |
| FULL vs minus_news | −4.51(0.00)* | −4.93(0.00)* | +11.91(0.00)* | 42,477 |
| FULL vs LSTM_only | −0.94(0.35) | +1.59(0.11) | +17.26(0.00)* | 42,477 |

At h10 the positive FULL-vs-HAR QLIKE dm (+2.48*) means HAR beats FULL. On VN100 h10 the gate and
news components significantly help FULL (−5.57*, −4.51*), but the whole model still trails HAR.

## VN30 reference (seed 42) — for contrast

### VN30 per-rung TEST QLIKE / R2
| rung | h1 QLIKE | h1 R2 | h5 QLIKE | h5 R2 | h10 QLIKE | h10 R2 |
|---|---|---|---|---|---|---|
| HAR | 0.4813 | 0.8192 | 0.5735 | 0.7672 | 0.6139 | 0.7532 |
| FULL | 0.4780 | 0.8221 | 0.5724 | 0.7733 | 0.6924 | 0.7458 |
| minus_graph | 0.4741 | 0.8220 | 0.5692 | 0.7719 | 0.6222 | 0.7511 |
| minus_gate | 0.4731 | 0.8245 | 0.5741 | 0.7718 | 0.6261 | 0.7545 |
| minus_news | 0.4724 | 0.8239 | 0.5715 | 0.7720 | 0.7348 | 0.7461 |

### VN30 DM (FULL vs comparator), seed 42
| A vs B | h1 QLIKE | h5 QLIKE | h10 QLIKE |
|---|---|---|---|
| FULL vs HAR | −2.01(0.04)* | −0.52(0.60) | +4.96(0.00)* |
| FULL vs minus_graph | +2.18(0.03)* | +2.93(0.00)* | +4.58(0.00)* |
| FULL vs minus_gate | +3.70(0.00)* | −1.04(0.30) | +5.46(0.00)* |
| FULL vs minus_news | +6.38(0.00)* | +0.77(0.44) | −3.58(0.00)* |
| FULL vs LSTM_only | +4.56(0.00)* | +2.24(0.02)* | +4.67(0.00)* |
n(h1/h5/h10) = 14,596 / 14,464 / 14,299.

## VN30 vs VN100 — key comparisons

FULL−vs−HAR (QLIKE DM, the headline question "does the deep model beat HAR"):
| horizon | VN30 | VN100 |
|---|---|---|
| h1 | −2.01 (p=0.04)* FULL wins | −3.41 (p<0.01)* FULL wins (stronger, also SE/AE sig.) |
| h5 | −0.52 (p=0.60) tie | −2.69 (p=0.01)* FULL wins |
| h10 | +4.96 (p<0.01)* HAR wins | +2.48 (p=0.01)* HAR wins |

Component contribution direction (QLIKE; "helps" = removing it significantly worsens FULL):
| component | VN30 h1 | VN100 h1 | VN30 h5 | VN100 h5 | VN30 h10 | VN100 h10 |
|---|---|---|---|---|---|---|
| graph | hurts (+2.18*) | neutral (+0.21) | hurts (+2.93*) | HELPS (−6.10*) | hurts (+4.58*) | neutral (−1.05) |
| gate | hurts (+3.70*) | neutral (+0.30) | neutral (−1.04) | neutral (+0.71) | hurts (+5.46*) | HELPS (−5.57*) |
| news | hurts (+6.38*) | neutral (−1.63) | neutral (+0.77) | HURTS (+4.39*) | HELPS (−3.58*) | HELPS (−4.51*) |

## Caveats (explicit)

1. Different universes — not apples-to-apples on absolute metrics. VN100 includes many smaller,
   noisier tickers, so its absolute R2 is far lower (0.05–0.09 vs VN30 0.77–0.82) and QLIKE far
   higher (0.88–0.97 vs 0.48–0.57). RMSE/MSE are also on a different scale (VN100 has many low-vol
   tickers). "More data → better absolute metrics" cannot be read off these numbers because the test
   population changed; the absolute-metric gap reflects the universe, not data quantity.
2. News is zero-filled for the 71 non-VN30 tickers (68% of VN100), so any news/gate contribution on
   VN100 is understated / confounded with an implicit "news-present" ticker subgroup.
3. Single seed, small epoch cap (VN100 = 8, VN30 reference = 12), no seed ensembling → indicative
   only; do not treat as a settled effect. Effect sizes are tiny in QLIKE units.
4. Large-n significance: VN100 test n ≈ 43k (3x VN30). DM power scales with n, so even negligible
   QLIKE differences can reach p<0.05 on VN100; read the effect magnitude alongside the p-value.
5. Epoch cap differs between the two runs (8 vs 12), a second reason absolute comparisons are loose.

## Verdict — does more data help?

- The run works on VN100 with 218,462 pooled training observations (3.0x the VN30 ~73k).
- Absolute metrics do NOT visibly improve — they are worse in level (lower R2, higher QLIKE), but
  that is the universe changing, not evidence against more data (caveat 1).
- The FULL-model-vs-HAR verdict is clearer with more data at short horizons: on VN100 FULL beats HAR
  on QLIKE at both h1 (p<0.01) and h5 (p=0.01), whereas on VN30 the h5 QLIKE margin was a tie
  (p=0.60). At h10 HAR beats FULL on BOTH universes (VN30 +4.96*, VN100 +2.48*), so more data does
  not overturn HAR's advantage at the longer horizon.
- Component contributions do NOT become uniformly stronger, but several flip toward "helps" with more
  data: the vol→PK graph flips from significantly hurting on VN30 (h1/h5/h10 all +*) to significantly
  HELPING on VN100 h5 (−6.10*, p<0.001; neutral at h1/h10) — consistent with a graph benefiting from
  more nodes/edges (165 → 520). The gate flips from hurting on VN30 h10 (+5.46*) to helping on VN100
  h10 (−5.57*). News helps on VN100 h10 (−4.51*) but hurts on VN100 h5 (+4.39*) and is neutral at
  h1; its estimate is confounded by zero-fill for 68% of tickers (caveat 2).
- Overall: more pooled data makes the FULL-vs-HAR win more robust at short horizons and turns
  graph/gate/news components positive in several cells (most cleanly the graph at h5), but does not
  raise absolute skill, does not beat HAR at h10, and does not fully rescue the news/gate components
  (under-powered by zero-fill). Indicative; worth a fuller multi-seed, news-complete follow-up
  before drawing conclusions.

## Artifacts
- Wrapper: `scripts/run_vn100_ablation.py`
- VN100 results: `results/trackA_ablation_h{1,5,10}_seed42_2026-08-16_vn100quick/`
- VN30 reference: `.worktrees/trackA-gat/results/trackA_ablation_h{1,5,10}_seed42_2026-08-15_085544_loo/`
