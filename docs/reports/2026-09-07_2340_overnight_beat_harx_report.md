# Overnight experiments to beat HAR-X at h5/h10/h22 (VN100 + VN30)

Goal: find a model that beats HAR-X on QLIKE at the long horizons where the deep models (LSTM/VolGA) only
tie it. Autonomous run 2026-09-07 night. All numbers honest; DM = date-clustered Diebold–Mariano on QLIKE.

## Result headline
**An equal-weight stack of VolGA (deep) + HAR-X-Q (HARQ-corrected linear) significantly beats HAR-X at
h1/h5/h10 on BOTH VN100 and VN30 (6/8 cells, gains +1.1…+3.6%)**, and is numerically best (never worse) at
h22. This is the strongest result: it beats HAR-X, VolGA, and HAR-X-Q individually. Two building blocks:

1. **HARQ** (realized-quarticity measurement-error correction, Bollerslev–Patton–Quaedvlieg 2016) beats
   HAR-X at 7/8 cells alone (small, consistent, significant).
2. **The stack** combines HARQ's *consistency* with VolGA's *lower average* — their errors are decorrelated
   (linear-quarticity vs deep-nonlinear), so a 1/N average cancels noise and beats both.

### The stack (equal weight w=0.5, a-priori — no fitting) vs HAR-X
| market | h | HAR-X | VolGA | HAR-X-Q | **stack** | gain | DM p |
|---|---|---|---|---|---|---|---|
| VN100 | 1 | 0.5000 | 0.4833 | 0.4976 | **0.4817** | +3.65% | 0.0000 ✓ |
| VN100 | 5 | 0.5607 | 0.5565 | 0.5594 | **0.5525** | +1.47% | 0.0013 ✓ |
| VN100 | 10 | 0.5999 | 0.5985 | 0.5989 | **0.5929** | +1.15% | 0.0425 ✓ |
| VN100 | 22 | 0.6385 | 0.6436 | 0.6377 | **0.6322** | +0.99% | 0.284 |
| VN30 | 1 | 0.4801 | 0.4639 | 0.4763 | **0.4633** | +3.49% | 0.0000 ✓ |
| VN30 | 5 | 0.5602 | 0.5576 | 0.5579 | **0.5524** | +1.39% | 0.0421 ✓ |
| VN30 | 10 | 0.6091 | 0.6045 | 0.6083 | **0.6007** | +1.37% | 0.0496 ✓ |
| VN30 | 22 | 0.6782 | 0.6782 | 0.6785 | **0.6675** | +1.59% | 0.358 |

`baselines/2026-09-07_deep_harq_stack/`; results `results/deep_harq_stack/stack_{market}_h{h}.json`.
**Honest control:** a weight fit on validation leans VolGA-heavy (0.75–1.0) and does WORSE out-of-sample than
the equal weight — the forecast-combination puzzle (DeMiguel et al. 2009). So the reported weight is the
a-priori 0.5, not a tuned one. h22 is numerically best on both markets but not significant.

### The HARQ component alone (7/8 significant)

| market | h | HAR-X | HAR-X-Q | gain | DM p | sig |
|---|---|---|---|---|---|---|
| VN100 | 1 | 0.5000 | 0.4976 | +0.48% | 0.0000 | ✓ |
| VN100 | 5 | 0.5607 | 0.5594 | +0.23% | 0.0001 | ✓ |
| VN100 | 10 | 0.5999 | 0.5989 | +0.16% | 0.0000 | ✓ |
| VN100 | 22 | 0.6385 | 0.6377 | +0.12% | 0.0493 | ✓ (borderline) |
| VN30 | 1 | 0.4801 | 0.4763 | +0.80% | 0.0000 | ✓ |
| VN30 | 5 | 0.5602 | 0.5579 | +0.39% | 0.0000 | ✓ |
| VN30 | 10 | 0.6091 | 0.6083 | +0.13% | 0.0382 | ✓ |
| VN30 | 22 | 0.6782 | 0.6785 | −0.04% | 0.7200 | ✗ |

The HAR-X baseline reproduces the canonical HAR-X QLIKE exactly (validates the walk-forward replication), so
the only difference is the single added HARQ regressor. Baseline: `baselines/2026-09-07_harq/`; results
`results/harq/harq_{market}_h{h}.json`.

### What HARQ is
HAR-X regresses future variance on `[pk, har_w, har_m, market_pk, volume_zscore]`. HARQ adds one term,
`har_daily · sqrt(RQ)`, where RQ is a realized-quarticity proxy (rolling-5 mean of `pk²`, a
variance-of-variance). It makes the daily-RV coefficient time-varying: when the daily estimate is noisy
(high RQ) the model attenuates it. Robustness (`scripts/eda/harq_deepdive.py`): significant at h10 across
train-start windows 2016/2018/2020 (p<0.02) on both markets; the daily-only form is robust, the full HARQ-F
(all terms interacted) is unstable and non-significant; the pk² and range⁴ RQ proxies give identical results.

## What did NOT work (all non-significant at long h on VN)
| lever | long-h result | why |
|---|---|---|
| Regime blend (deep ↔ HAR-X) | ~+0.2–1.9%, n.s. | deep and HAR-X share the same decayed own-vol signal |
| Long-memory HAR (+quarterly 66d, +semiannual 132d) | ~+1.0%, n.s. | still own-vol; redundant with the 22d monthly term |
| VIX (SP500, exogenous) | −2 to −4%, **worse** | redundant with market_pk; variance-risk-premium over-forecast bias |
| Mean-reversion features (slope/change/dev) | ~0%, n.s. | redundant with HAR-X's multi-scale terms (linear) |
| SHAR / semivariance / leverage | ~0% or worse | daily-only semivariance proxy too weak on this data |
| MR-regularizer `QLIKE + λ·(log f − log HARX)²` (VolGA) | see below | trades calm↔storm; best gated_bi +1% n.s. |

### The MR-regularizer study (VolGA h5, shrink-to-HAR-X in the training loss)
Added a mean-reversion penalty to VolGA's QLIKE training loss (`qa_train.py`, knobs
`--mr-lambda/--mr-mode/--mr-ksteep`). Per-decile evidence (VN100 h5):
- **symmetric** `λ·(log f−log HARX)²`: best λ=1 overall 0.5554 (+0.96% n.s.); HURTS calm D0–D4 (+0.015),
  HELPS storm D5–D9 (−0.019) — a trade, confirming a global shrink erodes the calm-cell edge.
- **gated one-sided** (only over-forecast, high-vol regime): OPPOSITE — helps calm, HURTS storm D9 (+0.16).
  The gate on HAR-X level cannot separate a reverting storm tail from an ongoing storm, and storm QLIKE
  damage is bidirectional (missed spikes = under-forecast), which a one-sided clamp does not fix.
- **gated bidirectional** (gate + both directions): best mode — uniform small gain across all deciles
  (overall 0.5548, +1.07%), preserves calm, but still **not significant** (p=0.389).

Conclusion of the shrink-to-HAR-X family: at long horizons any shrink strong enough to fix the storm cells
touches the calm cells; the best design caps at ~1% and never reaches significance. The exploitable signal
is not in a common HAR-X target.

## Root cause of the h5+ failure (from the per-decile + delayed-echo analysis)
At h5/h10 the deep forecast is a spike echo delayed by ~h days (structural h-step lag); it over/under-shoots
the mean-reverted storm tail, so on storm deciles D5–D9 VolGA is worse than HAR-X (h5: 0.4553 vs 0.4249)
while it wins the calm deciles (0.6596 vs 0.6966). The two roughly cancel → a tie. Deep nonlinearity and
extra own-vol features do not add long-horizon information (Christoffersen–Diebold 2000: own-vol
forecastability decays with horizon). The only thing that helped was correcting HAR-X's **measurement
error** (HARQ) — an orthogonal, statistically-consistent, small effect — not adding more signal.

## Recommendation / next steps
1. **Adopt HAR-X-Q** as the linear baseline: it dominates HAR-X on QLIKE at h1/h5/h10 on both markets
   (significant) at zero extra data cost. Honest caveat: gains are small (0.1–0.8%) and VN30 h22 is neutral.
2. **VolGA + HARQ (follow-up, GPU):** the deep model wins h1 and is numerically lowest at h5 (0.5574) but
   non-significant (seed noise), whereas HARQ's gain is small but consistent (significant). Adding HARQ as a
   6th node feature to VolGA could combine a consistent AND larger edge — worth testing (needs a frozen-config
   change: N_NODE_FEATURES 5→6).
3. Blend VolGA with **HAR-X-Q** (instead of HAR-X) in `baselines/2026-09-07_regime_blend/`.

## Reproduce
- HARQ walk-forward: `python baselines/2026-09-07_harq/code/harq_walkforward.py --market vn100 --horizon 5`.
- Probes (local): `scripts/eda/{harfamily_probe,harq_deepdive,longmem_har_probe,vix_sp500_probe,meanrev_har_probe}.py`.
- MR-regularizer: `run_qlike_anchor.py --mr-lambda L --mr-mode {sym,gated,gated_bi} --cells-out <path>`.
