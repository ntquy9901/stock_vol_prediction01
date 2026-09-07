# QLIKE-loss anchor provenance and the h5/h10 forecast-failure mechanism

Scope: three linked analyses on the `qlike_anchor` (anchor=none) TEST forecasts, verifying (1) which
training variant feeds the paper, (2) why the deep model wins the calm deciles, and (3) where and why
the deep model's edge fails at h5/h10. Companion visual: `docs/reports/2026-09-07_perstock_daily_dashboard.html`.

Source cells: `results/qlike_anchor/cells/cells_{sp500_clean,vn100,vn30}_qlike_none_h{1,5,10,22}.parquet`
(SP500 cells re-fetched from Drive for the per-day episodes; ~1.28 GB each, gitignored). Per-decile
aggregates: `results/qlike_anchor/decile_summary.json` (committed). Training code:
`baselines/2026-09-07_qlike_anchor/code/qa_train.py`.

Cell schema: one row per (model, split, ticker, date); `y_true[t]` is the realized Parkinson variance on
day t (identical across horizons); `y_pred` at horizon h is the forecast FOR day t made h days earlier
(verified: `corr(y_h5[t], y_h1[t]) = 1.000` for `y_true`). A larger h therefore uses staler information.

## 1. The paper's QLIKE numbers use anchor=none, not anchor=harx, and are not an average

`qa_train.py` exposes two knobs: `loss ∈ {mse, qlike}` and `anchor ∈ {none, harx}`.

- `anchor=none` (`qa_train.py:114`, `zscore_forecast`): forecast = `max(z_out·std + mean, floor)` — the deep
  model predicts the level directly. HAR-X never enters the loss or the forecast.
- `anchor=harx` (`qa_train.py:113`, `anchor_forecast`): forecast = `HAR-X · exp(clip(z_out, −c, c))` — a
  bounded residual on top of HAR-X (`z=0` falls back to HAR-X). This is the only variant that couples the
  deep model to HAR-X during training.

The main paper tables read from `results/edge_hmatched/edgehm_{vn100,vn30}_h*.json`, where the learned
models are trained on the MSE loss (paper `soict_harlstmgat_2026-09-07_final.tex:144,449`); QLIKE there is
the evaluation metric only. The QLIKE-loss ablation (`\subsection` at `.tex:448`, Table `tab:qlikeloss`)
reads the `anchor=none` numbers. Verification (VN100, h1 QLIKE):

| source | LSTM | VolGA |
|---|---|---|
| `qa_vn100_qlike_none_h1.json` | 0.4840 | 0.4833 |
| paper Table (LSTM_QL / VolGA_QL) | 0.4840 | 0.4833 |
| `qa_vn100_qlike_harx_h1.json` | 0.4903 | 0.4905 |
| average of none+harx | 0.4872 | 0.4869 |

The paper values match `none` exactly. `anchor=harx` is worse than `none` at every horizon/market and is
reported only as a negative diagnostic (a global HAR-X anchor drags the deep model toward HAR-X on the calm
cells and removes the calm-cell edge); it contributes no number to the paper. No reported QLIKE is an
average of the two variants.

## 2. The calm-decile (D0–D4) edge is learned from the QLIKE objective, not from anchoring

On the calmest decile both models over-forecast the realized floor, but the QLIKE-trained deep model
(anchor=none, no coupling to HAR-X) over-forecasts less, so its D0 QLIKE is lower. VN100, D0 (calmest
decile of realized variance):

| VN100 h22, D0 | forecast/realized (median) | D0 QLIKE (mean) |
|---|---|---|
| HAR-X | 11.16× | 1.906 |
| LSTM (anchor=none) | 9.82× | 1.734 |

QLIKE penalises over-forecasting a small `y` asymmetrically, so a model minimising QLIKE lowers its
calm-day forecast; the linear HAR-X (OLS) does not lower it as far. The mechanism is independent learning
from the objective, not an anchor pulling the forecast toward HAR-X. (The word "nails" in the earlier
edge-decay report refers to HAR-X capturing mean-reversion on high-vol nodes, not to anchoring.)

## 3. Where and why the deep edge fails at h5/h10

### Where (per-decile) — the storm-decile flip
`LSTM − HAR-X` QLIKE by realized-vol decile (`decile_summary.json`; negative = deep better):

| decile D9 (stormiest) | h1 | h5 | h10 |
|---|---|---|---|
| S&P 500 | −0.007 | +0.13 | +0.15 |
| VN100 | −0.067 | +0.086 | +0.036 |

At h1 the deep model beats HAR-X on every decile. At h5/h10 it keeps the calm-decile win (D0–D4) but
turns worse than HAR-X on the storm deciles D5–D9. Those storm-decile losses are the cells that erase the
deep model's h5/h10 edge; on S&P 500 they are largest.

### Why (per-day) — the deep forecast is a spike echo delayed by ~h days
Because `y_pred` at horizon h is the forecast for day t made h days earlier, the deep forecast reacts to a
spike with a delay equal to the horizon. On a storm episode the h5/h10 forecast peaks AFTER the market has
already reverted, so it over-forecasts the post-storm calm days — exactly the D5–D9 cells above. HAR-X is a
flatter long-memory average that reverts sooner and takes a smaller QLIKE penalty on those tail days. No
model anticipates day-0 (VN and US storms here are exogenous — the Apr-2025 tariff shock), so the h5/h10
damage is the mis-timed echo, not the missed spike itself.

S&P 500, ticker MCHP, market storm peak 2025-04-09 (LSTM forecasts, log scale in the dashboard):

| date | realized | LSTM h1 | LSTM h5 | LSTM h10 |
|---|---|---|---|---|
| 2025-04-09 (spike) | 2.46e-2 | 5.10e-3 | 5.42e-4 | 5.58e-4 |
| 2025-04-16 (reverted) | 1.12e-3 | 1.71e-3 | 3.87e-3 | 5.83e-4 |
| 2025-04-24 (calm) | 1.78e-3 | 9.81e-4 | 1.90e-3 | 3.55e-3 |

The h5 echo peaks ~2025-04-16 (~5 trading days late; realized already reverted to 1.1e-3, forecast 3.9e-3,
≈3.5× over). The h10 echo peaks ~2025-04-24 (~10 trading days late; realized 1.8e-3, forecast 3.5e-3, ≈2×
over). The same pattern holds on VN100 (CTS, 2025-10-20) and VN30 (VCB, 2025-04-09).

## 4. What makes the forecast "late": the h-step information lag (structural), not a model defect

The forecast for target day t can only use data through t−h. A cross-correlation confirms the forecast is
essentially a lagged view of the realized series: the lag k that maximizes `corr(log forecast(t), log
realized(t−k))` is exactly h, for BOTH the deep model and HAR-X.

| best lag k (arg max corr) | h1 | h5 | h10 |
|---|---|---|---|
| VN100 LSTM | 1 | 5 | 10 |
| VN100 HAR-X | 1 | 5 | 10 |
| S&P 500 LSTM | 1 | 5 | 11 |
| S&P 500 HAR-X | 1 | 5 | 10 |

The lag-0 correlation (forecast vs same-day realized) is far lower (e.g. VN100 h10: 0.27 at lag0 vs 0.50 at
lag10), so the forecast tracks the day h steps earlier, not the target day. Two consequences:

- **The delay is structural and identical across models** — it is the h-step forecasting gap, unavoidable
  for any h-ahead forecast; it is not a deep-model bug and cannot be removed while still forecasting h ahead.
- **The delay becomes an error only because volatility mean-reverts within h days.** The deep model
  extrapolates the stale elevated state forward with high amplitude (sharp persistence); HAR-X averages over
  daily+weekly+monthly terms, so an isolated spike is diluted (1 of 22 days in the monthly term) → its echo
  is damped and reverts sooner → smaller QLIKE penalty on the reverted tail. The lookback window (SEQ=10)
  keeps the spike day in the input for ~10 days, sustaining the elevated deep forecast.

So the fixable part is not the delay (tier-1, structural) but the **echo amplitude** (tier-2): make the deep
model mean-revert on stale-elevated nodes at long horizons.

### Concrete example (h=5, one stock)
Number the trading sessions P0, P1, …; suppose a volatility spike hits at P10 and then mean-reverts. A
forecast for target session t may use data only through its origin t−5:

| target t | origin t−5 | origin has seen the P10 spike? | forecast | realized at t | outcome |
|---|---|---|---|---|---|
| P10 (spike) | P5 | no (P5 < P10) | low | very high | under-forecast the spike |
| P15 | P10 | just now (origin = spike day) | high (echo) | already reverted, low | over-forecast |
| P17 | P12 | yes (P12 still elevated) | high | low | over-forecast |

The forecast for P15 is high because its origin P10 is the spike day and the model projects that elevated
state 5 sessions forward; by P15 the market has reverted, so it over-forecasts. The forecast peak lands ~5
sessions after the true peak (P10 → P15); at h=10 it lands at P20. Because every target is anchored h
sessions back, the whole forecast curve is the realized curve shifted right by h — the cross-correlation
lag of exactly h measured above. The lag itself is identical for HAR-X and the deep model; only the echo
amplitude differs (HAR-X dilutes the one-day spike inside its weekly/monthly averages, so its echo is
smaller and reverts sooner).

### Literature basis
Tier-1 — volatility mean-reverts, so multi-step forecastability decays with horizon (the reason a long-h
echo of a transient spike is wrong): Christoffersen & Diebold (2000) give a model-free result that
volatility forecastability decays quickly with horizon (not predictable beyond ~10–20 trading days) [1];
corroborated/extended by West & Cho (1995), Raunig (2006/2008), Galbraith & Kisinbay (2005) (spot vs
forward accuracy) and the term-structure-of-volatility-predictability literature [3]. The HAR model
(Corsi, 2009) builds daily+weekly+monthly averages that approximate this mean-reversion, which is why its
echo is damped. Tier-2 — deep sequence models tend to learn the "copy the last value" shortcut, producing
a forecast shifted one step behind the target (persistence/"mimicking"); this is documented and targeted by
a copy-penalizing loss in "Time Series Forecasting Models Copy the Past: How to Mitigate" (arXiv:2207.13441)
[2]. The regime-blend fix (Section 5) enforces the tier-1 mean-reversion exactly where the tier-2 echo is
strongest (high-vol regime × long horizon).

References:
- [1] Christoffersen, P. & Diebold, F.X. (2000). *How Relevant is Volatility Forecasting for Financial Risk
  Management?* Review of Economics and Statistics 82(1):12–23 (NBER w6844).
  https://www.sas.upenn.edu/~fdiebold/papers2/Christoffersen-Diebold%20(2000).pdf
- [2] *Time Series Forecasting Models Copy the Past: How to Mitigate.* arXiv:2207.13441.
  https://arxiv.org/pdf/2207.13441
- [3] *The term structure of volatility predictability.* International Journal of Forecasting (2019).
  https://www.sciencedirect.com/science/article/abs/pii/S0169207019302341
- Andersen, Bollerslev, Christoffersen, Diebold. *Volatility and Correlation Forecasting* (survey).
  https://www.sas.upenn.edu/~fdiebold/papers/paper67/abcd.pdf
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.* Journal of Financial
  Econometrics 7(2):174–196.

## 5. Spike experiment — causal regime-conditional blend of deep and HAR-X

Test of the tier-2 fix. Blend `f = w·deep + (1−w)·HAR-X`, with w chosen per regime bin and per horizon.
Regime = the HAR-X forecast level (a causal recent-volatility proxy, known at origin t−h; the target-day
decile is used only for post-hoc diagnosis). Bin edges (10 quantile bins on HAR-X) and the per-bin w
(grid 0..1) are fit on the VALIDATION split of the walk-forward cells and applied unchanged to TEST — no
test leakage. All three models share the QLIKE floor 1e-8; significance = date-clustered Diebold-Mariano
(`stats.date_clustered_dm`). Code: `scripts/eda/regime_blend_spike.py`; results:
`results/qlike_anchor/regime_blend_spike.json`.

| panel | h | QLIKE deep | QLIKE HAR-X | QLIKE blend | Δ vs deep | DM blend<deep (p) | mean w |
|---|---|---|---|---|---|---|---|
| VN100 | 1 | 0.4840 | 0.5000 | 0.4820 | +0.41% | **0.012** | 0.81 |
| VN100 | 5 | 0.5601 | 0.5607 | 0.5559 | +0.74% | 0.070 | 0.78 |
| VN100 | 10 | 0.5973 | 0.5999 | 0.5934 | +0.64% | 0.081 | 0.77 |
| VN100 | 22 | 0.6390 | 0.6385 | 0.6324 | +1.04% | **0.001** | 0.72 |
| VN30 | 1 | 0.4634 | 0.4801 | 0.4623 | +0.24% | 0.153 | 0.84 |
| VN30 | 5 | 0.5585 | 0.5602 | 0.5559 | +0.48% | **0.041** | 0.77 |
| VN30 | 10 | 0.6037 | 0.6091 | 0.6021 | +0.26% | **0.021** | 0.89 |
| VN30 | 22 | 0.6777 | 0.6782 | 0.6692 | +1.25% | **0.005** | 0.74 |
| S&P 500 | 1 | 0.3603 | 0.4061 | 0.3550 | +1.46% | **0.000** | 0.67 |
| S&P 500 | 5 | 0.4448 | 0.4550 | 0.4397 | +1.16% | 0.208 | 0.64 |
| S&P 500 | 10 | 0.4714 | 0.4797 | 0.4650 | +1.36% | 0.172 | 0.75 |

(S&P 500 h22 cells not fetched.) Findings:

- The blend **never hurts**: QLIKE improves over the deep model at every panel/horizon (+0.24% … +1.46%),
  and beats HAR-X significantly at h1 on all three panels. The gain is significant vs the deep model in
  about half the cells (all VN h22, VN30 h5/h10, h1 on VN100 and S&P 500).
- The fit reproduces the mechanism automatically: `mean w` falls with horizon and is lowest on S&P 500
  (more shrinkage toward HAR-X exactly where the deep model over-forecasts most).
- Per-decile decomposition confirms the intent — the blend keeps the deep model on the calm deciles and
  recovers HAR-X on the storm deciles (S&P 500 h10, D9: deep 1.785 → blend 1.670; VN100 h22, D9: 1.800 →
  1.751), with per-bin weights dropping to 0.0–0.35 in the storm-regime bins.

Verdict: the tier-2 fix works and is leakage-safe, but the magnitude is modest (sub-1.5% QLIKE). It
justifies a full leave-one-out baseline (smoother/2-D regime signal, per-fold refit, all metrics + MCS),
not a claim of a large gain.

## 6. Baseline — per-fold, 2-D-regime logistic blend (`baselines/2026-09-07_regime_blend/`)

The spike is promoted to a proper baseline with two fixes: **per-fold refit** (weights for fold k come only
from fold k's validation block — removes the pooled-val look-ahead the spike had) and a **smooth 2-D causal
gate** `w = sigmoid(θ0 + θ1·z_level + θ2·z_disagree)`, `z_level = log(HAR-X)` (recent-vol regime),
`z_disagree = log(deep) − log(HAR-X)` (echo signal), θ fit on val QLIKE per fold (Nelder–Mead). Frozen
forecasts, no retraining. Code `code/regime_blend.py` + `code/run_regime_blend.py`; result
`results/qlike_anchor/regime_blend_result.json`; tests 12/12 pass, 100% line + branch coverage.

Test QLIKE (deep base = VolGA), Δ vs HAR-X, date-clustered DM (blend vs HAR-X):

| market | h | deep(VolGA) | HAR-X | blend | Δ vs HAR-X | DM p | verdict |
|---|---|---|---|---|---|---|---|
| VN100 | 1 | 0.4833 | 0.5000 | 0.4857 | +2.84% | **0.000** | beats HAR-X (sig) |
| VN100 | 5 | 0.5565 | 0.5607 | 0.5589 | +0.32% | 0.780 | ≤ HAR-X, n.s. |
| VN100 | 10 | 0.5985 | 0.5999 | 0.5977 | +0.36% | 0.796 | ≤ HAR-X, n.s. |
| VN100 | 22 | 0.6436 | 0.6385 | 0.6371 | +0.23% | 0.927 | ≤ HAR-X, n.s. |
| VN30 | 1 | 0.4639 | 0.4801 | 0.4698 | +2.14% | **0.001** | beats HAR-X (sig) |
| VN30 | 5 | 0.5576 | 0.5602 | 0.5556 | +0.81% | 0.613 | ≤ HAR-X, n.s. |
| VN30 | 10 | 0.6045 | 0.6091 | 0.5975 | +1.90% | 0.194 | ≤ HAR-X, n.s. |
| VN30 | 22 | 0.6782 | 0.6782 | 0.6712 | +1.04% | 0.791 | ≤ HAR-X, n.s. |

(LSTM base is comparable: h1 significant on both panels; longer horizons ≤ HAR-X but not significant.)

Findings:

- **h1: the blend beats HAR-X on QLIKE with DM significance** on both VN100 and VN30 (p ≤ 0.001), for both
  deep bases — the daily-horizon deep edge is retained and combined with HAR-X.
- **h5/h10/h22: the blend's QLIKE is ≤ HAR-X at every cell but the difference is not significant** (DM
  p ≫ 0.05). This is the genuine ceiling from Sections 3–4, not a defect: at longer horizons the deep signal
  does not add QLIKE value over HAR-X, so a convex combination of the two cannot significantly beat either.
- The blend is **no-harm**: it never loses to HAR-X on QLIKE, and `mean_w ≈ 0.63–0.81` (auto-shrinks toward
  HAR-X more at longer horizons), so it degrades gracefully to HAR-X where the deep model is weak.
- Minor cost: at h1 the blend is a touch worse than the pure deep model (e.g. VN100 VolGA 0.4833 → 0.4857)
  because the val-fit gate shrinks slightly toward HAR-X; still far better than HAR-X.

Go/no-go (per `requirements/requirements.md`): MUST (never worse than HAR-X) met; TARGET (significant beat
at all four horizons) NOT met — only h1 is significant. The honest outcome is a significant daily-horizon
improvement plus a no-harm calibration at longer horizons, not an all-horizon win over HAR-X.

## Dashboard
`docs/reports/2026-09-07_perstock_daily_dashboard.html` (generator: `build_perstock_daily_dashboard.py`):
- Section A — `LSTM − HAR-X` QLIKE by decile for S&P 500 + VN100 + VN30 at h1/h5/h10 (the storm-decile flip).
- Section B — per-day storm episodes (S&P 500 MCHP, VN100 CTS, VN30 VCB) on a log axis, each horizon's echo
  peak annotated, showing the delayed-echo over-forecast.
- Section C — per-stock daily realized vs forecasts with market storm days shaded (VN context).

## Implication for the improvement direction
The failure is a timing/regime effect confined to the storm deciles at longer horizons (Sections 3–4). The
Section 5 spike shows a causal, regime-conditional blend targets it without the global HAR-X anchor that
removes the calm-cell edge (Section 1): it preserves the deep model on calm nodes and recovers HAR-X on
storm nodes, improving QLIKE over the deep model at every panel/horizon (significant in about half the
cells). The gain is modest (sub-1.5%). Next step: a full leave-one-out baseline
(`baselines/2026-09-07_regime_blend/`) with a smoother/2-D causal regime signal (e.g. the deep−HAR-X
disagreement in addition to the HAR-X level), per-fold refit, all five metrics, and an MCS/DM panel — not a
claim of a large gain from this spike alone.
