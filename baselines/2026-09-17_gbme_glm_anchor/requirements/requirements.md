# Requirements — Gamma-GLM base-margin anchor + GBDT residual (Hướng 3)

Date: 2026-09-17. Lifecycle: SDD-lite (Specify → Plan → Tasks → TDD → Validate). Constitution: `CLAUDE.md`.

## 1. Problem & motivation
The delivered HAR-anchored residual (E8, `baselines/2026-08-21_har_anchored_residual`) failed partly from a
**loss-function mismatch**: HAR is fit by OLS (MSE) but the target metric is QLIKE, so the anchor biases calm-day
forecasts and hurts QLIKE. This baseline tests the *methodologically correct* anchor: a Stage-1 **Gamma GLM with
log-link, fit by IRLS on the gamma deviance (≡ QLIKE)**, used as the **base margin** for a Stage-2 gamma GBDT
residual. This keeps the anchor loss-consistent and adds the one thing trees structurally lack — **linear
extrapolation beyond the training range** (relevant for storm days).

## 2. Hypothesis
- **H1:** an anchor fit on the SAME loss (gamma deviance) does NOT bias QLIKE, so GLM-anchored GBDT is at least
  not worse than plain GBDT, and its linear-extrapolation term may lower QLIKE on storm/tail days.
- **Prior (honest):** LOW. The champion own-history gamma-GBM is already at the QLIKE frontier (test R² 0.07–0.19,
  under-dispersion is Bayes-optimal, storm onset exogenous — `docs/reports/2026-09-15_2230_gbme_improvement…md`).
  Linear extrapolation of own-history cannot predict exogenous storm onset. This is a clean falsification.

## 3. Inputs / features
- Champion panel via `full_matrix.panel` (own-history OWN-8 single-sourced from `2026-09-13_paper_models`
  config; `+EARN` when the market carries earnings). Target `y` = h-step-ahead Parkinson variance σ².
- Markets: HOSE (primary here); horizons h ∈ {1, 5, 10, 22}. Same walk-forward folds/embargo/floor as the
  sibling GBME battery (`vn_gbm_graph_stage1.FOLDS`, `full_matrix.FL`).

## 4. Models (same library = XGBoost, to isolate the anchor effect)
- **GBME** — canonical champion `HistGradientBoostingRegressor(loss="gamma")` (validity anchor / real target to beat).
- **XGB** — plain XGBoost `reg:gamma`, no base margin (XGBoost reproduction of the champion).
- **GLM+XGB** — XGBoost `reg:gamma` with `base_margin = η_GLM` (Stage-1 Gamma GLM log-link); final
  ŷ = exp(η_GLM + Σ trees).

## 5. Outputs
- `results/gamma_gbm/glm_anchor_<market>_h<h>.json`, one per horizon (atomic checkpoint), carrying all-5 metrics
  (train/val/test) for every model + fit_diagnostics + DM (GLM+XGB vs GBME, and vs XGB) + gain + HOSE per-fold
  QLIKE + regime-spike robustness.

## 6. Acceptance criteria (Go / No-Go, pre-registered)
- **Success (Go):** GLM+XGB beats GBME on QLIKE with DM p < 0.05 at BOTH h1 and h5, **and** the gain survives
  regime-spike exclusion (COVID-2020 / 2022 / Apr-2025; sign holds). Else **NO-GO** (expected).
- **Mechanism read:** GLM+XGB vs XGB (same library) isolates the base-margin contribution.
- C0 line coverage = 100% on the GLM/anchor logic; prefix/causality: no future rows influence a fold's fit.
