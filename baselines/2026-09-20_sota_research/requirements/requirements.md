# Overnight SOTA research + experiments — stock volatility forecasting

Date: 2026-09-20 (overnight, autonomous). Author-requested: research + run new SOTA / cross-domain /
agentic / explainable-AI / trustworthy-AI directions, with quality gates, record roadmap + HTML results.

## Grounding (what we already know — do NOT re-tread)
- **Own-history gamma-XGBoost (VolTree) is at the QLIKE frontier** for this data. Under-dispersion is
  Bayes-optimal shrinkage, not a bug. Beating point-QLIKE with external info is largely saturated.
- **NO-GO history (do not repeat):** cross-firm graphs (corr/sector/news co-mention), GNN families
  (GCN/GAT/SAGE dilute own-AR), foundation models standalone (TimesFM/Chronos 60-90x worse than HAR),
  DTW features (overfit), LSTM/GBME error-corr blends (~0.1), stacking/DAE/GLM-anchor, incremental
  news, shock-detection/limit-lock hurdle, richer-feature EDA (except market-dispersion), topology→OOS
  (assoc≠predictive). Foundation/GNN as FEATURES gave only tiny wins (estimators +0.4%, per-stock
  embedding +1.35% h1).
- **Real levers found:** earnings (SP500 +2.7-3.2% DM-sig, HOSE inert), leaf-cooccurrence graph
  smoothing (HOSE h1/h5 +0.11-0.22%), GK/RS/YZ estimators (+0.4%). Kalman storm-overlay = only honest
  UNTRIED point-QLIKE lever noted.
- **Mandates:** leak-free (α on validation), QLIKE + date-clustered DM, HOSE COVID/2022/tariff
  spike-robustness, over/underfit evidence in result JSONs, baselines/ 5-subfolder structure.

## Strategic pivot (honest, high-value)
Point-QLIKE is saturated → the highest-value NEW contributions are on **different axes** where the
paper currently says nothing, and which suit the tree-based VolTree:
1. **Distributional / calibrated forecasting** (uncertainty, not just point) — CRPS, conformal
   prediction intervals, coverage. VolTree already gamma → natural probabilistic forecast.
2. **Explainable AI (XAI)** — SHAP global/local, per-market/per-horizon attributions, leaf-graph as
   an interpretability construct. Tree-based ⇒ exact TreeSHAP.
3. **Trustworthy AI** — calibration/reliability, robustness (spike), stability, uncertainty-aware
   decisions, cross-market fairness.
These do NOT compete where own-history is saturated; they add a publishable follow-up axis.

## Research tracks (parallel agents surveying 2024-2025 SOTA)
- **A. SOTA sequence architectures 2024-25:** Mamba/SSM, KAN, iTransformer, TSMixer, TimeMixer,
  TimeXer, SOFTS — as RESIDUAL learners over VolTree (only residual room), not standalone.
- **B. Probabilistic + rough volatility:** conformal (split/CQR/EnbPI), CRPS/pinball, distributional
  boosting (NGBoost/XGBoost-distribution), rough-vol (rough Bergomi, log-vol Hurst, fractional).
- **C. Agentic AI for forecasting 2024-25:** LLM-agent hypothesis/feature discovery, agent-team
  ablation/model-selection, multi-agent forecasting frameworks; feasibility on public/PII-safe data.
- **D. Explainable + Trustworthy AI for finance/vol:** TreeSHAP, concept-based, counterfactual,
  calibration, conformal-as-trust, uncertainty quantification, robustness certification.
- **E. Volatility-forecasting-specific SOTA lit 2024-25:** neural-HAR, HARNet, realized-vol DL,
  regime-switching/Kalman overlays, intraday-to-daily transfer.

## Experiments runnable tonight (with existing data + gamma-XGBoost pipeline)
- **Exp 1 — XAI:** TreeSHAP on VolTree (SP500 + HOSE, all horizons): global feature importance,
  per-market/per-horizon attributions, dependence, local examples. Verifies HAR-lag dominance +
  earnings/leaf-graph roles. Deterministic, fast.
- **Exp 2 — Conformal / calibrated intervals:** split-conformal + CQR on VolTree forecasts,
  walk-forward, leak-safe (calibrate on validation). Report empirical coverage vs nominal + mean
  interval width + CRPS, per market/horizon. NEW capability.
- **Exp 3 — Rough-vol features (honest test):** add log-vol Hurst / rough-vol features to OWN-8,
  QLIKE + DM vs VolTree (expect marginal per prior; record honestly).
- **Exp 4 (if time / GPU) — 2024 arch residual:** Mamba or KAN as residual over VolTree on SP500,
  QLIKE + DM + overfit evidence.

## Success criteria / go-no-go
- Each experiment: 5-metric eval + date-clustered DM where a comparison is claimed + spike-robustness
  on HOSE + over/underfit evidence + leak-safety note. NO-GO recorded honestly (no p-hacking).
- Deliverables: this roadmap; per-experiment result JSON/HTML; consolidated HTML report at
  `docs/reports/2026-09-20_overnight_sota/`; memory update with findings.
