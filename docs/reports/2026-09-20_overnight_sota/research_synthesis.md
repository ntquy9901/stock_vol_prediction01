# SOTA research synthesis (2024-2025) for stock-volatility forecasting — overnight 2026-09-20

Four parallel research agents surveyed the literature. Filtered through this project's ground truth:
own-history gamma-XGBoost ("VolTree") is at the QLIKE frontier; deep/graph/foundation models have been
NO-GO; earnings (SP500 +2.7-3.2%) and the leaf-cooccurrence graph (HOSE h1/h5) are the real levers.
Citations were web-verified by the agents; a few future-dated arXiv ids were flagged unverifiable and
excluded. **Bottom line: the honest high-value directions are NOT more point-QLIKE chasing but new axes
— calibrated intervals (conformal), explainability (TreeSHAP), and trustworthy-AI — which suit the
tree-based model and where the paper currently says nothing.**

## Track A — SOTA sequence architectures 2024-25 (residual-learner framing)
- **Governing filter:** the VolTree residual is what own-history + trees cannot explain; only
  CROSS-SECTIONAL structure (not univariate) can help. So univariate SSMs (Mamba, KAN) and foundation
  models see the same own-history the tree saturated → NO-GO by construction.
- **External confirmation of NO-GO for foundation models:** "Forecasting Realized Volatility with TSFMs
  vs Econometric Benchmarks" (arXiv:2607.05291, unverified id) — Moirai/TimesFM/Chronos/Toto QLIKE
  ratios >1.0 vs Log-HAR at all horizons; zero-shot never beats HAR (corroborates Goel 2025, Rahimikia
  2025). Matches this project's local finding.
- **Only worth a probe (cross-channel MLPs on the residual PANEL):**
  1. **iTransformer** (arXiv:2310.06625, ICLR 2024) — attention across variates/stocks; inductive bias
     = cross-series correlation, exactly the untapped axis.
  2. **SOFTS** (arXiv:2404.14197, NeurIPS 2024) — MLP + STar Aggregate-Redistribute core; noise-robust,
     fits thin VN markets better; best risk-adjusted second.
  3. **TimeMixer** (arXiv:2405.14616, ICLR 2024) — multiscale decomposition; regime/storm-onset angle.
- **NO-GO (report-complete):** Mamba/S-Mamba (2403.11144), KAN (2404.19756) + KAN4TSF (2408.11306),
  TimeXer (2402.19072, exogenous already in tree), TSMixer (2303.06053, superseded).
- **Verdict:** one shared experiment (iTransformer+SOFTS on the per-stock VolTree-residual panel, val-fit
  blend, DM + HOSE spike gate) decisively tests whether cross-sectional residual structure exists. Base
  rate favors NO-GO (own-history saturation; prior GNN-dilution). Deferred (GPU + build cost); not tonight.

## Track B — Probabilistic / conformal / rough-vol  ⭐ TOP PICK
- **Conformal prediction = the paper's new axis (GO, done tonight = Exp 2):**
  - **CQR** (Romano et al., NeurIPS 2019, arXiv:1905.03222) — two XGBoost quantile heads + conformalize
    on a calibration slice → adaptive intervals (wide in storms, tight in calm). Best fit.
  - Split conformal (baseline, constant width); **ACI/Conformal-PID** (Gibbs-Candès NeurIPS 2021
    arXiv:2106.00170; Angelopoulos NeurIPS 2023 arXiv:2307.16895) for coverage under regime shift.
  - Leak-safe recipe: fit on train-minus-val, conformity Q on the held-out VALIDATION slice, freeze for
    test — mirrors the existing α-on-validation pattern. Metrics: marginal coverage, width, conditional
    (spike vs calm) coverage, rolling coverage.
- **Distributional boosting (PARTIAL GO):** reg:gamma already implies a per-row mean; extract closed-form
  gamma quantiles via method-of-moments shape → CRPS/pinball with zero retraining. NGBoost (ICML 2020
  arXiv:1910.03225) / XGBoostLSS (arXiv:1907.03178) = full distributional baselines (fast follow).
- **Rough volatility (CAUTIOUS GO, expected null):** Gatheral et al. 2018 (arXiv:1410.3394) log-vol
  Hurst≈0.1; but Cont & Das (arXiv:2203.13820) "fact or artefact?" — measured roughness of realized
  variance is largely estimation artifact. OOS: own-path roughness rarely beats HAR (matches project).
  Options-implied rough-Heston helps but needs option data (unavailable for VN daily-OHLC). Add causal
  Hurst/variogram features as a documented falsification with spike-robustness.

## Track C — Agentic AI for forecasting 2024-25
- **Key negative:** "Are Language Models Actually Useful for Time Series Forecasting?" (NeurIPS 2024,
  arXiv:2406.16964) — removing/re-initializing the LLM does NOT degrade forecasts. **LLMs are not a
  credible numerical volatility engine.** ECC-Analyzer (ICAIF'24, arXiv:2404.18470) even states directly
  prompting an LLM for volatility ≈ random guessing.
- **Where agentic helps = reasoning/orchestration, not the number:** TSAG (LLM orchestrates verified
  numeric tools), "Bridging the Last Mile" (arXiv:2606.02497, LLM as bounded auditable revision layer on
  a numeric forecast). Multi-agent trading stacks (TradingAgents 2412.20138, FinAgent, FinVision) target
  RETURN/PnL not variance-QLIKE → skip; critique arXiv:2603.27539 shows the multi-agent part rarely helps.
- **Most transferable + PII-safe offline prototype:** LLM-driven automated FEATURE DISCOVERY over the
  existing numeric own-history columns (CAAFE NeurIPS; LLM-FE 2025): a generator LLM proposes causal
  transforms as code, a critic LLM screens for look-ahead (per the leakage boundary rule), survivors are
  fit in the existing walk-forward and gated by QLIKE + DM + spike-robustness. Purely numeric, no news,
  no PII. Expected mostly NO-GO (sweeps exhausted) but a publishable honest test of "can an agent
  out-discover hand-crafted own-history features." Future prototype (needs LLM API).

## Track D — Explainable + Trustworthy AI  ⭐ RUNNABLE (Exp 1 done; more tonight)
- **Critical rigor note:** gamma reg uses a LOG LINK, so TreeSHAP is additive in **log-margin space**,
  not raw variance — state this. (Our Exp 1 uses XGBoost native `pred_contribs` = exact margin SHAP.)
- **XAI core (cheap, tree-based):** exact TreeSHAP (Lundberg et al., Nature Mach. Intell. 2020,
  arXiv:1802.03888) global/local/interaction; interventional vs path-dependent (report which); SHAP
  interaction values (Beyond-TreeSHAP, arXiv:2401.12069) → motivates the leaf-graph; dependence/ALE;
  marginal-vs-conditional permutation importance (Hooker-Mentch 2021; Chamma CPI arXiv:2309.07593) with
  the placebo caveat this project already hit.
- **Trustworthy AI:** conformal coverage as a trust guarantee (ties to Track B), calibration/reliability
  (Gneiting et al. JRSS-B 2007), ECE (Guo ICML 2017), CRPS, Kupiec/Christoffersen interval backtests,
  explanation stability across folds/regimes (Kendall τ; Quantus JMLR 2023), robustness under shift
  (reuse COVID/2022/tariff), SR 11-7 / SR 26-2 model-risk framing for finance-audit.
- **Leaf-graph = interpretability artifact:** it is a supervised RF-PROXIMITY / leaf-kernel (Breiman
  2001; Davies-Ghahramani "Random Forest Kernel" arXiv:1402.4293). Explain via Louvain/Leiden communities
  (discovered peer-groups vs sector/liquidity), centrality (hub stocks the smoother most alters), and
  cross-regime community stability — a XAI result specific to this paper's contribution.
- **Evaluation to report:** faithfulness deletion-curve (ΔQLIKE vs top-k SHAP removed), explanation
  robustness (RIS/Kendall τ), interval coverage/ECE/CRPS, Kupiec+Christoffersen, DM/MCS.

## Prioritized experiment plan (this session)
1. **Exp 1 — TreeSHAP explainability of VolTree** (running): global mean|SHAP| per market/horizon,
   HAR/momentum/earnings group shares, ranking. Rigorous alternative to placebo-prone permutation.
   Smoke (HOSE h1): HAR 85.8% (weekly 46 / monthly 26 / daily 14), momentum 13.2%, earnings 1.1%.
2. **Exp 2 — Conformal intervals (split + CQR)** (DONE, h1, train-capped 250k for speed): target 90%.
   **SP500 CQR = 89.9% coverage (near-nominal!)**, split 87.6% (under, expected for constant-width on
   heteroscedastic vol); per-regime CQR spike 88% / calm 90%. **HOSE under-covers: CQR 82.5%, split
   80.5%** (thin/fat-tailed VN market breaks exchangeability more; calibration on 22 val dates misses the
   tails) — an honest finding that MOTIVATES adaptive/online conformal (ACI/PID) as the HOSE fix. CQR is
   adaptively wider than split (SP500 7.5e-4 vs 5.6e-4). NEW calibrated-uncertainty axis works on SP500;
   VN needs adaptive conformal. (Train-cap is a speed subsample; conformal coverage is model-agnostic so
   the cap does not explain the HOSE gap — exchangeability does.)
3. **Exp 3 (fast follow)** — gamma-quantile CRPS/pinball extraction (near-free); optionally rough/Hurst
   feature falsification with spike-robustness.
4. **Deferred (build/GPU/API):** iTransformer+SOFTS residual-panel probe; LLM feature-discovery agent.

Full citation lists are in the four agent transcripts; the verified anchors are inline above.
