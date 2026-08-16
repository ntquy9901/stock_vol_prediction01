# Long-Horizon Daily Volatility Forecasting: Research Survey and Recommendations

Date: 2026-08-16
Scope: How to improve model accuracy at LONG horizons (h10, h22 trading days) for
DAILY (not intraday) stock-volatility forecasting, and how to close/beat the HAR gap
observed for this project at h10/h22 on QLIKE.

Project this targets: direct forecast of single-day Parkinson VARIANCE at t+h, horizons
h in {1, 5, 10, 22}. Universe VN30 (pooled). Model: per-ticker LSTM + cross-stock GAT
(directed volume->volatility edge) + gated news branch; HAR = linear baseline. Empirical
finding motivating this survey: the deep/GNN model is competitive at short h (h1, and on
MAE), but HAR beats the deep model on QLIKE at h10/h22, and R^2 decays badly at long h.

Method note: sources below were located via web search on 2026-08-16 and are cited with
URLs. Where a claim rests on a search-engine summary rather than the fetched full text,
that is noted. Nothing here was invented; unverifiable items are flagged.

---

## Part A — Literature survey

### A.1 Multi-step / long-horizon strategy: DIRECT vs ITERATED

The definitive reference is Ghysels, Plazzi & Valkanov, "Direct Versus Iterated
Multiperiod Volatility Forecasts," *Annual Review of Financial Economics* (2019). They
compare iterated, direct, scaled-root-of-time, and MIDAS forecasts on 30 assets at
horizons of 5, 10, 22, 44 and 66 days.
https://www.annualreviews.org/content/journals/10.1146/annurev-financial-110217-022808

Key results relevant here:
- The preferred multi-step strategy is model-family-dependent. For parametric
  conditional-variance models (GARCH, MEM), ITERATED beats DIRECT. For
  realized-variance / HAR-type models, the OPPOSITE holds: DIRECT horizon-specific
  estimation wins.
- Long-horizon volatility is far more predictable than the older "scale by sqrt(h)"
  literature suggested, out to ~60 trading days, with MIDAS a convenient middle ground
  (uses daily data but produces a direct horizon-h forecast).
- Theoretical backdrop (Marcellino, Stock & Watson 2006): iterated is more efficient
  when the one-step model is well specified; direct is more robust to misspecification.
  Direct RV/HAR models sidestep misspecification of the daily dynamics.

Applied confirmation on a large cross-section: a 2025 low-volatility-investing study
(1,699 US stocks) follows Ghysels et al. — direct forecasting for HAR/MIDAS, iterated
for GARCH/MEM — and finds RV-based models with DIRECT forecasting outperform all other
individual models. https://www.sciencedirect.com/science/article/pii/S0169207025000743

Implication for this project: the project ALREADY uses the direction the literature
prefers (direct horizon-specific forecasts for an RV-type deep model). So the long-h gap
is NOT a direct-vs-iterated problem. The gap must be closed by better long-h inductive
bias, better target definition, the right loss, longer memory, and shrinkage/combination
(Parts A.2–A.5, B).

### A.2 Why HAR is strong at long horizons; how deep models absorb that bias

HAR (Corsi 2009) approximates long-memory volatility with a parsimonious cascade of
daily / weekly / monthly RV components, motivated by heterogeneous market agents at
different horizons.
https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf

Long-memory evidence and horizon dependence:
- RV exhibits long memory (persistent, slowly-decaying autocorrelation). Baillie,
  Calonaci & Cho (2019) find the fractional long-memory parameter often still adds
  forecasting value even alongside HAR structure.
  https://onlinelibrary.wiley.com/doi/abs/10.1111/jtsa.12470
- Horizon dependence of short- vs long-memory models: short-memory AR/ARMA can win at
  short horizons, while ARFIMA (explicit fractional integration) tends to dominate at
  medium-to-long horizons; HAR's cascade produces better long-horizon forecasts than
  same-order AR at the price of some short-horizon accuracy.
  https://arxiv.org/pdf/1712.08057

The practical lesson: HAR's edge at long h comes from (a) an explicit LONG (monthly)
persistence component and (b) high forecast persistence approximating long memory. A
deep model that does not encode a monthly/long component, or that is not persistent
enough, will lose exactly where the project loses (h10/h22).

### A.3 Deep learning at long horizons — does it beat HAR?

The single most relevant empirical result: Christensen, Siggaard & Veliyev, "A Machine
Learning Approach to Volatility Forecasting," *Journal of Financial Econometrics* 21(5),
2023 (arXiv:2601.13014). ML (regularization, trees, feed-forward NN) beats the HAR
lineage on Dow Jones constituents EVEN when the only predictors are daily/weekly/monthly
RV lags — and the gains are MORE pronounced at longer horizons (one-day -> one-week ->
one-month), attributed to higher model persistence better approximating long memory.
https://academic.oup.com/jfec/article-abstract/21/5/1680/6612759 ;
https://arxiv.org/pdf/2601.13014

Countervailing / cautionary evidence:
- A Dec-2025 study (S&P500/NASDAQ100/DJIA, 2000-2025, h=1/5/22, QLIKE+RMSE+MAE with
  Diebold-Mariano) reports ML models tend to outperform HAR-RV/ARIMA/GARCH across
  horizons — but flags tuning intensity and trading-cost caveats.
  https://www.mdpi.com/1911-8074/18/12/685
- Time-series foundation models (TimesFM, Chronos, Moirai, TTM, Toto...): the most
  systematic zero-shot comparison (VOLARE dataset, 50 assets, arXiv:2607.05291) finds
  that once each asset's loss is weighted equally against a well-specified Log-HAR, only
  the small TTM model beats HAR at every horizon, and only narrowly; most foundation
  models (incl. TimesFM 2.5, Chronos-Bolt, Moirai, Toto) have loss ratios > 1 (they LOSE
  to Log-HAR on the typical asset). An equal-weight average of TTM + Log-HAR enters the
  Model Confidence Set for 98-100% of assets — combination beats either alone.
  https://arxiv.org/abs/2607.05291
- Goel et al. (2025, arXiv:2505.11163) find TimesFM needs FINE-TUNING (not zero-shot) to
  compete with HAR. https://arxiv.org/pdf/2505.11163

Net reading: deep models CAN beat HAR at long daily horizons, but only when they (i) are
persistent enough / carry long-memory bias, (ii) are trained/evaluated on the right loss,
and (iii) are often COMBINED with HAR rather than replacing it. Well-specified HAR /
Log-HAR remains a hard baseline; "deep replaces HAR" is not reliably supported, "deep +
HAR combination beats HAR" is.

### A.4 Loss function: QLIKE vs MSE (directly relevant — the project loses on QLIKE)

The project loses to HAR specifically on QLIKE at h10/h22 while training on MSE. The
literature is direct on this:
- HARNet (CNN for RV, arXiv:2205.07719): QLIKE-optimized models beat baseline HAR on
  QLIKE AND on MAE/MSE; the reverse (train MSE) "usually fails" to generalize to QLIKE.
  https://arxiv.org/pdf/2205.07719
- "qlikeHAR": estimating even the linear HAR with a QLIKE loss yields large
  out-of-sample QLIKE gains vs MSE-estimated HAR (same model, different loss). This means
  part of HAR's QLIKE edge can come from loss alignment, which a deep model can also
  adopt. (Reported in HARNet and the GNNHAR line of work.)
- GNNHAR (below) finds QL-loss training gives "substantial improvements" over MSE, via
  better handling of heteroskedasticity, and QL-trained nonlinear models are more
  resilient in turbulent periods.
- Independent corroboration: Bergsli et al. (2022) and Pourrezaee & Hajizadeh (2025) —
  QLIKE-trained DL beats MSE-trained for crypto vol, especially in tail-risk events
  (summarized via search; primary PDFs not individually fetched).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10185465/

This is the single best-supported, lowest-effort lever for the exact metric where the
project is losing.

### A.5 Target definition and horizon aggregation (often overlooked)

Standard practice in the HAR/Ghysels literature evaluates the AVERAGE (or cumulative)
realized variance over the multi-period horizon [t+1, t+h], NOT the single-day variance
exactly at t+h. Corsi (2009) evaluates multistep forecasts by comparing AGGREGATED
realized and predicted volatility over the multi-period horizon.
https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf
Ghysels et al. (2019) and the low-vol study forecast the average h-day realized variance.
https://www.sciencedirect.com/science/article/pii/S0169207025000743

Why this matters for this project: forecasting a SINGLE far-future day t+h is intrinsically
noisier (it is one realization) than forecasting the h-day AVERAGE, which smooths idiosyncratic
daily noise and is more predictable — so single-day t+h targets mechanically depress long-h
R^2. Corsi/Ghysels-style targets are h-day averages. This is a target-definition effect, not a
model failure. (Flag: this project's current single-day-at-t+h target is a legitimate task, but
it is a harder task than the average-RV target most HAR long-h results are reported on; direct
numeric comparisons to those papers' R^2 are therefore not like-for-like.)

### A.6 MIDAS / mixed frequency

MIDAS regressions use daily data to produce a DIRECT multiperiod forecast — a middle
ground between direct and iterated, and a strong long-h performer in Ghysels et al.
https://www.annualreviews.org/content/journals/10.1146/annurev-financial-110217-022808
For this project (daily-only inputs, no intraday), classic across-frequency MIDAS is less
applicable, but the MIDAS idea maps to a learnable, parsimonious weighting of many past
daily lags into the horizon-h forecast (a data-driven HAR/MIDAS lag kernel) — see B.3.

### A.7 Graph / GNN models at long horizons (honest evidence)

GNNHAR — Zhang, Zohren et al., "Forecasting Realized Volatility with Spillover Effects:
Perspectives from Graph Neural Networks," *International Journal of Forecasting* 41(1),
2025. Code: https://github.com/chaozhang-ox/GNNHAR ;
paper: https://www.sciencedirect.com/science/article/abs/pii/S0169207024000967 ;
PDF: https://web.media.mit.edu/~xdong/paper/ijf25.pdf ;
listing: https://ideas.repec.org/a/eee/intfor/v41y2025i1p377-397.html

Findings (verified via fetch of the listing + search summaries):
- Nonlinear spillover modeling improves accuracy PARTICULARLY at short-term horizons "up
  to one week." The graph benefit is concentrated at short h — it does not clearly
  persist to longer horizons.
- Multi-hop neighbor information does NOT give a clear advantage; the benefit comes from
  the nonlinearity, not deeper graph reach.
- QL-loss training substantially beats MSE; QL-trained nonlinear models are more resilient
  in turbulent periods.
- Evaluation uses MSE, QLIKE and the Model Confidence Set (Hansen, Lunde & Nason 2011).

This is directly consistent with THIS project's own pattern: graph/deep help at h1 (short)
and fade by h10/h22. Honest expectation: the graph is unlikely to be the thing that fixes
long-h; do not over-invest in multi-hop/edge engineering for h22.

Related: a dynamic-GNN volatility paper (arXiv:2410.16858) and graph-signal-processing RV
work (arXiv:2410.22706) exist but do not overturn the "spillover helps short-h most"
message. https://arxiv.org/pdf/2410.16858 ; https://arxiv.org/pdf/2410.22706

### A.8 Ensembling / shrinkage / Model Confidence Set

- Combining forecasts based on Model Confidence Sets can achieve superior performance;
  MCS (Hansen, Lunde & Nason 2011) is the standard test for whether a combination beats
  its components. https://www.sciencedirect.com/science/article/abs/pii/S0169207016300747
- For HAR-family RV, shrinkage (elastic net / lasso over HAR variants + predictors) can
  beat BOTH individual HAR models AND naive combinations under MCS (oil vol; robust).
  https://www.sciencedirect.com/science/article/abs/pii/S0140988319300258
- Combining many realized measures via OLS beats HAR across all horizons; ensembles give
  top performance across horizons though not necessarily best at every single horizon.
  https://fmai.memberclicks.net/assets/docs/Derivatives2021/VolatilityML_Tang.pdf
- Foundation-model study (A.3): equal-weight (deep + Log-HAR) enters the MCS more often
  than either component alone. https://arxiv.org/abs/2607.05291

Takeaway: shrinking the deep forecast toward HAR (or combining them) is one of the most
reliably reported ways to get long-h robustness without abandoning the deep model.

### A.9 Direct multi-horizon / seq2seq heads

Temporal Fusion Transformer (Lim et al., 2021, *IJF*): LSTM encoder + multi-head attention
+ variable selection, producing DIRECT multi-horizon (and multi-quantile) outputs; it was
benchmarked on a volatility dataset and beat Seq2Seq/MQRNN/DeepAR.
https://www.sciencedirect.com/science/article/pii/S0169207021000637 ;
https://ar5iv.labs.arxiv.org/html/1912.09363
Relevance: supports a single model with a shared encoder and a multi-output head emitting
{h1,h5,h10,h22} jointly (multi-task), rather than four independent models — this shares
statistical strength and is the standard "direct multi-horizon" design.

### A.10 Vietnamese practice

Vietnam volatility/forecasting work is dominated by GARCH-family and price/return
prediction, with limited multi-horizon RV/HAR:
- HSX volatility with GARCH/EGARCH/TGARCH, VN-Index 2001-2019 (GARCH(1,1)/EGARCH(1,1)
  preferred). https://koreascience.kr/article/JAKO201915658234490.page
- Bayesian GARCH(1,1) on the 30 VN30 constituents, combined via a beta-transformed linear
  mixture, ~61% accuracy predicting VN30 daily price 22 trading days ahead (a rare
  explicitly-22-day VN result). https://crimsonpublishers.com/siam/fulltext/SIAM.000598.php
- LSTM + technical indicators on VN-Index / VN30 (Nature HSSC 2024).
  https://www.nature.com/articles/s41599-024-02807-x
- LSTM + Ichimoku strategy across VN30 constituents (2012-2020).
  https://www.researchgate.net/publication/362772148_An_Empirical_Examination_on_Forecasting_VN30_Short-Term_Uptrend_Stocks_Using_LSTM_along_with_the_Ichimoku_Cloud_Trading_Strategy
- Ensemble-LSTM interval-valued prediction for VN indicators (Computational Economics 2025).
  https://link.springer.com/article/10.1007/s10614-025-10924-1
- Regime-switching / MS-GARCH / MSM on VN-Index (documents long-memory volatility,
  clustering, leverage) — motivates a long-memory + regime component.

Gap confirmed by search: no located VN study applies a HAR/GNN-HAR realized-volatility
framework with a leave-one-out ablation across h={1,5,10,22} on VN30. This project is
positioned in a genuine gap; the closest multi-horizon VN precedent is the Bayesian-GARCH
22-day VN30 study.

---

## Part B — Ranked shortlist of concrete techniques for THIS model's h10/h22

Ranked by expected long-h payoff / effort. Each maps onto the existing
LSTM + GAT + news + HAR setup.

### B1. Train (and select) with QLIKE loss, not MSE  [HIGHEST PRIORITY, LOW effort]
- Idea: replace/augment the MSE training objective with QLIKE (Q = RV_true/RV_pred -
  log(RV_true/RV_pred) - 1 on the variance scale, respecting the positivity floor already
  used in eval). Keep MSE as an optional auxiliary term if needed for stability.
- Why long-h: the project loses to HAR specifically on QLIKE at h10/h22. QLIKE-trained
  networks improve the QLIKE metric substantially and often MAE/MSE too (HARNet); MSE
  training "usually fails" on QLIKE; GNNHAR reports the same QL-over-MSE gain from better
  heteroskedasticity handling. This directly attacks the exact losing metric.
- Fit: the model outputs a variance already; train on QLIKE on the ORIGINAL (inverse-
  transformed, positivity-floored) scale, with the SAME floor as HAR to keep the DM
  comparison fair (matches the project's known "H2: identical positivity floor" rule).
- Caveat: QLIKE needs strictly positive predictions — enforce a softplus/positivity floor
  identical to HAR's before the loss.

### B2. Hybrid: predict HAR + neural RESIDUAL (shrinkage toward HAR)  [HIGH, LOW-MED]
- Idea: final_pred_h = HAR_h(x) + g * NN_residual_h(x), where HAR_h is the (frozen or
  jointly-fit) linear HAR forecast and the network learns only the residual; g is a
  learnable/regularized gate (start small so the model defaults to HAR). Equivalent to
  strong shrinkage-to-HAR that increases with h.
- Why long-h: guarantees the model is never worse than HAR by construction at the limit
  g->0, and the reliably-reported winner in the foundation-model and combination studies
  is deep+HAR combination / shrinkage, not deep alone (A.3, A.8). Removes the failure mode
  where the deep model underperforms the linear baseline at long h.
- Fit: HAR is already implemented as the baseline; wrap it as a residual base learner
  inside the current head. Cheap and low-risk. Report g by horizon (expect g to shrink as
  h grows).

### B3. Add an explicit HAR-style LONG (monthly/quarterly) component + longer lookback  [HIGH, MED]
- Idea: (a) ensure node/temporal features include HAR daily(1)/weekly(5)/monthly(22)
  aggregates explicitly (not only raw lags), and add a longer component (e.g. 66-day
  "quarterly") for h22; (b) increase the LSTM lookback for longer horizons (horizon-scaled
  context), or use a MIDAS-style learnable decay kernel over many daily lags.
- Why long-h: HAR's long-h edge is its explicit monthly component and high persistence;
  Christensen et al. attribute ML long-h gains to persistence approximating long memory;
  ARFIMA/long-memory dominates at longer h. Giving the network the same low-frequency
  inductive bias + more context is what lets deep models match HAR at long h.
- Fit: augment the existing per-ticker feature set with HAR aggregates and a 66-day term;
  optionally horizon-condition the lookback window length.

### B4. Ensemble / MCS-selected combination of {deep, HAR, (Log-HAR)}  [HIGH robustness, LOW]
- Idea: report an equal-weight or lasso/elastic-net-weighted average of the deep model and
  HAR (and optionally Log-HAR / ARFIMA), then use the Model Confidence Set (Hansen-Lunde-
  Nason) to select the surviving set per horizon.
- Why long-h: equal-weight deep+HAR entered the MCS more often than either alone across
  50 assets (A.3); shrinkage combinations beat both individual HAR and naive combos under
  MCS (A.8). Cheap insurance that the reported long-h number is at least HAR-competitive.
- Fit: the project already computes HAR and DM tests; add MCS (reference implementation in
  the GNNHAR repo, MCS.py) and an averaging/lasso combiner as a post-hoc layer.

### B5. Log / variance-stabilizing transform for the target  [MED, LOW]
- Idea: model log-variance (Log-HAR analogue) internally, then exponentiate with the
  bias correction, instead of modeling raw variance.
- Why long-h: Log-HAR is the strong benchmark in the foundation-model study; log-scale
  tames the heavy right tail of variance, stabilizes long-h training, and aligns with
  QLIKE's multiplicative error structure. Complements B1.
- Fit: change target transform in the dataset + inverse transform in eval; keep the
  positivity floor consistent for QLIKE/DM.

### B6. Multi-horizon (direct multi-output) head instead of 4 separate models  [MED, MED]
- Idea: one shared encoder, a single head emitting {h1,h5,h10,h22} jointly (multi-task),
  optionally TFT-style or with per-horizon output layers; horizon can be an input feature.
- Why long-h: shares statistical strength across horizons (the well-estimated short-h
  signal regularizes the noisy long-h heads); this is the standard direct-multi-horizon
  design (TFT). It also enforces cross-horizon monotonic-persistence structure implicitly.
- Fit: refactor the four direct heads into one multi-output head; reuse the existing
  encoder + GAT + news branch.

### B7. Report/compare the h-day AVERAGE-RV target (target-definition check)  [MED insight, LOW]
- Idea: alongside single-day t+h, also produce and evaluate the average RV over [t+1,t+h]
  (the standard HAR/Ghysels target), for both the deep model and HAR.
- Why long-h: single-day-far-ahead targets are intrinsically noisier; the average target
  is more predictable and is what most long-h HAR results report. This (a) likely raises
  long-h R^2 for BOTH models and (b) tells you how much of the "bad long-h R^2" is task
  difficulty vs model deficiency. Important for honest paper framing, not just accuracy.
- Fit: add an average-RV target variant to the dataset; run the existing pipeline on it.
- Caveat: this changes the task; report it as a complementary result, not a silent swap.

### B8. Long-memory / fractional component (ARFIMA-style)  [MED, MED-HIGH]
- Idea: add a fractional-integration or explicit long-memory feature/branch (e.g. an
  ARFIMA forecast as an input feature, or a fractional-difference transform of RV).
- Why long-h: ARFIMA dominates at medium-to-long horizons; the long-memory parameter often
  adds value beyond HAR (Baillie et al.). Targeted at exactly the h10/h22 regime.
- Fit: compute an ARFIMA/fractional feature offline and feed it as an extra input; higher
  effort and less certain payoff than B1-B4, hence lower rank.

### B9. Graph work is LOW priority for long h  [LOW for h22]
- Honest evidence (GNNHAR, and this project's own pattern): spillover/graph helps mainly at
  short horizons (<= 1 week); multi-hop does not clearly help. Horizon-conditioned edges or
  a temporal graph are research-interesting but unlikely to be what closes the h22 gap.
  Prefer B1-B6 first; keep the graph as-is for long h.

---

## Part C — Realistic expectations

- Long-horizon DAILY volatility is intrinsically hard. Even in top venues, well-specified
  HAR / Log-HAR is a strong baseline that most deep and foundation models fail to beat
  at long h on a per-asset, equal-weighted basis (arXiv:2607.05291). Expect to MATCH or
  MODESTLY beat HAR at h10/h22, not to dominate it.
- Low long-h R^2 is normal. R^2 mechanically decays with h, and is further depressed by a
  single-day-t+h target (vs h-day average, B7). A low but positive long-h R^2 that is
  HAR-competitive on QLIKE is a legitimate, publishable result — the parsimony-null
  framing (nothing cleanly beats HAR) is itself consistent with the literature.
- The most credible "deep beats HAR at long h" results (Christensen et al. 2023) rely on
  high persistence + long-memory approximation, and even they find the best model is
  horizon-dependent. The realistic goal is: deep model + HAR combination (B2/B4) that is
  in the Model Confidence Set at every horizon, and a deep model trained on QLIKE (B1)
  that no longer loses the QLIKE metric it is evaluated on.
- Whether deep can beat HAR at long daily horizons: the literature says CONDITIONALLY YES
  (with long-memory bias, right loss, and often combination), but NOT reliably as a
  drop-in replacement. deep+HAR shrinkage/combination is the safer, better-supported route.

---

## Sources (verified URLs)

- Ghysels, Plazzi & Valkanov (2019), Direct vs Iterated Multiperiod Volatility Forecasts —
  https://www.annualreviews.org/content/journals/10.1146/annurev-financial-110217-022808
- Ruiz et al. (2023), Direct vs iterated multiperiod VaR —
  https://onlinelibrary.wiley.com/doi/10.1111/joes.12522
- Volatility forecasting for low-volatility investing (2025, direct HAR/MIDAS practice) —
  https://www.sciencedirect.com/science/article/pii/S0169207025000743
- Corsi (2009), A Simple Approximate Long-Memory Model of Realized Volatility —
  https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf
- Baillie, Calonaci & Cho (2019), Long Memory, RV and HAR —
  https://onlinelibrary.wiley.com/doi/abs/10.1111/jtsa.12470
- On Long Memory Origins and Forecast Horizons (arXiv:1712.08057) —
  https://arxiv.org/pdf/1712.08057
- Christensen, Siggaard & Veliyev (2023), A Machine Learning Approach to Volatility
  Forecasting, JFE — https://academic.oup.com/jfec/article-abstract/21/5/1680/6612759 ;
  arXiv:2601.13014 — https://arxiv.org/pdf/2601.13014
- Deep Learning & Transformer Architectures for Volatility Forecasting (MDPI 2025, h=1/5/22)
  — https://www.mdpi.com/1911-8074/18/12/685
- Forecasting RV with Time Series Foundation Models vs Econometric Benchmarks
  (arXiv:2607.05291) — https://arxiv.org/abs/2607.05291
- Goel et al. (2025), Foundation Time-Series AI Model for RV Forecasting (arXiv:2505.11163)
  — https://arxiv.org/pdf/2505.11163
- HARNet: CNN for Realized Volatility Forecasting (arXiv:2205.07719) —
  https://arxiv.org/pdf/2205.07719
- Do ANNs provide improved volatility forecasts: Asian markets (QLIKE vs MSE) —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10185465/
- Zhang et al. (2025), GNNHAR: Forecasting RV with Spillover Effects, IJF —
  https://www.sciencedirect.com/science/article/abs/pii/S0169207024000967 ;
  PDF https://web.media.mit.edu/~xdong/paper/ijf25.pdf ;
  listing https://ideas.repec.org/a/eee/intfor/v41y2025i1p377-397.html ;
  code https://github.com/chaozhang-ox/GNNHAR
- Dynamic GNN for volatility (arXiv:2410.16858) — https://arxiv.org/pdf/2410.16858
- Graph Signal Processing for global RV (arXiv:2410.22706) — https://arxiv.org/pdf/2410.22706
- Lim et al. (2021), Temporal Fusion Transformers, IJF —
  https://www.sciencedirect.com/science/article/pii/S0169207021000637 ;
  arXiv https://ar5iv.labs.arxiv.org/html/1912.09363
- Model Confidence Sets and forecast combination (IJF) —
  https://www.sciencedirect.com/science/article/abs/pii/S0169207016300747
- Forecasting oil price volatility: combination vs shrinkage (MCS) —
  https://www.sciencedirect.com/science/article/abs/pii/S0140988319300258
- Automatic system using many realized measures (ensemble beats HAR across horizons) —
  https://fmai.memberclicks.net/assets/docs/Derivatives2021/VolatilityML_Tang.pdf
- Vietnam GARCH/EGARCH/TGARCH on HSX —
  https://koreascience.kr/article/JAKO201915658234490.page
- Bayesian GARCH VN30, 22-day-ahead —
  https://crimsonpublishers.com/siam/fulltext/SIAM.000598.php
- LSTM + technical indicators VN-Index/VN30 (Nature HSSC 2024) —
  https://www.nature.com/articles/s41599-024-02807-x
- LSTM + Ichimoku VN30 —
  https://www.researchgate.net/publication/362772148_An_Empirical_Examination_on_Forecasting_VN30_Short-Term_Uptrend_Stocks_Using_LSTM_along_with_the_Ichimoku_Cloud_Trading_Strategy
- Ensemble-LSTM interval-valued VN prediction (Comp. Economics 2025) —
  https://link.springer.com/article/10.1007/s10614-025-10924-1

### Verification notes
- Fetched full listing text: GNNHAR IdeasRepec page (A.7 horizon/loss findings confirmed).
- All other claims are from web-search result summaries with the URLs above; direct-quote
  numerics (e.g. TFT P50 losses, foundation-model loss ratios, Christensen horizon gains)
  were reported by the search engine from those sources but the individual full PDFs were
  not each fetched line-by-line. Treat specific numbers as "as reported by the source
  summary"; the qualitative conclusions are consistent across multiple independent sources.
- Some arXiv IDs (2601.13014, 2607.05291, 2505.11163) are 2026/2025-dated preprints
  surfaced by search; IDs are recorded as returned and should be re-checked against
  arXiv before formal citation in a paper.
