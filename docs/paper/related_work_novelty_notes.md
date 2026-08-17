# Related Work & novelty positioning — is the beat-HAR (combination) architecture new?

Literature check (WebSearch, 2026-08-17) to position the forecast-combination result honestly and
supply verified citations for the paper's Related Work. **The combination architecture is NOT
methodologically novel** — every component has established prior work. The contribution is empirical
(VN emerging market, Parkinson variance from daily data, leakage-safe multi-seed evaluation) and the
honest finding that the deep/GNN value materialises only via combination with HAR at short horizons.

## Verified citations (use these)

**Forecast combination (the core technique — old and standard):**
- **Bates, J. M. & Granger, C. W. J. (1969).** "The Combination of Forecasts." *Operational Research
  Quarterly* 20(4), 451–468. DOI 10.1057/jors.1969.103. Seminal: a composite of two forecasts yields
  lower MSE than either; weights from past errors.
- **Timmermann, A. (2006).** "Forecast Combinations." In *Handbook of Economic Forecasting*, Vol. 1,
  Ch. 4, pp. 135–196 (Elliott, Granger, Timmermann eds.), Elsevier. The authoritative survey.
- **Stock, J. H. & Watson, M. W. (2004).** "Combination Forecasts of Output Growth in a Seven-Country
  Data Set." *Journal of Forecasting* 23, 405–430. Origin of the **"forecast combination puzzle"** —
  simple **equal weights** are hard to beat with estimated "optimal" weights (justifies our fixed 0.5).
- **Clemen, R. T. (1989).** review of 200+ combination studies; simple average as the benchmark.
- **Smith, J. & Wallis, K. F. (2009).** bias–variance explanation of why equal weights win.

**HAR itself as a combination (directly on point):**
- **Clements, A. & Vasnev, A. L. (2024).** "Forecast combination puzzle in the HAR model." *Journal
  of Forecasting* 43(1), 118–137. DOI 10.1002/for.3029. **Views the HAR model itself as a forecast
  combination** of three predictors (daily/weekly/monthly). Shows "combining HAR-family forecasts" is
  already a studied idea — so combining HAR with another model is a natural, non-novel extension.

**HAR baseline:**
- **Corsi, F. (2009).** "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of
  Financial Econometrics* 7(2), 174–196.

**GNN + HAR for volatility (the project's direct predecessor — now PUBLISHED):**
- **Zhang, C., Pu, X., Cucuringu, M. & Dong, X. (2025).** "Forecasting realized volatility with
  spillover effects: Perspectives from graph neural networks." *International Journal of Forecasting*
  41(1), 377–397. DOI 10.1016/j.ijforecast.2024.... (arXiv:2308.01419 working paper, 2023). GNNHAR /
  GHAR. Their published conclusions **independently match ours on VN30**: (i) multi-hop spillover
  alone gives no clear advantage; (ii) nonlinear spillover helps mainly short-term (≤ 1 week); (iii)
  QLIKE-loss training substantially beats MSE. Code: github.com/chaozhang-ox/GNNHAR.
- Earlier: Zhang, Pu, Cucuringu, Dong (2022), "Multivariate Realized Volatility Forecasting with Graph
  Neural Network," *ACM ICAIF*.

**ML/NN vs & combined with HAR (context):**
- **Christensen, K., Siggaard, M. & Veliyev, B. (2023).** "A Machine Learning Approach to Volatility
  Forecasting." ML (RF, NN) beats the HAR lineage on DJIA; **gains larger at LONGER horizons** — the
  OPPOSITE of our VN result (deep helps at SHORT h), worth noting as an emerging-market / daily-
  Parkinson vs intraday-RV divergence.
- **Hu, N., Yin, X. & Yao, Y. (2025).** "A novel HAR-type realized volatility forecasting model using
  graph neural network." *International Review of Financial Analysis* 98, 103881. CNN-HAR-KS, China
  market (a differently-titled CNN approach, not a GNN spillover model).

## Honest novelty positioning for the paper
1. **Do NOT claim the combination architecture (HAR ⊕ deep) as novel.** Cite Bates–Granger (1969),
   Timmermann (2006), Stock–Watson (2004) for combination; Clements–Vasnev (2024) for "HAR-as-
   combination"; Zhang et al. (2025) for GNN+HAR.
2. **Claim the empirical/contextual contribution:** first (to our knowledge) leakage-safe, multi-seed,
   Diebold–Mariano evaluation of a directed-spillover graph-attention + PhoBERT-news model against a
   strong HAR baseline on **Vietnamese (VN30) daily Parkinson-variance** data, with the finding that
   graph/news add no reliable OOS value and the deep model beats HAR only in combination at short
   horizons.
3. **Strength — independent corroboration:** our VN30 results reproduce Zhang et al. (2025)'s
   published conclusions (short-horizon nonlinear spillover; QLIKE ≫ MSE; multi-hop no gain) on a new
   market, which is itself a useful robustness datapoint.

*Citations were retrieved and cross-checked via web search; verify page numbers / DOIs against the
publisher record before final submission.*
