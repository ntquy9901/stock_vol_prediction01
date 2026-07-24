# Evaluating News Sentiment & Event Features Before Training HAR + GNN

## Objective

Before adding any news-derived feature into HAR, GNN, or hybrid
forecasting models, verify that it contains predictive information
rather than noise.

------------------------------------------------------------------------

# Level 1. Statistical Significance

For every candidate feature:

-   Positive score
-   Negative score
-   Fear score
-   Optimism score
-   Uncertainty score
-   Event type
-   Event severity
-   Event confidence

Evaluate:

-   Pearson Correlation
-   Spearman Correlation
-   Kendall Tau
-   Mutual Information (recommended)
-   Distance Correlation (for nonlinear relationships)

### Interpretation

-   MI ≈ 0 → Feature is likely useless.
-   Pearson ≈ 0 but MI \> 0 → Possible nonlinear predictive
    relationship.

**Recommendation:** Mutual Information is generally more informative
than Pearson for financial forecasting.

------------------------------------------------------------------------

# Level 2. Event Study

Treat each event independently.

Examples:

-   CEO resignation
-   Earnings release
-   Dividend announcement
-   Stock buyback
-   Fraud
-   M&A
-   Government policy
-   Interest rate decision
-   CPI / Inflation

Analyze:

-   T-5 ... T0 ... T+10
-   Average volatility
-   Average return
-   Abnormal return
-   Cumulative Abnormal Return (CAR)

Goal:

Determine whether each event consistently changes future volatility.

------------------------------------------------------------------------

# Level 3. Predictive Power

Before deep learning, train simple machine learning models.

Recommended:

-   XGBoost
-   LightGBM
-   CatBoost
-   Random Forest

Targets:

-   Volatility T+1
-   Volatility T+5
-   Volatility T+10

Compare:

Baseline (HAR only)

vs.

HAR + Sentiment

vs.

HAR + Event

A meaningful RMSE reduction indicates the feature adds predictive value.

------------------------------------------------------------------------

# Level 4. SHAP Analysis

Train a tree-based model and compute SHAP values.

Questions answered:

-   Which features contribute most?
-   Are effects positive or negative?
-   Are contributions stable?

Example:

Fear Score ........ 12%

Dividend .......... 8%

Fraud ............. 6%

Positive Score .... 0.5%

Features with negligible contribution should be removed.

------------------------------------------------------------------------

# Level 5. Ablation Study

Evaluate each feature group independently.

Example:

  Model             RMSE
  ----------------- -------
  HAR               0.195
  HAR + Sentiment   0.186
  HAR + Event       0.178
  HAR + Embedding   0.164
  HAR + GNN         0.161
  HAR + All         0.159

Purpose:

Measure the true contribution of every feature family.

------------------------------------------------------------------------

# Information Coefficient (IC)

Widely used in quantitative finance.

Measure:

Feature

↓

Future Return or Future Volatility

Compute IC daily and monitor stability.

Example:

Fear Score

IC = 0.28

Stable IC over long periods suggests a strong predictive feature.

------------------------------------------------------------------------

# Lag Analysis

News effects are rarely immediate.

Evaluate lags:

-   Lag 0
-   Lag 1
-   Lag 2
-   ...
-   Lag 20

Example:

Fear score may peak at Lag = 3.

Use the optimal lag instead of same-day sentiment.

------------------------------------------------------------------------

# Granger Causality Test

Question:

Does sentiment provide predictive information beyond HAR?

Interpretation:

p-value \< 0.05

→ Sentiment adds incremental predictive power.

Otherwise, sentiment may simply reflect historical price movements.

------------------------------------------------------------------------

# Feature Stability

A good feature should remain useful across multiple years.

Evaluate:

-   Rolling Mutual Information
-   Rolling Information Coefficient
-   Rolling SHAP

Avoid features that only perform well during one market regime.

------------------------------------------------------------------------

# Recommended Evaluation Pipeline

1.  Mutual Information screening.
2.  Cross-correlation / Lag analysis.
3.  Event Study.
4.  Granger Causality Test.
5.  LightGBM/XGBoost + SHAP.
6.  Ablation Study.

Only features that pass these evaluations should be included in HAR,
GNN, or hybrid volatility forecasting models.

------------------------------------------------------------------------

# Recommended Tool Stack

## Statistics

-   scipy
-   statsmodels
-   pingouin
-   dcor
-   scikit-learn (Mutual Information)

## Machine Learning

-   LightGBM
-   XGBoost
-   CatBoost
-   Random Forest

## Explainability

-   SHAP

## Visualization

-   matplotlib
-   plotly

## Time Series

-   statsmodels
-   arch
-   darts

## Suggested Workflow

News → Feature Extraction → Statistical Screening → Event Study → Lag
Analysis → Granger Test → ML Feature Importance (SHAP) → Ablation Study
→ Final HAR + GNN Training
