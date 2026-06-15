# Requirements Document: Stock Volatility Prediction for VN30

**Project:** Stock Volatility Prediction for VN30 Stocks  
**Date:** 2026-06-15  
**Author:** QUY  
**Data:** Daily OHLCV format (30 stocks, 2006-2026)  

---

## Executive Summary

Build a robust stock volatility prediction system for VN30 stocks using daily OHLCV data, implementing HAR (Heterogeneous Autoregressive) methodology adapted for daily frequency, with comparison against modern ML approaches including LSTM, GNN, and TimesFM foundation models.

---

## 1. Research Context

### 1.1 Paper Analysis: HAR-X Methodology

**Source:** "HAR-X: A Robust and Flexible Multi-Horizon Time Series Forecasting Model" (Ma & Zhang)

**Key Findings:**
- HAR-X extends traditional HAR models with flexible multi-horizon forecasting
- Core methodology uses three volatility horizons: daily (lag 1), weekly (avg 5 days), monthly (avg 22 days)
- Requires **intraday high-frequency data** to calculate Realized Volatility (RV)
- Demonstrates superior performance over traditional HAR variants

**Critical Constraint:**
- Paper's HAR-X methodology **cannot be directly implemented** with daily OHLCV data only
- Requires adaptation using daily volatility proxies

### 1.2 Data Constraints

**Available Data:**
- 30 VN30 stocks in OHLCV format
- Date range: 2006-2026 (varies by stock)
- Daily frequency (no intraday data)
- Located: `data/raw/prices/`

**Implication:**
- Cannot calculate traditional Realized Volatility (RV)
- Must use range-based or return-based volatility estimators

---

## 2. Functional Requirements

### 2.1 Core Prediction Requirements

**FR-1: Multi-Horizon Volatility Forecasting**
- **Horizons:** 1-day, 5-day (week), 10-day (2 weeks), **22-day (month)** ahead predictions ✅
- **Output:** Volatility forecasts for all 30 VN30 stocks
- **Frequency:** Daily predictions
- **Format:** Numerical volatility values with confidence intervals

**✅ 22-DAY CONSISTENCY CONFIRMED:**
- HAR monthly feature: 22-day lookback window (backward)
- Monthly forecast horizon: 22-day forward prediction
- **Rationale:** Academic standard (~22 trading days/month), consistency across feature engineering & forecasting

**FR-2: Baseline Model Implementation**
- **Primary Baseline:** HAR-R (Range-based HAR) adapted for daily data
- **Secondary Baselines:** 
  - HAR-X with daily volatility proxies
  - Traditional statistical models (GARCH, EWMA)
- **Evaluation:** RMSE, MAE, R², Directional Accuracy

**FR-3: Advanced Model Comparison**
- **Neural Approaches:**
  - HAR-R + LSTM (non-linear temporal patterns)
  - HAR-R + GNN (stock relationship modeling)
  - Pure LSTM (without HAR features)
- **Foundation Models:**
  - TimesFM (Google's time series foundation model)
  - Chronos (Amazon's time series foundation model)
- **Ensemble Methods:** Combination of multiple approaches

**FR-4: Unified Model Architecture**
- **Single Model for All Stocks:** Train one model applicable to all 30 VN30 stocks
- **Input Features:** Stock identifier + temporal features + HAR features
- **Output:** Multi-horizon predictions (1, 5, 10, 20 days)
- **Scalability:** Architecture must support adding new stocks

### 2.2 Data Processing Requirements

**FR-5: Volatility Calculation for Daily Data**
- **Primary Method:** Parkinson volatility estimator
  ```
  Parkinson Volatility = (log(High/Low)²) / (4*log(2))
  ```

**HAR Feature Configuration:**
- **Daily Horizon:** 1 trading day (lag 1)
- **Weekly Horizon:** 5 trading days (rolling average)
- **Monthly Horizon:** 22 trading days (rolling average) ✅ **CONFIRMED**

**Rationale for 22-day monthly horizon:**
- Standard industry practice: ~22 trading days per month
- Consistent with original HAR methodology (Corsi, 2009)
- Balances recent information capture with statistical stability
- Widely adopted in financial volatility forecasting literature

**Alternative Methods:**
  - Garman-Klass estimator
  - Squared returns (RV proxy)
  - Absolute returns (robust alternative)
- **Validation:** Compare estimators and select best performing

**FR-6: Feature Engineering**
- **HAR Features (CONFIRMED CONFIGURATION):**
  ```python
  HAR_FEATURES = {
      'daily': {'window': 1, 'description': 'Daily volatility - lag 1 day'},
      'weekly': {'window': 5, 'description': 'Weekly volatility - 5 trading days'},
      'monthly': {'window': 22, 'description': 'Monthly volatility - 22 trading days'} ✅
  }
  ```
- **Calculation Method:** Parkinson volatility estimator
  ```python
  def calculate_parkinson_volatility(high, low):
      return (np.log(high/low)**2) / (4*np.log(2))
  ```
- **Additional Features:**
  - Lagged returns (1, 5, 10, 20 days)
  - Volume indicators
  - Technical indicators (RSI, MACD, etc.)
  - Temporal features (day of week, month, quarter)

**FR-6.1: Feature Windows Configuration**
- **Input Data:** Daily OHLCV for 30 VN30 stocks
- **Minimum Data Requirement:** 22+ days of history for initial HAR features
- **Rolling Windows:** All features use rolling calculations with specified windows
- **Missing Data Handling:** Forward-fill for gaps < 5 days, interpolation for larger gaps

**FR-7: Data Quality Assurance**
- **Missing Data Handling:** Forward-fill, back-fill, interpolation
- **Outlier Detection:** Statistical methods for anomalous price movements
- **Data Validation:** Ensure OHLCV consistency (High ≥ Close ≥ Low, etc.)
- **Data Versioning:** Track data preprocessing steps and versions

### 2.3 Model Training Requirements

**FR-8: Training Pipeline**
- **Temporal Split:** Train/validation/test with chronological ordering
- **Cross-Validation:** Rolling window validation for time series
- **Hyperparameter Tuning:** Grid search with time series cross-validation
- **Model Versioning:** Track experiments with MLflow or similar

**FR-9: Evaluation Framework**
- **Metrics:** RMSE, MAE, R², Directional Accuracy, MAPE
- **Benchmark Comparison:** Compare against naive forecasts (random walk)
- **Statistical Testing:** Diebold-Mariano test for forecast significance
- **Visualization:** Actual vs predicted plots, error distributions

**FR-10: Production Deployment**
- **Model Serving:** REST API for predictions
- **Batch Inference:** Daily batch predictions for all stocks
- **Real-time Updates:** Incremental model updates with new data
- **Monitoring:** Model performance tracking and degradation alerts

---

## 3. Non-Functional Requirements

### 3.1 Code Quality Standards

**NFR-1: Clean Code Compliance**
- Follow ML/DS Common Rules (`D:\bmad-projects\ml-ds-common-rules`)
- Descriptive variable names (no abbreviations like `lr`, `clf`)
- Small, focused functions (< 30 lines)
- Clear documentation with docstrings
- Type hints for function signatures

**NFR-2: Testing Requirements**
- **Unit Tests:** 85%+ coverage for data processing, feature engineering
- **Integration Tests:** End-to-end pipeline testing
- **Model Tests:** Validate model behavior on synthetic data
- **Performance Tests:** Inference latency < 100ms per stock
- **Testing Framework:** pytest with fixtures for test data

**NFR-3: Research Best Practices**
- **Reproducibility:** Random seeds for all stochastic operations
- **Experiment Tracking:** MLflow for hyperparameters, metrics, artifacts
- **Code Documentation:** README with setup, usage, and examples
- **Git Hygiene:** Meaningful commits, feature branches, PR reviews

### 3.2 Architecture Requirements

**NFR-4: Scalability**
- **Data Processing:** Handle 30 stocks × 20 years of daily data efficiently
- **Model Training:** Support batch training and incremental updates
- **Inference:** Parallel predictions for multiple stocks
- **Storage:** Efficient data formats (Parquet, HDF5)

**NFR-5: Maintainability**
- **Modular Design:** Separate data, model, evaluation modules
- **Configuration:** External configuration for hyperparameters
- **Logging:** Comprehensive logging for debugging and monitoring
- **Error Handling:** Graceful failure handling for edge cases

**NFR-6: Performance**
- **Training Time:** Complete training < 1 hour for baseline models
- **Inference Latency:** < 100ms per stock for predictions
- **Memory Usage:** < 8GB RAM for standard operations
- **Data Loading:** Efficient lazy loading and caching

### 3.3 Security and Compliance

**NFR-7: Data Security**
- **Access Control:** Restricted access to raw data and models
- **Encryption:** Encrypt sensitive model parameters
- **Audit Logging:** Track all data access and model updates

**NFR-8: Financial Compliance**
- **Backtesting Integrity:** Prevent look-ahead bias in testing
- **Model Validation:** Independent validation of model performance
- **Documentation:** Maintain model documentation for regulatory review

---

## 4. Technical Architecture

### 4.1 Recommended Approach

**Phase 1: Baseline Implementation (Weeks 1-2)**
```python
# HAR-R Implementation Architecture
def calculate_parkinson_volatility(df):
    """Calculate Parkinson volatility from OHLC data."""
    return (np.log(df['High']/df['Low'])**2) / (4*np.log(2))

def create_har_features(volatility_series):
    """Create HAR features (daily, weekly, monthly)."""
    return pd.DataFrame({
        'daily': volatility_series.rolling(1).mean(),
        'weekly': volatility_series.rolling(5).mean(),
        'monthly': volatility_series.rolling(22).mean()
    })

def train_har_baseline(X_train, y_train, horizon):
    """Train HAR baseline model for specific horizon."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
```

**Phase 2: Advanced Models (Weeks 3-6)**
```python
# HAR-R + LSTM Architecture
class HAR_LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)  # 4 horizons
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# HAR-R + GNN Architecture
class HAR_GNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_stocks=30):
        super().__init__()
        self.graph_conv = GraphConv(feature_dim, hidden_dim)
        self.temporal = LSTM(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 4)
```

**Phase 3: Foundation Models (Weeks 7-8)**
```python
# TimesFM Integration
def train_timesfm_model(train_data):
    """Train TimesFM foundation model for volatility prediction."""
    model = TimesFM.from_pretrained("timesfm-200m")
    # Fine-tune for volatility prediction task
    model.fine_tune(train_data, task="volatility_forecasting")
    return model
```

### 4.2 Technology Stack

**Core Technologies:**
- **Language:** Python 3.11+
- **Data Processing:** pandas, numpy, scikit-learn
- **Deep Learning:** PyTorch, TensorFlow
- **Time Series:** statsmodels, pmdarima
- **GNN:** PyTorch Geometric
- **Foundation Models:** TimesFM, Chronos

**Supporting Tools:**
- **Experiment Tracking:** MLflow
- **Testing:** pytest, pytest-cov
- **Documentation:** Sphinx, mkdocs
- **Version Control:** Git
- **Containerization:** Docker

---

## 5. Implementation Plan

### 5.1 Sprint Planning

**Sprint 1: Data Foundation (Week 1)**
- [ ] Load and validate VN30 OHLCV data
- [ ] Implement Parkinson volatility calculator
- [ ] Create HAR features (daily, weekly, monthly)
- [ ] Set up data pipeline with testing
- [ ] Document data characteristics

**Sprint 2: Baseline Models (Week 2)**
- [ ] Implement HAR-R baseline
- [ ] Implement HAR-X with daily proxies
- [ ] Create evaluation framework
- [ ] Establish performance benchmarks
- [ ] Document baseline results

**Sprint 3: Neural Models (Weeks 3-4)**
- [ ] Implement HAR-R + LSTM
- [ ] Implement pure LSTM baseline
- [ ] Create training pipeline for neural models
- [ ] Hyperparameter tuning and optimization
- [ ] Compare with baseline models

**Sprint 4: Graph Neural Models (Weeks 5-6)**
- [ ] Implement HAR-R + GNN
- [ ] Create stock correlation graph
- [ ] Train GNN model
- [ ] Evaluate GNN performance
- [ ] Compare with other approaches

**Sprint 5: Foundation Models (Weeks 7-8)**
- [ ] Implement TimesFM integration
- [ ] Fine-tune for volatility prediction
- [ ] Implement Chronos baseline
- [ ] Compare all approaches
- [ ] Select best performing model

**Sprint 6: Production Readiness (Weeks 9-10)**
- [ ] Optimize model for production
- [ ] Create prediction API
- [ ] Implement monitoring
- [ ] Final documentation
- [ ] Deployment preparation

### 5.2 Success Criteria

**Model Performance:**
- ✅ HAR-R baseline achieves RMSE < 0.15 for 1-day forecasts
- ✅ At least one advanced model improves over baseline by 10%+
- ✅ Directional accuracy > 55% for all horizons
- ✅ Models generalize across all 30 stocks

**Code Quality:**
- ✅ 85%+ test coverage maintained
- ✅ All code follows ML/DS Common Rules
- ✅ Zero critical code review issues
- ✅ Documentation complete

**Production Readiness:**
- ✅ Inference latency < 100ms per stock
- ✅ API uptime > 99%
- ✅ Monitoring and alerting functional
- ✅ Reproducible predictions with fixed seeds

---

## 6. Risk Assessment

### 6.1 Technical Risks

**Risk 1: Daily Data Insufficientity**
- **Impact:** High - Daily data may not capture intraday volatility patterns
- **Mitigation:** Use range-based estimators, compare multiple volatility proxies
- **Contingency:** Augment with additional features if performance inadequate

**Risk 2: Model Overfitting**
- **Impact:** Medium - Complex models may overfit to training data
- **Mitigation:** Strict time series cross-validation, regularization
- **Contingency:** Fall back to simpler models if overfitting detected

**Risk 3: Computational Resource Constraints**
- **Impact:** Medium - GNN and foundation models require significant resources
- **Mitigation:** Cloud-based training, optimized implementations
- **Contingency:** Scale back model complexity if necessary

### 6.2 Data Risks

**Risk 4: Data Quality Issues**
- **Impact:** High - Poor data quality leads to unreliable models
- **Mitigation:** Comprehensive data validation, outlier detection
- **Contingency:** Manual data cleaning if automated methods insufficient

**Risk 5: Non-Stationarity**
- **Impact:** Medium - Market regimes change over time
- **Mitigation:** Rolling window training, regime detection
- **Contingency:** Adaptive model retraining

---

## 7. Acceptance Criteria

### 7.1 Model Requirements

**AC-1: Baseline Performance**
- HAR-R model achieves acceptable baseline performance
- RMSE < 0.20 for all prediction horizons
- Improves over naive random walk baseline

**AC-2: Advanced Model Improvement**
- At least one advanced model improves over HAR-R by 10%+
- Models demonstrate consistent performance across stocks
- No severe overfitting observed

**AC-3: Multi-Horizon Accuracy**
- Directional accuracy > 55% for all horizons (1, 5, 10, **22 days**) ✅
- Performance degrades gracefully with longer horizons
- Confidence intervals properly calibrated

### 7.2 Code Quality Requirements

**AC-4: Testing Coverage**
- 85%+ code coverage maintained
- All critical paths have unit tests
- Integration tests cover end-to-end pipeline

**AC-5: Code Standards**
- Zero violations of ML/DS Common Rules
- All functions have appropriate documentation
- Code review approval required

### 7.3 Production Requirements

**AC-6: Performance**
- Inference latency < 100ms per stock
- Memory usage < 8GB for standard operations
- API response time < 1 second for 30 stocks

**AC-7: Reliability**
- 99%+ uptime for prediction service
- Automated monitoring and alerting
- Graceful error handling

---

## 8. Documentation Requirements

### 8.1 Technical Documentation

**TD-1: Architecture Documentation**
- System architecture diagrams
- Data flow documentation
- Model architecture specifications
- API documentation

**TD-2: Code Documentation**
- README with setup instructions
- Inline code documentation
- Function/class docstrings
- Usage examples

**TD-3: Model Documentation**
- Model performance reports
- Experiment tracking logs
- Hyperparameter configurations
- Training and validation curves

### 8.2 User Documentation

**UD-1: User Guide**
- How to run predictions
- How to interpret results
- Troubleshooting guide
- FAQ documentation

**UD-2: Developer Guide**
- Development setup instructions
- Code contribution guidelines
- Testing procedures
- Deployment guide

---

## 9. Quality Gates

### 9.1 Development Quality Gates

**QG-1: Code Review Gate**
- All code must pass peer review
- Zero critical issues
- Documentation complete
- Tests adequate

**QG-2: Testing Gate**
- All tests must pass
- Coverage threshold met
- Performance benchmarks met
- Security scans pass

**QG-3: Documentation Gate**
- Technical documentation complete
- User guides available
- API documentation current
- Examples provided

### 9.2 Model Quality Gates

**QG-4: Performance Gate**
- Baseline performance achieved
- Improvement over baseline demonstrated
- Cross-validation successful
- Generalization confirmed

**QG-5: Robustness Gate**
- Stress testing passed
- Edge cases handled
- Error recovery functional
- Monitoring operational

---

## 10. Success Metrics

### 10.1 Model Performance Metrics

**Primary Metrics:**
- **RMSE:** Root Mean Squared Error < 0.15 (1-day), < 0.25 (22-day)
- **MAE:** Mean Absolute Error < 0.12 (1-day), < 0.20 (22-day)
- **R²:** > 0.6 for all horizons
- **Directional Accuracy:** > 55% for all horizons (1, 5, 10, 22 days)

**Secondary Metrics:**
- **MAPE:** Mean Absolute Percentage Error < 25%
- **Theil's U:** < 1.0 (better than random walk)
- **Diebold-Mariano:** Significant improvement over baseline (p < 0.05)

**Secondary Metrics:**
- **MAPE:** Mean Absolute Percentage Error < 25%
- **Theil's U:** < 1.0 (better than random walk)
- **Diebold-Mariano:** Significant improvement over baseline (p < 0.05)

### 10.2 Project Success Metrics

**Delivery Metrics:**
- ✅ On-time delivery (10-week timeline)
- ✅ All functional requirements met
- ✅ All non-functional requirements met
- ✅ Documentation complete

**Quality Metrics:**
- ✅ 85%+ test coverage maintained
- ✅ Zero critical bugs in production
- ✅ Code review approval rate > 95%
- ✅ Documentation completeness > 90%

---

## Appendix A: References

**A1. Academic Papers**
- HAR-X: Ma & Zhang (2024) - "HAR-X: A Robust and Flexible Multi-Horizon Time Series Forecasting Model"
- HAR Original: Corsi (2009) - "A Simple Approximate Long-Memory Model of Realized Volatility"
- HAR-R: Range-based volatility estimators literature

**A2. Technical Resources**
- ML/DS Common Rules: `D:\bmad-projects\ml-ds-common-rules`
- Time Series Database Comparison: Tiger Data (2024)
- Foundation Models: TimesFM, Chronos documentation

**A3. Data Resources**
- VN30 Stock Data: `D:\bmad-projects\stock_vol_prediction01\data\raw\prices\`
- Data Summary: `collection_summary.csv`

---

## Appendix B: Glossary

**HAR:** Heterogeneous Autoregressive - Time series model using multiple time horizons

**HAR-R:** Range-based HAR - HAR adapted for daily OHLCV data using price ranges

**HAR-X:** Extended HAR with flexible multi-horizon forecasting capabilities

**RV:** Realized Volatility - Volatility measure from intraday high-frequency data

**OHLCV:** Open-High-Low-Close-Volume - Standard financial data format

**RMSE:** Root Mean Squared Error - Model accuracy metric

**GNN:** Graph Neural Network - Neural network for graph-structured data

**TimesFM:** Time Series Foundation Model - Google's pre-trained time series model

**VN30:** Vietnam 30 Index - Main Vietnamese stock market index

---

**Document Status:** Draft  
**Version:** 1.0  
**Last Updated:** 2026-06-15  
**Next Review:** After Sprint 2 completion