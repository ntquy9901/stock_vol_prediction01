# VN30 Sentiment Analysis Implementation Plan

**Project:** Stock Volatility Prediction - Sentiment Integration  
**Date:** 2026-06-27  
**Status:** Planning Phase - Ready for Implementation  
**Timeline:** 16 weeks (aligned with technical research roadmap)

---

## Executive Summary

This plan implements **AI-driven sentiment analysis** for VN30 stocks using the technical research findings, focusing on **agentic AI approaches** and **financial-specific models** (FinBERT) to enhance volatility prediction accuracy.

**Key Targets:**
- **Sentiment Accuracy:** > 80% (human-labeled benchmark)
- **Processing Latency:** < 5 seconds for daily sentiment updates
- **Cost Optimization:** 40-85% LLM API cost reduction through optimization
- **Integration Ready:** Compatible with existing volatility prediction pipeline

---

## Phase 1: Foundation (Weeks 1-4) - Data Infrastructure

### **Week 1-2: Environment Setup & Data Collection**

#### **1.1 Technology Stack Installation**
```bash
# Core dependencies (based on technical research)
pip install transformers torch pandas numpy
pip install fastapi uvicorn
pip install crewai langchain-community
pip install kafka-python redis
pip install yfinance requests beautifulsoup4
```

**Selected Technologies (from research):**
- **Language:** Python 3.12+
- **Framework:** FastAPI (high-performance async)
- **AI Agents:** CrewAI (82% task success rate, easier learning curve)
- **NLP Model:** FinBERT (financial sentiment analysis)
- **Database:** Redis (cache), TimescaleDB (time-series storage)
- **Message Queue:** Kafka (event-driven architecture)

#### **1.2 News Source Integration**

**Primary Vietnamese Financial News Sources:**
1. **VN_Express** (https://vnexpress.net/kinh-doanh/tai-chinh)
2. **CafeF** (https://cafef.vn)
3. **Stocking.vn** (https://stocking.vn)
4. **Vietstock** (https://vietstock.vn)
5. **Kenh Tai Chinh** (https://kenhtaichinh.vn)

**Collection Strategy:**
```python
# src/sentiment/data_collection/news_collector.py
class NewsCollector:
    """Collect financial news for VN30 stocks"""
    
    def __init__(self):
        self.sources = {
            'vnexpress': VnExpressParser(),
            'cafef': CafeFParser(),
            'stocking': StockingParser(),
            'vietstock': VietstockParser(),
            'kenhtaichinh': KenhTaiChinhParser()
        }
        self.tickers = self._get_vn30_tickers()
    
    def collect_daily_news(self, date: str) -> List[NewsArticle]:
        """Collect all news articles for given date"""
        articles = []
        for source, parser in self.sources.items():
            articles.extend(parser.fetch_news(date, self.tickers))
        return articles
    
    def save_raw_data(self, articles: List[NewsArticle], output_path: str):
        """Save raw articles to CSV"""
        df = pd.DataFrame([art.__dict__ for art in articles])
        df.to_csv(output_path, index=False)
```

#### **1.3 Social Media Integration**

**Social Media Sources:**
1. **Twitter/X** - Official company accounts, financial hashtags
2. **StockTwits** - Stock-focused social platform
3. **Reddit** - r/vietnamstocks, r/investing
4. **Facebook Groups** - Vietnamese investment communities

**Collection Method:**
- **API Collection:** Official APIs where available (Twitter API v2)
- **Web Scraping:** For platforms without APIs (ethical scraping, rate-limited)
- **Agentic AI:** Automated agents for content discovery and collection

---

### **Week 3-4: Sentiment Analysis Implementation**

#### **1.4 Core Sentiment Model - FinBERT**

**Why FinBERT?**
- **Financial-specific:** Trained on financial text, outperforms general models
- **Multilingual:** Supports Vietnamese (through translation)
- **Proven Accuracy:** 83-85% accuracy on financial sentiment
- **Open Source:** No API costs, can be deployed locally

**Implementation:**
```python
# src/sentiment/models/finbert_sentiment.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FinBERTSentiment:
    """Financial sentiment analysis using FinBERT"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Sentiment labels
        self.labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    def analyze_text(self, text: str) -> SentimentResult:
        """Analyze sentiment of single text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get probabilities
        probs = predictions.cpu().numpy()[0]
        sentiment_score = (probs[2] - probs[0])  # Positive - Negative
        
        return SentimentResult(
            sentiment_score=float(sentiment_score),
            sentiment_label=self.labels[probs.argmax()],
            positive_score=float(probs[2]),
            negative_score=float(probs[0]),
            neutral_score=float(probs[1])
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for multiple texts (batch processing)"""
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        return results
```

#### **1.5 Agentic AI Enhancement - CrewAI**

**Multi-Agent Architecture (from technical research):**
```python
# src/sentiment/processing/agent_orchestrator.py
from crewai import Agent, Task, Crew

class SentimentAgentOrchestrator:
    """Agentic AI system for enhanced sentiment analysis"""
    
    def __init__(self):
        # Agent 1: News Discovery Agent
        self.news_discovery_agent = Agent(
            role="News Discovery Specialist",
            goal="Find relevant financial news for VN30 stocks",
            backstory="Expert in Vietnamese financial news discovery and curation",
            tools=[web_search_tool, news_scraper_tool],
            verbose=True
        )
        
        # Agent 2: Sentiment Analysis Agent  
        self.sentiment_agent = Agent(
            role="Financial Sentiment Analyst",
            goal="Analyze sentiment using FinBERT with financial context",
            backstory="Expert in financial sentiment analysis with domain knowledge",
            tools=[finbert_tool, translation_tool],
            verbose=True
        )
        
        # Agent 3: Quality Assurance Agent
        self.qa_agent = Agent(
            role="Quality Assurance Specialist", 
            goal="Validate sentiment analysis accuracy and consistency",
            backstory="Expert in data quality validation and error detection",
            tools=[validation_tool, human_review_tool],
            verbose=True
        )
    
    def process_daily_sentiment(self, date: str) -> DailySentimentReport:
        """Process sentiment analysis for given date"""
        
        # Task 1: Collect news
        collection_task = Task(
            description=f"Collect all VN30 news for {date}",
            agent=self.news_discovery_agent,
            expected_output="List of news articles with metadata"
        )
        
        # Task 2: Analyze sentiment
        analysis_task = Task(
            description="Analyze sentiment using FinBERT",
            agent=self.sentiment_agent,
            expected_output="Sentiment scores with confidence levels"
        )
        
        # Task 3: Quality validation
        validation_task = Task(
            description="Validate sentiment analysis quality",
            agent=self.qa_agent,
            expected_output="Validated sentiment report with confidence metrics"
        )
        
        # Create crew and execute
        crew = Crew(
            agents=[
                self.news_discovery_agent,
                self.sentiment_agent, 
                self.qa_agent
            ],
            tasks=[collection_task, analysis_task, validation_task],
            verbose=True
        )
        
        result = crew.kickoff()
        return self._parse_result(result)
```

**Benefits of Agentic Approach:**
- **82% Task Success Rate** (CrewAI performance)
- **Parallel Processing**: Agents work simultaneously
- **Error Recovery**: Self-healing and retry logic
- **Human Oversight**: Critical decisions require human approval

---

## Phase 2: Enhancement (Weeks 5-8) - Advanced Features

### **Week 5-6: HAR-like Feature Engineering**

**Sentiment Feature Implementation:**
```python
# src/sentiment/processing/har_sentiment_features.py
class HARSentimentFeatures:
    """HAR-like sentiment feature engineering"""
    
    def __init__(self, daily_windows: List[int] = [1, 5, 22]):
        self.daily_windows = daily_windows  # Daily, Weekly, Monthly
    
    def create_har_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Create HAR-like sentiment features"""
        
        features_df = sentiment_df.copy()
        
        # HAR sentiment windows
        for window in self.daily_windows:
            features_df[f'sent_{window}d'] = (
                features_df['sentiment_score']
                .rolling(window=window, min_periods=1)
                .mean()
            )
        
        # Moving averages
        features_df['sent_ma5'] = (
            features_df['sentiment_score']
            .rolling(window=5, min_periods=1)
            .mean()
        )
        features_df['sent_ma22'] = (
            features_df['sentiment_score']
            .rolling(window=22, min_periods=1)
            .mean()
        )
        
        # Sentiment volatility
        features_df['sentiment_volatility'] = (
            features_df['sentiment_score']
            .rolling(window=22, min_periods=5)
            .std()
        )
        
        # Sentiment momentum
        features_df['sentiment_momentum'] = (
            features_df['sentiment_score']
            .diff(periods=1)
        )
        
        # Trend indicators
        features_df['positive_trend'] = (
            features_df['sent_ma5'] > features_df['sent_ma22']
        )
        
        return features_df
```

### **Week 7-8: Cost Optimization Implementation**

**LLM Cost Management Strategies (from research):**
```python
# src/sentiment/processing/cost_optimizer.py
class SentimentCostOptimizer:
    """Optimize sentiment analysis costs (40-85% reduction)"""
    
    def __init__(self):
        self.cache = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = 86400  # 24 hours
    
    def analyze_with_cache(self, text: str) -> SentimentResult:
        """Check cache before expensive model inference"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(text)
        
        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result:
            print(f"Cache hit! Saving API call cost...")
            return pickle.loads(cached_result)
        
        # Cache miss - run model
        result = self.finbert_model.analyze_text(text)
        
        # Store in cache
        self.cache.setex(
            cache_key, 
            self.cache_ttl, 
            pickle.dumps(result)
        )
        
        return result
    
    def batch_process_optimization(self, texts: List[str]) -> List[SentimentResult]:
        """Optimize batch processing"""
        
        # Strategy 1: Remove duplicates
        unique_texts = list(set(texts))
        
        # Strategy 2: Cache-aware processing
        results = []
        cache_hits = 0
        
        for text in unique_texts:
            cache_key = self._generate_cache_key(text)
            if self.cache.exists(cache_key):
                results.append(pickle.loads(self.cache.get(cache_key)))
                cache_hits += 1
            else:
                result = self.finbert_model.analyze_text(text)
                results.append(result)
                self.cache.setex(cache_key, self.cache_ttl, pickle.dumps(result))
        
        print(f"Cache hit rate: {cache_hits}/{len(unique_texts)} ({cache_hits/len(unique_texts)*100:.1f}%)")
        
        # Strategy 3: Batch size optimization for GPU utilization
        # Process in batches of 16 (optimal for most GPUs)
        return results
```

**Expected Cost Savings:**
- **Caching**: 60-80% reduction for repeated content
- **Batch Processing**: 30-50% reduction through GPU optimization
- **Model Optimization**: Additional 10-15% through quantization
- **Total Savings**: 40-85% cost reduction achievable

---

## Phase 3: Integration (Weeks 9-12) - System Integration

### **Week 9-10: Pipeline Integration**

**Integration with Existing Volatility Pipeline:**
```python
# src/sentiment/processing/pipeline_integration.py
class SentimentPipelineIntegration:
    """Integrate sentiment with existing volatility prediction pipeline"""
    
    def __init__(self):
        self.volatility_data_path = 'data/processed/vn30_only/'
        self.sentiment_data_path = 'data/processed/vn30_sentiment/daily/'
    
    def create_combined_dataset(self, ticker: str) -> pd.DataFrame:
        """Combine price and sentiment data"""
        
        # Load volatility data
        vol_df = pd.read_csv(
            f'{self.volatility_data_path}{ticker}_processed.csv'
        )
        
        # Load sentiment data
        sent_df = pd.read_csv(
            f'{self.sentiment_data_path}{ticker}_sentiment_daily.csv'
        )
        
        # Merge on date
        combined_df = pd.merge(
            vol_df,
            sent_df[['date', 'sentiment_score', 'sent_daily', 'sent_weekly', 'sent_monthly']],
            on='date',
            how='inner'
        )
        
        # Create trading signals
        combined_df['combined_signal'] = self._create_trading_signals(combined_df)
        
        return combined_df
    
    def _create_trading_signals(self, df: pd.DataFrame) -> pd.Series:
        """Create trading signals based on price + sentiment"""
        
        signals = []
        
        for idx, row in df.iterrows():
            # Buy signal: Price drop + High positive sentiment
            if (row['close'] < row['ma20']) and (row['sentiment_score'] > 0.5):
                signals.append('Buy_Signal')
            
            # Sell signal: Price rise + High negative sentiment  
            elif (row['close'] > row['ma20']) and (row['sentiment_score'] < -0.3):
                signals.append('Sell_Signal')
            
            # Hold signal
            else:
                signals.append('Hold')
        
        return pd.Series(signals, index=df.index)
```

### **Week 11-12: API and Real-time Processing**

**FastAPI Implementation:**
```python
# src/sentiment/processing/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="VN30 Sentiment Analysis API")

class SentimentRequest(BaseModel):
    tickers: List[str]
    date_range: tuple[str, str]

class SentimentResponse(BaseModel):
    ticker: str
    date: str
    sentiment_score: float
    sentiment_label: str
    confidence: float

@app.post("/api/v1/sentiment/batch")
async def analyze_sentiment_batch(request: SentimentRequest) -> List[SentimentResponse]:
    """Batch sentiment analysis endpoint"""
    
    results = []
    
    for ticker in request.tickers:
        # Get sentiment data
        sentiment_data = sentiment_service.get_daily_sentiment(
            ticker, 
            request.date_range
        )
        
        results.append(sentiment_data)
    
    return results

@app.get("/api/v1/sentiment/{ticker}/latest")
async def get_latest_sentiment(ticker: str) -> SentimentResponse:
    """Get latest sentiment for specific ticker"""
    
    try:
        sentiment = sentiment_service.get_latest_sentiment(ticker)
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
```

**Real-time WebSocket Integration:**
```python
# src/sentiment/processing/websocket_handler.py
from fastapi import WebSocket
import asyncio
import json

@app.websocket("/ws/sentiment/stream")
async def sentiment_websocket(websocket: WebSocket):
    """Real-time sentiment streaming"""
    
    await websocket.accept()
    
    try:
        while True:
            # Get real-time sentiment updates
            sentiment_updates = await sentiment_service.stream_updates()
            
            # Send to client
            await websocket.send_json(sentiment_updates)
            
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
```

---

## Phase 4: Production (Weeks 13-16) - Deployment

### **Week 13-14: Production Deployment**

**Docker Configuration:**
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "sentiment.processing.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Kubernetes Deployment:**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment
  template:
    metadata:
      labels:
        app: sentiment
    spec:
      containers:
      - name: sentiment-api
        image: sentiment-analysis:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

### **Week 15-16: Monitoring and Optimization**

**Monitoring Implementation:**
```python
# src/sentiment/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
sentiment_requests = Counter('sentiment_requests_total', 'Total sentiment requests')
sentiment_latency = Histogram('sentiment_latency_seconds', 'Sentiment analysis latency')
sentiment_errors = Counter('sentiment_errors_total', 'Total sentiment errors')
cache_hit_rate = Gauge('sentiment_cache_hit_rate', 'Cache hit rate')

@sentiment_latency.time()
def analyze_sentiment_with_monitoring(text: str):
    """Analyze sentiment with monitoring"""
    sentiment_requests.inc()
    
    try:
        result = sentiment_service.analyze(text)
        return result
    except Exception as e:
        sentiment_errors.inc()
        raise e
```

---

## Cost Management Strategy

### **Expected Costs (Monthly)**

**Without Optimization:**
- **FinBERT API Calls**: ~$500/month (estimated)
- **Cloud Infrastructure**: ~$300/month
- **Data Storage**: ~$50/month
- **Total**: ~$850/month

**With Optimization (40-85% reduction):**
- **FinBERT Costs**: ~$100/month (80% reduction via caching + batch processing)
- **Cloud Infrastructure**: ~$200/month (spot instances + rightsizing)
- **Data Storage**: ~$30/month (compression + lifecycle policies)
- **Total**: ~$330/month (61% savings)

---

## Success Metrics and KPIs

### **Technical Metrics**
- **Sentiment Accuracy**: > 80% (vs human-labeled benchmark)
- **API Latency**: < 100ms (p95)
- **Processing Latency**: < 5 seconds for daily updates
- **System Uptime**: > 99.9%

### **Business Metrics**
- **Cost Reduction**: 40-85% LLM cost savings
- **Trading Performance**: Improved Sharpe ratio with sentiment integration
- **Data Coverage**: > 90% of trading days have sentiment data

### **Operational Metrics**
- **Deployment Frequency**: Weekly updates for model improvements
- **Mean Time to Recovery**: < 1 hour for production issues
- **On-call Response**: < 15 minutes

---

## Risk Assessment

### **Technical Risks**
- **Model Accuracy**: FinBERT may not capture Vietnamese nuances
  - **Mitigation**: Translation to English, fine-tune on Vietnamese financial text
  
- **API Rate Limits**: News sources may block scraping
  - **Mitigation**: Respectful scraping, API usage where available

- **Data Quality**: Noisy social media data
  - **Mitigation**: Quality scoring, human review for critical decisions

### **Operational Risks**
- **Cost Overruns**: LLM costs may exceed budget
  - **Mitigation**: Strict cost monitoring, optimization strategies

- **Compliance**: Data privacy and regulatory requirements
  - **Mitigation**: GDPR compliance, audit trails, data retention policies

---

## Next Steps

1. **Week 1**: Set up development environment and install dependencies
2. **Week 2**: Implement news collection from 2-3 sources
3. **Week 3**: Deploy FinBERT model and test on sample data
4. **Week 4**: Create initial sentiment CSV files for all VN30 stocks

**Ready to begin implementation?**

---

**Implementation Plan Version:** 1.0  
**Last Updated:** 2026-06-27  
**Based On:** Comprehensive Technical Research (2026-06-27)  
**Compliance:** Follows ML/DS common rules and project standards
