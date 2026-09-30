# 🤖 AI Stock Forecasting & Market Sentiment Analysis Platform
## Complete Project Documentation & PDF Guide (Hinglish Version)

> **Educational Disclaimer:** Yeh platform experimental machine-learning predictions generate karta hai jo research aur educational purpose ke liye hain. Yeh kisi bhi prakar ki **Financial Advice NAHI hai**. Past performance se future returns ki koi guarantee nahi hoti.

---

## 📋 Table of Contents (Vishay Suchi)
1. [Project Overview & Parichay](#1-project-overview--parichay)
2. [System Architecture (Design Diagram)](#2-system-architecture-design-diagram)
3. [Complete Technology Stack (Bariqiyan)](#3-complete-technology-stack-bariqiyan)
4. [Sare Features Aur Unka Use (Core Features)](#4-sare-features-aur-unka-use-core-features)
   - 4.1 [Market Dashboard & Price Charts](#41-market-dashboard--price-charts)
   - 4.2 [Technical Indicators Suite](#42-technical-indicators-suite)
   - 4.3 [6 Multi-Model Machine Learning Engine](#43-6-multi-model-machine-learning-engine)
   - 4.4 [FinBERT Financial News Sentiment Engine](#44-finbert-financial-news-sentiment-engine)
   - 4.5 [SHAP Prediction Explainability](#45-shap-prediction-explainability)
   - 4.6 [Model Comparison Registry](#46-model-comparison-registry)
   - 4.7 [Historical Backtesting Engine](#47-historical-backtesting-engine)
5. [Behind-The-Scenes Working (AI Workflow)](#5-behind-the-scenes-working-ai-workflow)
   - 5.1 [Data Fetching & Caching Pipeline](#51-data-fetching--caching-pipeline)
   - 5.2 [Feature Engineering (No Data Leakage)](#52-feature-engineering-no-data-leakage)
   - 5.3 [Time-Series Chronological Validation Split](#53-time-series-chronological-validation-split)
   - 5.4 [Training & Multi-Step Forecasting](#54-training--multi-step-forecasting)
   - 5.5 [Backtest Strategy Simulation](#55-backtest-strategy-simulation)
6. [Step-by-Step User Operating Guide](#6-step-by-step-user-operating-guide)
7. [REST API Endpoint Reference](#7-rest-api-endpoint-reference)
8. [PDF Export Karne Ka Tarika](#8-pdf-export-karne-ka-tarika)

---

## 1. Project Overview & Parichay

The **AI Stock Forecasting & Market Sentiment Analysis Platform** ek complete full-stack quantitative financial application hai. Is project ka main goal hai:
- Stock market ke **Technical Analysis** (Price & Volume signals), **Machine Learning Models** (XGBoost, Neural Networks), aur **Financial News Sentiment (FinBERT NLP)** ko ek saath combine karke future stock price trends ko predict karna.

### Major Capabilities:
- **Global & Indian Market Support**: US Stocks (`AAPL`, `MSFT`, `NVDA`, `TSLA`), Indian NSE Stocks (`TATAMOTORS.NS`, `RELIANCE.NS`, `TCS.NS`), Crypto (`BTC-USD`), aur Market Indices (`^NSEI`).
- **Dual Prediction Tasks**: **Regression** (future target price predict karna) aur **Classification** (future direction **UP** vs **DOWN** predict karna).
- **Explainable AI (XAI)**: SHAP values ke zariye yeh dekhna ki AI model ne yeh prediction kyun ki.
- **Realistic Backtesting**: Past data par simulation chala kar win-rate, return %, aur risk drawdown analyze karna.

---

## 2. System Architecture (Design Diagram)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Frontend (React 18 + Vite)                        │
│         (Dashboard, Charts, Model Training UI, Backtest Visualizer)     │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                             REST API (Axios)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Backend (FastAPI Engine)                         │
│            (API Routes, Model Registry, Async Background Tasks)         │
└─────────────────────────────────────────────────────────────────────────┘
        │                    │                    │                    │
        ▼                    ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data Layer  │     │  ML Pipeline │     │ Sentiment NLP│     │ Backtest Eng │
│ yfinance API │     │ XGBoost, LSTM│     │ ProsusAI     │     │ Equity Curve │
│ File Cache   │     │ SHAP Engine  │     │ FinBERT      │     │ Sharpe Ratio │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 3. Complete Technology Stack (Bariqiyan)

| Component | Used Technology | Role & Purpose (Kyun Use Hua?) |
| :--- | :--- | :--- |
| **Frontend Framework** | **React 18** | High-performance user interface, dynamic state management. |
| **Language** | **TypeScript** | Strict type safety taaki complex financial data schemas crash na hon. |
| **Build Tool** | **Vite 8** | Ultra-fast development server aur code bundler. |
| **Styling & UI** | **Vanilla CSS + Tailwind CSS 4** | Modern fintech dark theme, dynamic gradients, glassmorphism design. |
| **Data Visualization** | **Recharts** | Interactive price charts, RSI/MACD subcharts, aur equity curves. |
| **Icons & Alerts** | **Lucide-React & Hot-Toast** | Financial UI icons aur instant notification alerts. |
| **Backend Framework** | **Python 3.11/3.13 + FastAPI** | High-speed asynchronous Python REST API framework. |
| **Web Server** | **Uvicorn** | Asynchronous ASGI server jo backend API ko host karta hai. |
| **Database** | **SQLAlchemy 2.x + SQLite/Postgres** | Trained models, prediction records, aur backtest results ko store karne ke liye. |
| **Machine Learning** | **scikit-learn, XGBoost** | Tabular financial data par decision trees aur gradient boosting algorithms. |
| **Deep Learning** | **PyTorch 2.3** | Time-series sequential data ke liye Recurrent Neural Networks (LSTM & GRU). |
| **Explainable AI (XAI)**| **SHAP (SHapley Additive exPlanations)**| Feature importance breakdown karne ke liye (XGBoost/Linear models). |
| **NLP Sentiment** | **HuggingFace FinBERT** | `ProsusAI/finbert` financial news text classifier. |
| **Market Data Source**| **yfinance (Default)** | Daily live stock price data fetch karne ke liye (Supports Alpha Vantage & Polygon.io). |

---

## 4. Sare Features Aur Unka Use (Core Features)

### 4.1 Market Dashboard & Price Charts
- **Kyun Use Hota Hai**: Stock ka current market price, daily price change ($ aur %), 52-week high/low, aur volume dekhne ke liye.
- **Khas Baat**: Interactive Recharts chart mein `1M`, `3M`, `6M`, `1Y`, `5Y` timeframes par zoom kar sakte hain.

### 4.2 Technical Indicators Suite
- **Kyun Use Hota Hai**: Stock trend aur momentum direction samajhne ke liye.
- **Key Indicators**:
  1. **RSI (14)**: Relative Strength Index. `>70` = Overbought (Mehenga), `<30` = Oversold (Sasta).
  2. **MACD (12, 26, 9)**: Moving Average Convergence Divergence. Buy/Sell trend crossovers.
  3. **Moving Averages (SMA 50/200, EMA 12/26)**: Trend direction (Uptrend vs Downtrend).
  4. **Bollinger Bands**: Price volatility channels (Upper, Lower, & Middle bands).
  5. **ATR (14)**: Average True Range (Market risk level measure).

### 4.3 6 Multi-Model Machine Learning Engine
- **Kyun Use Hota Hai**: AI predictions generate karne ke liye.
- **6 Supported AI Models**:
  1. **Naive Baseline**: Baseline persistence model.
  2. **Linear Model**: Ridge Regression / Logistic Classifier.
  3. **Random Forest**: Multiple decision trees ka ensemble.
  4. **XGBoost**: Extreme Gradient Boosting (Top performing tabular ML algorithm).
  5. **LSTM (Long Short-Term Memory)**: Deep Learning RNN for sequential data.
  6. **GRU (Gated Recurrent Unit)**: Lightweight sequential RNN model.

### 4.4 FinBERT Financial News Sentiment Engine
- **Kyun Use Hota Hai**: News headlines ka market mood quantify karne ke liye.
- **Khas Baat**: News text ko **ProsusAI/finbert** model mein daal kar Positive, Neutral, aur Negative probabilities nikali jaati hain, aur Compound Score (-1.0 to +1.0) calculate hota hai.

### 4.5 SHAP Prediction Explainability
- **Kyun Use Hota Hai**: AI prediction ke piche ka logic (Why) samajhne ke liye.
- **Khas Baat**: Yeh batata hai ki konse feature (e.g. RSI spike, 5-day return lag) ne prediction ko kitna push kiya.

### 4.6 Model Comparison Registry
- **Kyun Use Hota Hai**: Sare trained models ko ek saath compare karke best model chunne ke liye.
- **Metrics**: Regression ke liye (MAE, RMSE, R², Directional Accuracy %) aur Classification ke liye (Accuracy %, Precision, Recall, F1-Score).

### 4.7 Historical Backtesting Engine
- **Kyun Use Hota Hai**: AI model signal par past trading strategy simulate karne ke liye.
- **Details**: \$10,000 initial capital se start karta hai. Outputs: Cumulative Return %, Win Rate %, Max Drawdown %, Sharpe Ratio, aur Equity Curve graph.

---

## 5. Behind-The-Scenes Working (AI Workflow)

```
[Raw Stock Data] ──► [Feature Engineering] ──► [Chronological Split]
                                                    │ (70% Train / 15% Val / 15% Test)
                                                    ▼
[Inference / Forecast] ◄── [Save Artifact (.pkl)] ◄── [Train Model (XGBoost/LSTM)]
```

### 5.1 Data Fetching & Caching Pipeline
1. Application `yfinance` API ke zariye market data mangwata hai.
2. Raw data ko local cache (`data/raw/{SYMBOL}.csv`) mein save kar leta hai taaki dobara request na karni pade.
3. Agar internet band ho ya API rate limit ho jaye, toh system automatically labeled demo sample data (`data/sample/`) par switch hakar UI par clear notice dikha deta hai.

### 5.2 Feature Engineering (No Data Leakage)
- Raw prices ko 30+ quantitative technical indicators aur lag features mein convert karta hai.
- Strict protection: Sabhi indicators sirf past data (`shift(1)`) use karte hain — future look-ahead 100% prevented hai.

### 5.3 Time-Series Chronological Split
- **Train (70%)**: Oldest historical data jo model aur scalers ko fit karta hai.
- **Validation (15%)**: Hyperparameter tuning.
- **Test (15%)**: Newest historical data jo unbiased evaluation ke liye use hota hai.
- *Scalers sirf Training split par fit hote hain (no data leakage).*

### 5.4 Training & Multi-Step Forecasting
1. Background workers mein model training chalti hai.
2. Trained artifacts (weights, feature lists, fitted scalers) `./models/{id}.pkl` mein save hote hain.
3. Inferences `.pkl` file load karke 1, 5, 10, 20-day horizon forecast generate karte hain.

### 5.5 Backtest Strategy Simulation
- Starting Capital = \$10,000.
- Daily iteration:
  - If Model Prediction == `UP` ➔ Enter Long Position (Capture next-day return).
  - If Model Prediction == `DOWN` ➔ Stay Flat (0% return, capital safe).
- Sharpe Ratio Calculation: $\text{Sharpe} = \frac{\text{Mean Daily Return}}{\text{Std Daily Return}} \times \sqrt{252}$.

---

## 6. Application Chalane Ka Step-by-Step Guide

### Step 1: Application Kholein
Browser mein kholein: **[http://localhost:5173](http://localhost:5173)**

### Step 2: Stock Select Ya Search Karein
- Top bar mein **Stock Selector** dropdown (jaise `AAPL ▾`) par click karein.
- Ticker type karein (e.g. `TATAMOTORS.NS`, `NVDA`, `RELIANCE.NS`, `TSLA`) aur **Enter** dabaayein.

### Step 3: Technical Indicators Dekhein
- Dashboard par price chart, RSI, MACD, aur Moving Averages check karein.

### Step 4: AI Model Train Karein
- Navigation mein **Forecast ➔ Train Model** par jayein.
- Symbol, Model Algorithm (XGBoost/LSTM), Task Type, aur Feature Set select karke **Start Training** par click karein.

### Step 5: Predictions & SHAP Explainability Dekhein
- **Forecast ➔ Predict** par jayein.
- Model aur Forecast Horizon (1 to 20 days) select karke **Generate Forecast** par click karein.

### Step 6: Backtesting Strategy Run Karein
- **Backtesting** tab par jayein.
- Model aur Date Range choose karke **Run Backtest** par click karein aur Win Rate % & Equity Curve check karein!

---

## 7. REST API Endpoint Reference

| Category | Endpoint | Method | Purpose (Description) |
| :--- | :--- | :--- | :--- |
| **Health** | `/api/health` | `GET` | System health & DB connection status. |
| **Stocks** | `/api/stocks` | `GET` | Tracked stocks list. |
| **Stock Info**| `/api/stocks/{symbol}` | `GET` | Stock metadata (name, sector, market cap). |
| **History** | `/api/stocks/{symbol}/history` | `GET` | Daily OHLCV price history array. |
| **Indicators**| `/api/stocks/{symbol}/indicators` | `GET` | RSI, MACD, Bollinger Bands, ATR points. |
| **Refresh** | `/api/data/refresh` | `POST` | Fresh market data download trigger. |
| **Models** | `/api/models/train` | `POST` | Asynchronous model training launch. |
| **Model Run** | `/api/models/runs/{run_id}` | `GET` | Poll training status & progress. |
| **Predict** | `/api/predict` | `POST` | Generate multi-step forecast. |
| **Explain** | `/api/explain/{prediction_id}`| `GET` | SHAP feature attributions. |
| **Sentiment**| `/api/sentiment/analyze` | `POST` | FinBERT financial news sentiment. |
| **Backtest** | `/api/backtest` | `POST` | Historical strategy simulation. |

---

## 8. PDF Export Karne Ka Tarika

Iss document ko PDF mein save/export karne ke 3 aasan steps:

1. Apne computer mein **`docs/AI_Stock_Forecasting_Platform_Documentation_Hinglish.md`** file kholein.
2. Keyboard par **`Ctrl + P`** (macOS par **`Cmd + P`**) press karein.
3. Destination dropdown mein **"Save as PDF"** select karke **Save** button par click kar dein!
