# 🤖 AI Stock Forecasting & Market Sentiment Analysis Platform

> **Educational Disclaimer:** Predictions are experimental machine-learning forecasts for educational and research purposes only and are **not financial advice**. Past performance does not guarantee future results.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Project Overview

A complete, production-quality AI platform for stock market forecasting and sentiment analysis. Built as an MCA AI/ML portfolio project demonstrating:

- **Multi-model ML pipeline** (LSTM, GRU, XGBoost, Random Forest, Linear, Naive)
- **Proper time-series validation** (chronological split, no leakage)
- **Financial news sentiment** analysis with FinBERT
- **SHAP-based explainability** for model interpretations
- **Historical backtesting** with equity curves
- **Full-stack architecture** (FastAPI + React + PostgreSQL + Docker)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 Market Dashboard | Live price, volume, 52W high/low, interactive charts |
| 📈 Technical Indicators | RSI, MACD, Bollinger Bands, ATR, moving averages |
| 🧠 6 ML Models | Naive, Linear, Random Forest, XGBoost, LSTM, GRU |
| 📉 Regression & Classification | Predict price or direction (UP/DOWN) |
| 🔮 Multi-horizon Forecast | 1, 5, 10, or 20 period forecasts |
| 📰 News Sentiment | FinBERT-powered financial news analysis |
| 📊 Sentiment + Market Models | Compare market-only vs. market + sentiment |
| ⚗️ Backtesting | Directional strategy simulation with equity curves |
| 💡 SHAP Explainability | Feature attribution for individual predictions |
| 🗂️ Model Registry | Version tracking for all trained models |
| 📜 Prediction History | Persistent storage of all predictions |
| 🌗 Dark Theme | Professional fintech dashboard UI |

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Frontend ["Frontend (React + Vite)"]
        UI[Dashboard / Pages]
        Charts[Recharts]
    end

    subgraph Backend ["Backend (FastAPI)"]
        API[REST API Routes]
        ML[ML Pipeline]
        Sent[Sentiment Engine]
        BT[Backtesting Engine]
    end

    subgraph Data ["Data Layer"]
        PG[(PostgreSQL)]
        Cache[File Cache]
    end

    subgraph External ["External Services"]
        YF[yfinance]
        AV[Alpha Vantage]
        NW[NewsAPI / FinBERT]
        HF[HuggingFace Hub]
    end

    UI --> API
    API --> ML
    API --> Sent
    API --> BT
    ML --> PG
    ML --> Cache
    Sent --> PG
    API --> PG
    ML --> YF
    ML --> AV
    Sent --> NW
    Sent --> HF
```

---

## 🤖 ML Pipeline

```mermaid
flowchart LR
    A[Raw OHLCV] --> B[Feature Engineering]
    B --> C[30+ Features]
    C --> D{Chronological Split}
    D --> E[Train 70%]
    D --> F[Val 15%]
    D --> G[Test 15%]
    E --> H[Fit Scaler]
    H --> I[Scale All Splits]
    I --> J{Model Training}
    J --> K[XGBoost]
    J --> L[LSTM/GRU]
    J --> M[Random Forest]
    K & L & M --> N[Evaluate on Test]
    N --> O[Save Artifact]
    O --> P[Inference / Forecast]
```

---

## 📐 Feature Engineering

| Category | Features |
|----------|----------|
| Price | Open, High, Low, Close, Volume |
| Returns | Daily, Weekly, Monthly returns |
| Moving Averages | SMA 5/10/20/50, EMA 12/26 |
| Technical | RSI(14), MACD(12/26/9), Bollinger Bands, ATR |
| Volatility | Rolling std (5/20 day), Historical volatility |
| Lag Features | Close lag 1/2/3/5/10, Return lag 1/2/3/5/10 |
| Volume | Volume change, Volume SMA, Volume ratio |
| Position | %B (Bollinger), % from 52W high/low |
| Sentiment (optional) | FinBERT score, lag1, 3-day rolling |

> **Leakage Protection:** All features use only past information. Scalers fit on training data only.

---

## 🧪 Validation Strategy

```
Total data:  |─────── TRAIN (70%) ───────|── VAL (15%) ──|── TEST (15%) ──|
Time:         oldest                                                     newest
```

**Why not random split?** Time-series data is ordered. Random splitting allows the model to "see" future data during training, inflating metrics by 10–40%. This project enforces strict chronological ordering.

---

## 📊 Evaluation Metrics

| Task | Metrics |
|------|---------|
| Regression | MAE, RMSE, R², Directional Accuracy |
| Classification | Accuracy, Precision, Recall, F1, ROC-AUC, Directional Accuracy |

---

## 💬 Sentiment Analysis

Pipeline:
```
News Headlines → Text Cleaning → FinBERT → [Positive/Neutral/Negative]
                                          → Compound Score [-1, 1]
                                          → Daily Aggregate
                                          → DB Storage
                                          → Feature for ML
```

**Model:** `ProsusAI/finbert` — BERT fine-tuned on 10,000+ financial documents

---

## ⚗️ Backtesting

Strategy: **Directional Long-Flat**
- Predict UP → go long
- Predict DOWN → stay flat

Metrics: Total Trades, Win Rate, Cumulative Return, Max Drawdown, Sharpe Ratio

> ⚠️ This is an educational backtest. Does NOT account for transaction costs, slippage, or real-world execution.

---

## 🔍 Explainability

- **SHAP TreeExplainer** for XGBoost and Random Forest
- **SHAP LinearExplainer** for Linear models
- Per-prediction feature attributions
- Mean |SHAP| for global feature importance

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Recharts |
| Backend | Python 3.11, FastAPI, Pydantic v2, Uvicorn |
| ML | scikit-learn, XGBoost, PyTorch, SHAP |
| NLP | Transformers, FinBERT (ProsusAI/finbert) |
| Database | PostgreSQL 15, SQLAlchemy 2.x, Alembic |
| Data | yfinance (default), Alpha Vantage, Polygon.io |
| DevOps | Docker, Docker Compose |
| Testing | pytest |

---

## 📁 Project Structure

```
stock-forecasting-platform/
│
├── frontend/               # React + TypeScript + Vite
│   ├── src/
│   │   ├── api/            # Axios API client
│   │   ├── components/     # UI components
│   │   ├── context/        # React Context
│   │   └── pages/          # Page components
│   └── vite.config.ts
│
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI app entry
│   │   ├── config.py       # Pydantic settings
│   │   ├── database.py     # SQLAlchemy engine
│   │   ├── models/         # ORM models
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── api/            # Route handlers
│   │   ├── services/       # Market data service
│   │   ├── ml/             # Training & inference
│   │   ├── sentiment/      # FinBERT pipeline
│   │   └── backtesting/    # Backtest engine
│   └── tests/
│
├── data/
│   ├── raw/                # Cached OHLCV CSVs
│   ├── processed/          # Feature matrices
│   └── sample/             # Labeled synthetic fallback data
│
├── models/                 # Saved model artifacts (.pkl)
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts
├── docs/                   # Documentation
│
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Node.js 20+
- PostgreSQL 15 (or Docker)
- Git

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/stock-forecasting-platform.git
cd stock-forecasting-platform
```

### 2. Environment Variables

```bash
cp .env.example .env
# Edit .env with your settings
```

Required for basic operation (yfinance — no API key needed):
```env
DATABASE_URL=postgresql://stockuser:stockpassword@localhost:5432/stockdb
MARKET_DATA_PROVIDER=yfinance
NEWS_PROVIDER=sample  # Use 'newsapi' with NEWS_API_KEY for live news
```

### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Generate sample data (optional, for offline development)
cd ..
python scripts/generate_sample_data.py
cd backend
```

### 4. Database Setup

```bash
# Start PostgreSQL (or use Docker)
docker run -d -p 5432:5432 \
  -e POSTGRES_USER=stockuser \
  -e POSTGRES_PASSWORD=stockpassword \
  -e POSTGRES_DB=stockdb \
  postgres:15-alpine

# Tables are created automatically on first startup
```

### 5. Start Backend

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend: http://localhost:8000
API Docs: http://localhost:8000/docs

### 6. Frontend Setup

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Frontend: http://localhost:5173

---

## 🐳 Docker Setup

```bash
# Copy and configure .env
cp .env.example .env

# Start everything (PostgreSQL + Backend + Frontend)
docker compose up --build

# Frontend: http://localhost:5173
# Backend:  http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## 🔑 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `MARKET_DATA_PROVIDER` | `yfinance`, `alpha_vantage`, or `polygon` | `yfinance` |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key | Optional |
| `POLYGON_API_KEY` | Polygon.io API key | Optional |
| `NEWS_PROVIDER` | `newsapi` or `sample` | `sample` |
| `NEWS_API_KEY` | NewsAPI key for live news | Optional |
| `MODEL_ARTIFACTS_DIR` | Where to save trained models | `./models` |
| `DATA_DIR` | Data directory | `./data` |
| `SECRET_KEY` | App secret key | Change in prod |
| `ALLOWED_ORIGINS` | CORS allowed origins | localhost URLs |

---

## 📖 Running Locally — Step by Step

### Fetch Market Data
```
UI: Click "Refresh Data" in the top bar → selects current stock

API: POST /api/data/refresh
{
  "symbol": "AAPL",
  "force": false
}
```

### Train a Model
```
UI: Forecast → Train Model → Select model type, task, feature set → Start Training

API: POST /api/models/train
{
  "symbol": "AAPL",
  "model_type": "xgboost",
  "task": "regression",
  "feature_set": "market",
  "lookback_window": 60
}

# Poll status:
GET /api/models/runs/{run_id}
```

### Generate a Prediction
```
UI: Forecast → Generate Forecast → Select model + horizon → Generate

API: POST /api/predict
{
  "symbol": "AAPL",
  "model_id": 1,
  "horizon": 5
}
```

### Run Sentiment Analysis
```
UI: Sentiment → Analyze News

API: POST /api/sentiment/analyze
{"symbol": "AAPL", "days": 7}
```

### Run Backtest
```
UI: Backtesting → Configure → Run Backtest

API: POST /api/backtest
{
  "symbol": "AAPL",
  "model_id": 1,
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

---

## 🧪 Running Tests

```bash
cd backend
pip install -r requirements.txt
pytest -v

# Tests cover:
# - Feature engineering (leakage, indicators, splits)
# - Backtesting calculations (drawdown, Sharpe, strategy)
# - Market data validation and cleaning
# - Sentiment analysis aggregation
```

---

## 📡 API Documentation

Full OpenAPI documentation available at: **http://localhost:8000/docs**

Key endpoints:

```
GET  /api/health                    Health check
GET  /api/stocks                    List tracked stocks
GET  /api/stocks/{symbol}           Stock info
GET  /api/stocks/{symbol}/history   Price history
GET  /api/stocks/{symbol}/indicators Technical indicators
POST /api/data/refresh              Refresh market data
POST /api/models/train              Train ML model (background)
GET  /api/models/runs/{id}          Training status
GET  /api/models                    List all models
POST /api/predict                   Generate forecast
GET  /api/predictions               Prediction history
GET  /api/explain/{prediction_id}   SHAP explanation
POST /api/sentiment/analyze         Analyze news sentiment
POST /api/backtest                  Run backtest
```

---

## ⚠️ Limitations

1. **Market efficiency**: Known patterns are rapidly arbitraged away by professional quants
2. **Regime change**: Models trained in bull markets may fail in bear markets
3. **External events**: Earnings releases, Fed decisions, geopolitical events cannot be predicted from price history
4. **Backtesting does not include**: transaction costs, slippage, market impact, taxes
5. **Directional accuracy near 50%** may not be statistically significant — needs proper testing

---

## 🔮 Future Improvements

- [ ] Walk-forward cross-validation (20+ folds)
- [ ] Temporal Fusion Transformer (TFT)
- [ ] Ensemble model (blend XGBoost + LSTM)
- [ ] MLflow experiment tracking
- [ ] Optuna hyperparameter optimization
- [ ] WebSocket real-time price updates
- [ ] Options chain integration
- [ ] Statistical significance testing

---

## 🙋 Author

**[Your Name]**
MCA (Artificial Intelligence & Machine Learning)
[Your University]

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## ⚖️ Disclaimer

> This software is provided for **educational and research purposes only**. It is NOT financial advice. The predictions generated by this system are experimental machine-learning outputs and should not be used as a basis for real investment decisions. The authors accept no liability for any financial losses arising from the use of this software.
