# Resume Description — AI Stock Forecasting & Market Sentiment Analysis Platform

## Project Title
**AI Stock Forecasting & Market Sentiment Analysis Platform**

---

## 3 Strong Resume Bullet Points

> ⚠️ **Important**: Fill in the bracketed values with your actual measured results after running the pipeline. Never invent numbers.

- **Engineered an end-to-end ML forecasting pipeline** using LSTM, GRU, XGBoost, and Random Forest on 5+ years of OHLCV data, implementing chronological train/val/test splits (70/15/15) and SHAP explainability to achieve **[X]% directional accuracy on the test set** for AAPL, achieving **[X]% improvement over naive baseline**.

- **Built a full-stack financial analytics platform** (FastAPI + React + PostgreSQL + Docker) featuring real-time market data ingestion via yfinance/Polygon, FinBERT-based news sentiment analysis, historical backtesting (**[X]% simulated return, [X] Sharpe ratio** on 1-year backtest), and REST APIs serving **6 ML models** with asynchronous background training.

- **Implemented production-quality ML practices** including data-leakage-free feature engineering (30+ features: RSI, MACD, Bollinger Bands, lag features), scaler fitting exclusively on training data, walk-forward inference for backtesting, and containerized deployment with Docker Compose.

---

## Short Project Description (For CV / GitHub Bio)

An end-to-end AI-powered stock market analysis and forecasting platform demonstrating production-level ML engineering. Features multi-model comparison (LSTM, GRU, XGBoost, Random Forest, Linear, Naive), financial-news sentiment analysis with FinBERT, SHAP-based explainability, and historical backtesting — all served via FastAPI with a React dashboard.

**Built for**: MCA AI/ML final year project / portfolio

---

## Technologies

`Python` · `FastAPI` · `PyTorch` · `XGBoost` · `scikit-learn` · `SHAP` · `Transformers (FinBERT)` · `React` · `TypeScript` · `Recharts` · `PostgreSQL` · `SQLAlchemy` · `Alembic` · `Docker` · `yfinance`

---

## Key Achievements (Fill After Running)

- [ ] Directional accuracy on test set: ____%
- [ ] Best model (MAE): ____
- [ ] Best model (RMSE): ____
- [ ] Sentiment improvement over market-only: ____ (yes/no and by how much)
- [ ] Backtest simulated return: ____%
- [ ] Backtest Sharpe ratio: ____
- [ ] Number of features engineered: 30+
- [ ] Training time for LSTM on 5-year AAPL: ____s

---

## Interview Talking Points

1. "I implemented strict chronological splitting to prevent look-ahead bias — a common mistake in student ML projects"
2. "FinBERT outperforms general-purpose sentiment models on financial text because it was fine-tuned on financial corpora"
3. "SHAP values give me feature attribution for individual predictions without making causal claims"
4. "The backtesting module uses rolling predictions — each day's signal only uses data available up to that point"
5. "I containerized the entire stack with Docker Compose so it runs with a single command"
