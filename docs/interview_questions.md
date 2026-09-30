# 30 Interview Questions & Answers — AI Stock Forecasting Platform

## Data & ML Fundamentals

**Q1. Why did you use a chronological train/validation/test split instead of random split?**
Stock prices follow a time series. Using random splitting would allow the model to see future prices during training (look-ahead bias/data leakage), producing optimistic metrics that don't reflect real-world performance. Chronological splitting (oldest 70% → train, next 15% → val, last 15% → test) ensures the model is only evaluated on truly unseen future data.

**Q2. What is data leakage? Give a specific example from stock forecasting.**
Data leakage is when information from the future (or test set) inadvertently flows into model training, creating unrealistically good performance. Example: fitting a StandardScaler on the entire dataset before splitting. The scaler encodes statistics (mean, std) from test-period prices into the training process — the model implicitly "knows" future values. Fix: fit the scaler ONLY on training data and apply it to val/test.

**Q3. How did you prevent look-ahead bias in your feature engineering?**
- All lag features use `shift(n)` with n ≥ 1 (so close_lag_1[i] = close[i-1], never close[i])
- Rolling windows use the current bar's data (which is fine — the target is the NEXT bar)
- The target column uses `shift(-1)` and the last row (which has NaN target) is dropped before training
- No future returns or prices are used as inputs

**Q4. What is overfitting? How did you address it in this project?**
Overfitting is when a model learns training data patterns (including noise) too precisely, performing well on train but poorly on unseen data. Mitigations used:
- Validation set for early stopping (XGBoost eval_set, LSTM ReduceLROnPlateau)
- Random Forest: out-of-bag estimation
- LSTM/GRU: Dropout layers + gradient clipping
- Report test set metrics — not just train metrics

**Q5. Why did you use Ridge Regression instead of vanilla Linear Regression?**
Ridge adds L2 regularization (penalty on coefficient magnitude), which handles multicollinearity between correlated features (e.g., multiple moving averages are highly correlated). This prevents coefficients from exploding on collinear feature sets, improving generalization.

---

## Model Architecture

**Q6. Why did you choose LSTM for time-series forecasting?**
LSTMs have gated memory cells that can selectively remember relevant long-term patterns (e.g., seasonal trends, multi-day momentum) and forget irrelevant ones. Unlike vanilla RNNs, LSTMs mitigate the vanishing gradient problem, allowing them to learn dependencies over 60+ timestep windows. They naturally process sequential data without requiring manual feature engineering of lag terms.

**Q7. Why GRU, and how does it differ from LSTM?**
GRU (Gated Recurrent Unit) replaces LSTM's 3-gate system (input, forget, output) with 2 gates (reset, update). This makes it:
- Faster to train (fewer parameters)
- Often comparable or better on shorter sequences
- Less prone to overfitting on small datasets
GRU is preferred when computational efficiency matters and the sequence is not extremely long.

**Q8. Why XGBoost for tabular financial data?**
XGBoost excels at tabular data because:
- Handles mixed feature types and scales without normalization
- Built-in regularization (L1/L2) prevents overfitting
- Fast training with parallelized tree building
- Handles missing values natively
- eval_set enables early stopping based on validation loss
- Often outperforms deep learning on tabular data with limited samples

**Q9. What lookback window did you use for LSTM/GRU and why?**
Default: 60 trading periods (approximately 3 calendar months). This captures short-to-medium term trends and seasonal patterns while avoiding:
- Excessive noise from very short windows
- Computational cost and vanishing gradients from very long windows (250+)
The value is configurable via the training API.

**Q10. What is the naive baseline model and why include it?**
The naive model predicts tomorrow's price = today's price. It requires no training and serves as a minimum performance bar. Any trained model should beat this baseline. If your LSTM doesn't outperform naive, the model has no predictive value. It's essential for calibrating results honestly.

---

## Metrics

**Q11. Why use MAE for regression evaluation?**
MAE (Mean Absolute Error) is the average absolute difference between predictions and actual prices. It's:
- Interpretable: "on average, my predictions are off by $X"
- Robust to outliers (unlike MSE/RMSE which square errors)
- In the same unit as the target (dollars)

**Q12. Why report RMSE as well as MAE?**
RMSE penalizes large errors disproportionately (by squaring residuals before averaging). A model with lower RMSE but higher MAE has fewer catastrophic misses. Reporting both gives a fuller picture of error distribution.

**Q13. What is R² and when is it misleading for stock forecasting?**
R² measures the proportion of target variance explained by the model (1.0 = perfect, 0 = naive mean). Caution: for stock prices, a high R² can be misleading because prices are autocorrelated — a model that always predicts "yesterday's price" can achieve a high R² while providing zero actionable signal.

**Q14. What is Directional Accuracy and why is it arguably more important than MAE for trading?**
Directional Accuracy = fraction of predictions where the predicted direction (UP/DOWN) matches actual direction. For algorithmic trading, getting the direction right (even if the magnitude is off) is often what generates alpha. A model with MAE = $5 but 60% directional accuracy may be more useful than one with MAE = $1 but 51% directional accuracy.

**Q15. Why use F1 score for the classification task?**
F1 = harmonic mean of precision and recall. It's preferable to accuracy when the dataset may be slightly imbalanced (e.g., slightly more UP days than DOWN days in a bull market). F1 penalizes a model that always predicts the majority class.

---

## Sentiment Analysis

**Q16. What is FinBERT and why use it instead of VADER or TextBlob?**
FinBERT (ProsusAI/finbert) is BERT pre-trained on financial text (financial news, earnings reports, analyst reports). General-purpose models like VADER were trained on social media/reviews and misclassify domain-specific financial language. For example, "the company missed estimates by a wide margin" — VADER may score this neutrally, while FinBERT correctly identifies it as negative.

**Q17. How does the sentiment pipeline work end-to-end?**
1. Fetch news headlines/articles via NewsAPI (or labeled sample data if API unavailable)
2. Concatenate title + description for each article
3. Truncate to 512 tokens (BERT's limit)
4. FinBERT outputs probability scores for [positive, neutral, negative]
5. Compound score = positive_prob - negative_prob ∈ [-1, 1]
6. Aggregate by day by averaging compound scores
7. Daily sentiment is stored in DB and merged with market features

**Q18. How do you handle missing news data?**
The system has a clear hierarchy:
1. Try configured news provider (NewsAPI if key is set)
2. If unavailable, return pre-labeled sample headlines (is_sample=True)
3. ALWAYS display a warning banner when sample data is used
Never present sample data as real news.

---

## Backtesting

**Q19. How does the backtesting strategy work?**
A simple directional long-flat strategy:
- If model predicts UP for period i → go long (buy at close[i], exit at close[i+1])
- If model predicts DOWN → stay flat (no position, 0% return for that period)
Rolling predictions are generated day-by-day using only past data — no future information is used to generate the backtest signals.

**Q20. What is Maximum Drawdown and why does it matter?**
Max Drawdown = the largest peak-to-trough decline in portfolio equity during the backtest. It represents the worst-case scenario an investor would have experienced. Even a strategy with positive total return can be unacceptable if its max drawdown is 60% — most investors cannot emotionally or financially tolerate such losses.

**Q21. What is the Sharpe Ratio?**
Sharpe = (mean excess daily return) / (std of daily returns) × √252 (annualized). It measures return per unit of risk. A Sharpe > 1.0 is generally considered good. It accounts for volatility — a strategy returning 10% with Sharpe 0.3 is riskier than one returning 8% with Sharpe 1.5.

**Q22. Why doesn't your backtest account for transaction costs?**
This is an intentional limitation and it's clearly documented. Including realistic transaction costs (0.01–0.1% per trade, bid-ask spread, market impact) would significantly reduce or eliminate returns for most short-term strategies. The backtest is educational — a research starting point, not a live trading system.

---

## Explainability

**Q23. What is SHAP and how does it work?**
SHAP (SHapley Additive exPlanations) uses game theory (Shapley values) to assign each feature a contribution to a specific prediction. For a prediction f(x), the SHAP value for feature i represents how much that feature shifted the prediction away from the base rate. SHAP values sum to: f(x) - E[f(X)] (the difference between prediction and mean prediction).

**Q24. Why use TreeExplainer for XGBoost/Random Forest?**
TreeExplainer is exact for tree ensembles (polynomial time using tree traversal), making it much faster than KernelExplainer (which is model-agnostic but slow). For a Random Forest with 100 trees, TreeExplainer can compute SHAP values orders of magnitude faster.

**Q25. What can SHAP NOT tell you?**
SHAP explains the model's behavior, not real-world causation. A high SHAP value for "volume_ratio" means the model uses that feature heavily, not that volume actually causes price movements. Correlation in the model ≠ causation in the market. This is explicitly stated in every explanation.

---

## Architecture & Engineering

**Q26. Why FastAPI over Flask or Django?**
FastAPI provides: automatic OpenAPI docs, Pydantic v2 validation on all inputs, async support, type hints, ~3x faster than Flask for API-heavy workloads. Pydantic v2 catches invalid inputs before they reach ML code, preventing silent errors.

**Q27. Why PostgreSQL over SQLite?**
PostgreSQL supports: concurrent writes (needed for background training + API serving), proper indexing on composite keys (stock_id, date), JSON column type for model hyperparameters, connection pooling via pgBouncer if needed. SQLite is single-writer which would block during ML training.

**Q28. How does your architecture prevent the ML pipeline from blocking API requests?**
Background tasks in FastAPI (`BackgroundTasks`) allow training to run asynchronously. The client receives a run_id immediately and polls `/api/models/runs/{run_id}` for status. This prevents training (which can take minutes) from timing out HTTP connections.

---

## Limitations

**Q29. What are the fundamental limitations of ML for stock forecasting?**
1. Markets are semi-efficient — known patterns are arbitraged away quickly
2. Regime change — models trained in bull markets fail in bear markets
3. Black swan events (COVID, 2008) cannot be predicted from historical prices alone
4. Alternative data (satellite imagery, credit card data) provides edge that's unavailable to academic models
5. The model doesn't know about upcoming earnings releases, Fed announcements, or geopolitical events
6. Directional accuracy of 55% is meaningful — 51% is statistically indistinguishable from random

**Q30. What would you improve if you had more time?**
1. Walk-forward cross-validation (expanding window, 20+ validation folds)
2. Ensemble model (blend XGBoost + LSTM outputs)
3. Alternative data integration (options market, insider transactions)
4. Attention-based Transformer model (TFT — Temporal Fusion Transformer)
5. Live streaming data pipeline (WebSocket + Kafka)
6. Automated hyperparameter tuning (Optuna)
7. Production deployment with Kubernetes and model versioning (MLflow)
8. Statistical significance testing of directional accuracy (vs. random chance)
