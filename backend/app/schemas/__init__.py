"""
Pydantic v2 schemas for request/response validation.
These are separate from ORM models to maintain a clean API contract.
"""
from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ─── Stock Schemas ───────────────────────────────────────────────────────────

class StockBase(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20, description="Ticker symbol, e.g. AAPL")
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: str = "USD"
    exchange: Optional[str] = None


class StockCreate(StockBase):
    pass


class StockResponse(StockBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    market_cap: Optional[float] = None
    created_at: datetime
    updated_at: datetime


# ─── Price Schemas ────────────────────────────────────────────────────────────

class PricePoint(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None


class PriceHistoryResponse(BaseModel):
    symbol: str
    prices: list[PricePoint]
    count: int
    start_date: Optional[date] = None
    end_date: Optional[date] = None


# ─── Indicator Schemas ────────────────────────────────────────────────────────

class IndicatorPoint(BaseModel):
    date: date
    close: float
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_middle: Optional[float] = None
    volatility: Optional[float] = None


class IndicatorsResponse(BaseModel):
    symbol: str
    indicators: list[IndicatorPoint]


# ─── Data Refresh ─────────────────────────────────────────────────────────────

class DataRefreshRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    force: bool = False  # Force re-download even if data exists


class DataRefreshResponse(BaseModel):
    symbol: str
    records_added: int
    records_total: int
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    provider: str
    is_sample: bool = False  # True when fallback demo data is used


# ─── ML Model Schemas ─────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    model_type: str = Field(..., description="One of: naive, linear, random_forest, xgboost, lstm, gru")
    task: str = Field(default="regression", description="'regression' or 'classification'")
    feature_set: str = Field(default="market", description="'market' or 'market+sentiment'")
    hyperparameters: Optional[dict[str, Any]] = None
    lookback_window: int = Field(default=60, ge=10, le=250, description="Lookback periods for LSTM/GRU")

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        allowed = {"naive", "linear", "random_forest", "xgboost", "lstm", "gru"}
        if v not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return v

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        if v not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        return v


class ModelMetrics(BaseModel):
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    directional_accuracy: Optional[float] = None


class MLModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    model_type: str
    version: str
    stock_symbol: str
    task: str
    feature_set: str
    is_active: bool
    created_at: datetime
    training_start_date: Optional[date] = None
    training_end_date: Optional[date] = None


class ModelRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    model_id: int
    status: str
    train_metrics: Optional[ModelMetrics] = None
    val_metrics: Optional[ModelMetrics] = None
    test_metrics: Optional[ModelMetrics] = None
    directional_accuracy: Optional[float] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime


# ─── Prediction Schemas ───────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    model_id: int
    horizon: int = Field(default=5, ge=1, le=20, description="Number of periods to forecast")


class ForecastPoint(BaseModel):
    period: int
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    direction: Optional[str] = None  # "UP" or "DOWN" for classification


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    symbol: str
    model_id: int
    model_name: str
    task: str
    horizon: int
    forecast: list[ForecastPoint]
    model_metrics: Optional[dict] = None
    created_at: datetime


# ─── Sentiment Schemas ────────────────────────────────────────────────────────

class SentimentAnalyzeRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    days: int = Field(default=7, ge=1, le=30)


class HeadlineItem(BaseModel):
    title: str
    source: Optional[str] = None
    published_at: Optional[datetime] = None
    sentiment: str  # positive | neutral | negative
    score: float
    is_sample: bool = False


class DailySentiment(BaseModel):
    date: date
    compound_score: float
    positive: float
    neutral: float
    negative: float
    article_count: int


class SentimentResponse(BaseModel):
    symbol: str
    overall_score: float
    overall_label: str  # positive | neutral | negative
    daily_sentiment: list[DailySentiment]
    headlines: list[HeadlineItem]
    is_sample: bool = False  # Clearly marked when demo data is used


# ─── Backtest Schemas ─────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    model_id: int
    start_date: date
    end_date: date
    strategy: str = Field(default="directional", description="'directional' strategy")

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info) -> date:
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class EquityPoint(BaseModel):
    date: date
    equity: float
    cumulative_return: float


class BacktestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    symbol: str
    model_id: int
    start_date: date
    end_date: date
    strategy: str
    total_trades: int
    win_rate: Optional[float] = None
    cumulative_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    equity_curve: list[EquityPoint]
    disclaimer: str = (
        "This is an educational backtest using historical data. "
        "It does not account for transaction costs, slippage, market impact, "
        "or other real-world constraints. Past performance is not indicative of future results."
    )
    created_at: datetime


# ─── Explainability Schemas ───────────────────────────────────────────────────

class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    direction: str  # "positive" or "negative"


class ExplainResponse(BaseModel):
    prediction_id: int
    model_type: str
    top_positive_features: list[FeatureContribution]
    top_negative_features: list[FeatureContribution]
    feature_importance: dict[str, float]
    explanation_text: str
    disclaimer: str = (
        "Feature importance shows correlation in the model, not causal relationships. "
        "SHAP values explain model behavior, not real-world causation."
    )


# ─── Health Check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    database: str
    timestamp: datetime
