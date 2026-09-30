"""
SQLAlchemy ORM models.
All tables use explicit column types and indexes for query performance.
"""
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Stock(Base):
    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(200))
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    market_cap: Mapped[Optional[float]] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(10), default="USD")
    exchange: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    prices: Mapped[list["HistoricalPrice"]] = relationship("HistoricalPrice", back_populates="stock")
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="stock")
    sentiment_scores: Mapped[list["SentimentScore"]] = relationship("SentimentScore", back_populates="stock")
    news_articles: Mapped[list["NewsArticle"]] = relationship("NewsArticle", back_populates="stock")


class HistoricalPrice(Base):
    __tablename__ = "historical_prices"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_stock_date"),
        Index("ix_historical_prices_stock_date", "stock_id", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(BigInteger)
    adj_close: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    stock: Mapped["Stock"] = relationship("Stock", back_populates="prices")


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g. "xgboost", "lstm"
    version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    stock_symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    task: Mapped[str] = mapped_column(String(20), nullable=False)  # "regression" or "classification"
    feature_set: Mapped[str] = mapped_column(String(50), default="market")  # "market" or "market+sentiment"
    artifact_path: Mapped[Optional[str]] = mapped_column(String(500))
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON)
    feature_names: Mapped[Optional[list]] = mapped_column(JSON)
    training_start_date: Mapped[Optional[date]] = mapped_column(Date)
    training_end_date: Mapped[Optional[date]] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    model_runs: Mapped[list["ModelRun"]] = relationship("ModelRun", back_populates="model")
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="model")


class ModelRun(Base):
    """Records each training experiment with full metrics for reproducibility."""
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("ml_models.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, running, done, failed
    train_mae: Mapped[Optional[float]] = mapped_column(Float)
    train_rmse: Mapped[Optional[float]] = mapped_column(Float)
    train_r2: Mapped[Optional[float]] = mapped_column(Float)
    val_mae: Mapped[Optional[float]] = mapped_column(Float)
    val_rmse: Mapped[Optional[float]] = mapped_column(Float)
    val_r2: Mapped[Optional[float]] = mapped_column(Float)
    test_mae: Mapped[Optional[float]] = mapped_column(Float)
    test_rmse: Mapped[Optional[float]] = mapped_column(Float)
    test_r2: Mapped[Optional[float]] = mapped_column(Float)
    test_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    test_f1: Mapped[Optional[float]] = mapped_column(Float)
    test_roc_auc: Mapped[Optional[float]] = mapped_column(Float)
    directional_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    extra_metrics: Mapped[Optional[dict]] = mapped_column(JSON)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    model: Mapped["MLModel"] = relationship("MLModel", back_populates="model_runs")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_stock_created", "stock_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks.id"), nullable=False)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("ml_models.id"), nullable=False)
    task: Mapped[str] = mapped_column(String(20))
    horizon: Mapped[int] = mapped_column(Integer)  # number of periods ahead
    predicted_values: Mapped[list] = mapped_column(JSON)  # list of {date, value}
    confidence_intervals: Mapped[Optional[dict]] = mapped_column(JSON)
    model_metrics: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    stock: Mapped["Stock"] = relationship("Stock", back_populates="predictions")
    model: Mapped["MLModel"] = relationship("MLModel", back_populates="predictions")
    explanation: Mapped[Optional["ModelExplanation"]] = relationship("ModelExplanation", back_populates="prediction", uselist=False)


class NewsArticle(Base):
    __tablename__ = "news_articles"
    __table_args__ = (Index("ix_news_articles_stock_published", "stock_id", "published_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[Optional[str]] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(String(1000))
    source: Mapped[Optional[str]] = mapped_column(String(200))
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    is_sample: Mapped[bool] = mapped_column(Boolean, default=False)  # Marks demo/sample data
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    stock: Mapped["Stock"] = relationship("Stock", back_populates="news_articles")


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"
    __table_args__ = (
        UniqueConstraint("stock_id", "score_date", name="uq_sentiment_stock_date"),
        Index("ix_sentiment_stock_date", "stock_id", "score_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    stock_id: Mapped[int] = mapped_column(Integer, ForeignKey("stocks.id"), nullable=False)
    score_date: Mapped[date] = mapped_column(Date, nullable=False)
    compound_score: Mapped[float] = mapped_column(Float)  # -1 to 1
    positive: Mapped[float] = mapped_column(Float)
    neutral: Mapped[float] = mapped_column(Float)
    negative: Mapped[float] = mapped_column(Float)
    article_count: Mapped[int] = mapped_column(Integer, default=0)
    is_sample: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    stock: Mapped["Stock"] = relationship("Stock", back_populates="sentiment_scores")


class Backtest(Base):
    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    stock_symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("ml_models.id"), nullable=False)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    strategy: Mapped[str] = mapped_column(String(50), default="directional")
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[Optional[float]] = mapped_column(Float)
    cumulative_return: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    equity_curve: Mapped[Optional[list]] = mapped_column(JSON)
    trade_log: Mapped[Optional[list]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ModelExplanation(Base):
    __tablename__ = "model_explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("predictions.id"), unique=True)
    shap_values: Mapped[Optional[dict]] = mapped_column(JSON)  # feature -> shap value
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSON)  # feature -> importance
    top_positive_features: Mapped[Optional[list]] = mapped_column(JSON)
    top_negative_features: Mapped[Optional[list]] = mapped_column(JSON)
    explanation_text: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="explanation")
