"""
Application configuration using Pydantic Settings.
All secrets are loaded from environment variables - never hardcoded here.
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_IS_VERCEL = bool(os.environ.get("VERCEL"))


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "AI Stock Forecasting Platform"
    VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "production", "test"] = "development"
    LOG_LEVEL: str = "INFO"

    # Security
    SECRET_KEY: str = "change-this-in-production"
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # Database
    DATABASE_URL: str = (
        "sqlite:////tmp/dev.db"
        if _IS_VERCEL
        else "sqlite:///./data/dev.db"
    )

    # Market Data
    MARKET_DATA_PROVIDER: Literal["yfinance", "alpha_vantage", "polygon"] = "yfinance"
    ALPHA_VANTAGE_API_KEY: str = ""
    POLYGON_API_KEY: str = ""

    # News / Sentiment
    NEWS_PROVIDER: Literal["newsapi", "sample"] = "sample"
    NEWS_API_KEY: str = ""

    # Paths
    MODEL_ARTIFACTS_DIR: Path = Path("/tmp/models") if _IS_VERCEL else Path("./models")
    DATA_DIR: Path = Path("/tmp/data") if _IS_VERCEL else Path("./data")

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    @property
    def raw_data_dir(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def sample_data_dir(self) -> Path:
        return self.DATA_DIR / "sample"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance - loaded once per process."""
    return Settings()
