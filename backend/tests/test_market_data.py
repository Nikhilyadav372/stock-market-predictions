"""
Unit tests for the market data service.
Tests validation, cleaning, and fallback behavior.
"""
import pandas as pd
import numpy as np
import pytest
from datetime import date

from app.services.market_data import MarketDataService, _generate_synthetic_ohlcv, SampleDataProvider


class TestSyntheticDataGeneration:
    def test_returns_dataframe(self):
        start = date(2023, 1, 1)
        end = date(2023, 6, 30)
        df = _generate_synthetic_ohlcv("AAPL", start, end)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self):
        df = _generate_synthetic_ohlcv("AAPL", date(2023, 1, 1), date(2023, 6, 30))
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns

    def test_prices_are_positive(self):
        df = _generate_synthetic_ohlcv("TEST", date(2023, 1, 1), date(2023, 12, 31))
        assert (df["Close"] > 0).all()
        assert (df["Open"] > 0).all()
        assert (df["High"] > 0).all()
        assert (df["Low"] > 0).all()

    def test_high_gte_low(self):
        df = _generate_synthetic_ohlcv("TSLA", date(2023, 1, 1), date(2023, 6, 30))
        assert (df["High"] >= df["Low"]).all()

    def test_chronological_order(self):
        df = _generate_synthetic_ohlcv("MSFT", date(2022, 1, 1), date(2023, 1, 1))
        assert df.index.is_monotonic_increasing

    def test_deterministic_per_symbol(self):
        """Same symbol should produce the same synthetic data (seeded by symbol hash)."""
        df1 = _generate_synthetic_ohlcv("AAPL", date(2023, 1, 1), date(2023, 6, 30))
        df2 = _generate_synthetic_ohlcv("AAPL", date(2023, 1, 1), date(2023, 6, 30))
        pd.testing.assert_frame_equal(df1, df2)


class TestMarketDataServiceCleaning:
    def _make_df(self) -> pd.DataFrame:
        dates = pd.bdate_range("2023-01-01", periods=10)
        return pd.DataFrame(
            {
                "Open": [100.0] * 10,
                "High": [105.0] * 10,
                "Low": [95.0] * 10,
                "Close": [102.0] * 10,
                "Volume": [1_000_000] * 10,
                "Adj_Close": [102.0] * 10,
            },
            index=dates,
        )

    def test_clean_drops_zero_close(self):
        svc = MarketDataService.__new__(MarketDataService)
        df = self._make_df()
        df.loc[df.index[5], "Close"] = 0.0
        cleaned = svc._clean(df)
        assert 0.0 not in cleaned["Close"].values

    def test_clean_sorts_chronologically(self):
        svc = MarketDataService.__new__(MarketDataService)
        df = self._make_df()
        # Reverse the order
        df = df.iloc[::-1]
        cleaned = svc._clean(df)
        assert cleaned.index.is_monotonic_increasing
