"""
Unit tests for the backtesting engine.
"""
import numpy as np
import pandas as pd
import pytest

from app.backtesting.engine import run_backtest, _compute_max_drawdown, _compute_sharpe


def _make_price_series(n: int = 100, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.015, n)
    prices = 100 * np.cumprod(1 + returns)
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.Series(prices, index=dates, name="Close")


class TestRunBacktest:
    def test_all_long_direction(self):
        prices = _make_price_series()
        directions = ["UP"] * len(prices)
        result = run_backtest(prices, directions)
        # All-long: equity curve should roughly match buy-and-hold
        assert result["total_trades"] == len(prices) - 1
        assert 0.0 <= result["win_rate"] <= 1.0
        assert isinstance(result["cumulative_return"], float)

    def test_all_flat(self):
        prices = _make_price_series()
        directions = ["DOWN"] * len(prices)
        result = run_backtest(prices, directions)
        assert result["total_trades"] == 0
        assert result["cumulative_return"] == pytest.approx(0.0, abs=0.001)
        assert result["win_rate"] == 0.0

    def test_equity_curve_length(self):
        prices = _make_price_series(n=50)
        directions = ["UP"] * 50
        result = run_backtest(prices, directions)
        # Equity curve has one entry per trading day after the first
        assert len(result["equity_curve"]) == 49

    def test_length_mismatch_raises(self):
        prices = _make_price_series(n=10)
        with pytest.raises(ValueError, match="Length mismatch"):
            run_backtest(prices, ["UP"] * 5)  # wrong length

    def test_sharpe_reasonable(self):
        prices = _make_price_series()
        directions = ["UP"] * len(prices)
        result = run_backtest(prices, directions)
        # Sharpe ratio should be a real number
        assert isinstance(result["sharpe_ratio"], float)
        assert not np.isnan(result["sharpe_ratio"])


class TestMaxDrawdown:
    def test_flat_equity_zero_drawdown(self):
        equity = [10000.0] * 10
        dd = _compute_max_drawdown(equity, 10000.0)
        assert dd == pytest.approx(0.0)

    def test_monotone_rising_zero_drawdown(self):
        equity = [10000.0 + i * 100 for i in range(10)]
        dd = _compute_max_drawdown(equity, 10000.0)
        assert dd == pytest.approx(0.0)

    def test_known_drawdown(self):
        # Peak 12000, then 6000 → 50% drawdown
        equity = [10000, 12000, 9000, 6000]
        dd = _compute_max_drawdown(equity, 10000.0)
        assert dd == pytest.approx(50.0, abs=0.1)


class TestSharpe:
    def test_zero_std_returns_zero(self):
        assert _compute_sharpe([0.0, 0.0, 0.0]) == 0.0

    def test_positive_returns_positive_sharpe(self):
        # Use returns with positive mean AND non-zero variance so Sharpe > 0.
        # All-identical returns have zero std, which legitimately returns 0.0.
        import numpy as np
        rng = np.random.default_rng(99)
        returns = list(0.005 + rng.normal(0, 0.002, 50))  # mean 0.5%/day, small noise
        assert _compute_sharpe(returns) > 0
