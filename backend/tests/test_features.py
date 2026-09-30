"""
Unit tests for the feature engineering pipeline.
These tests verify data-leakage protections and correct indicator computation.
"""
import numpy as np
import pandas as pd
import pytest

from app.ml.features import (
    build_features,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    get_feature_columns,
    split_time_series,
)


def _make_price_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    prices = np.abs(prices) + 10  # ensure positive
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": prices * (1 + rng.uniform(-0.01, 0.01, n)),
            "High": prices * (1 + rng.uniform(0.005, 0.02, n)),
            "Low": prices * (1 - rng.uniform(0.005, 0.02, n)),
            "Close": prices,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


class TestComputeRSI:
    def test_rsi_range(self):
        df = _make_price_df()
        rsi = compute_rsi(df["Close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), "RSI must be in [0, 100]"

    def test_rsi_no_future(self):
        """RSI at index i must not use any data after index i."""
        df = _make_price_df()
        rsi_full = compute_rsi(df["Close"])
        # Truncate to first 200 rows and recompute
        rsi_partial = compute_rsi(df["Close"].iloc[:200])
        # The first 200 values must match
        pd.testing.assert_series_equal(
            rsi_full.iloc[:200].dropna(),
            rsi_partial.dropna(),
            check_names=False,
            rtol=1e-10,
        )


class TestComputeMACD:
    def test_macd_shape(self):
        df = _make_price_df()
        macd, signal, hist = compute_macd(df["Close"])
        assert len(macd) == len(df)
        assert len(signal) == len(df)
        assert len(hist) == len(df)

    def test_histogram_equals_macd_minus_signal(self):
        df = _make_price_df()
        macd, signal, hist = compute_macd(df["Close"])
        expected_hist = macd - signal
        pd.testing.assert_series_equal(hist, expected_hist, check_names=False, rtol=1e-10)


class TestBollingerBands:
    def test_upper_gt_lower(self):
        df = _make_price_df()
        upper, middle, lower = compute_bollinger_bands(df["Close"])
        valid = upper.dropna().index
        assert (upper[valid] >= lower[valid]).all(), "Upper band must be >= lower band"

    def test_middle_is_sma(self):
        df = _make_price_df()
        _, middle, _ = compute_bollinger_bands(df["Close"], window=20)
        sma = df["Close"].rolling(20).mean()
        pd.testing.assert_series_equal(middle, sma, check_names=False)


class TestBuildFeatures:
    def test_no_future_in_lag_features(self):
        """Close_lag_1[i] must equal Close[i-1]."""
        df = _make_price_df()
        feat = build_features(df)
        shifted = df["Close"].shift(1)
        common_idx = feat.index.intersection(shifted.dropna().index)
        pd.testing.assert_series_equal(
            feat.loc[common_idx, "close_lag_1"],
            shifted.loc[common_idx],
            check_names=False,
        )

    def test_target_is_next_close(self):
        """target_close[i] must equal Close[i+1]."""
        df = _make_price_df()
        feat = build_features(df)
        next_close = df["Close"].shift(-1)
        common_idx = feat.index.intersection(next_close.dropna().index)
        pd.testing.assert_series_equal(
            feat.loc[common_idx, "target_close"],
            next_close.loc[common_idx],
            check_names=False,
        )

    def test_feature_cols_are_subset(self):
        df = _make_price_df()
        feat = build_features(df)
        feature_cols = get_feature_columns()
        for col in feature_cols:
            assert col in feat.columns, f"Feature column {col} missing from feature matrix"

    def test_no_features_use_future_returns(self):
        """daily_return[i] must equal (Close[i] - Close[i-1]) / Close[i-1]."""
        df = _make_price_df()
        feat = build_features(df)
        expected = df["Close"].pct_change()
        common_idx = feat.index.intersection(expected.dropna().index)
        pd.testing.assert_series_equal(
            feat.loc[common_idx, "daily_return"],
            expected.loc[common_idx],
            check_names=False,
            rtol=1e-10,
        )


class TestSplit:
    def test_chronological_order(self):
        df = _make_price_df()
        feat = build_features(df)
        train, val, test = split_time_series(feat)

        # Verify no overlap
        assert train.index.max() < val.index.min(), "Train must end before val starts"
        assert val.index.max() < test.index.min(), "Val must end before test starts"

    def test_split_fractions(self):
        df = _make_price_df(n=1000)
        feat = build_features(df)
        train, val, test = split_time_series(feat, train_frac=0.70, val_frac=0.15)
        total = len(train) + len(val) + len(test)
        assert total == len(feat)
        assert abs(len(train) / total - 0.70) < 0.01
        assert abs(len(val) / total - 0.15) < 0.02

    def test_no_random_shuffle(self):
        """Running split twice must produce identical results."""
        df = _make_price_df()
        feat = build_features(df)
        t1, v1, te1 = split_time_series(feat)
        t2, v2, te2 = split_time_series(feat)
        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_frame_equal(v1, v2)
        pd.testing.assert_frame_equal(te1, te2)
