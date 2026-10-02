"""
Feature Engineering Pipeline

All features are computed using ONLY past and present information
to prevent look-ahead bias (data leakage).

Key rules enforced here:
  1. Rolling calculations use closed='left' or shift(1) so the current
     bar's own value does not contaminate the window — EXCEPT for same-bar
     technical indicators that are standard (RSI, Bollinger Bands use
     the close of the SAME bar, which is fine because the target is the
     NEXT bar's close).
  2. Lag features explicitly shift by at least 1.
  3. Scalers are fitted ONLY on training data — never the full dataset.
     See ml/train.py for scaler handling.
  4. The target column (next-bar close or direction) is created by
     shifting close by -1, so the last row always has NaN target and
     must be dropped before training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series, window: int = 20, n_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = middle + n_std * std
    lower = middle - n_std * std
    return upper, middle, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def build_features(df: pd.DataFrame, include_sentiment: bool = False) -> pd.DataFrame:
    """
    Build the complete feature matrix from raw OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV with columns [Open, High, Low, Close, Volume].
        Index must be a DatetimeIndex sorted chronologically.
    include_sentiment : bool
        If True, a 'sentiment_score' column is expected in df and will
        be included in the feature set.

    Returns
    -------
    pd.DataFrame
        Feature matrix. Rows with NaN features at the start are dropped.
        The last row has NaN targets (used only for live inference).
    """
    f = pd.DataFrame(index=df.index)

    # ── Raw OHLCV ─────────────────────────────────────────────────────────────
    f["close"] = df["Close"]
    f["open"] = df["Open"]
    f["high"] = df["High"]
    f["low"] = df["Low"]
    f["volume"] = df["Volume"]

    # ── Returns ───────────────────────────────────────────────────────────────
    f["daily_return"] = df["Close"].pct_change()
    f["weekly_return"] = df["Close"].pct_change(5)
    f["monthly_return"] = df["Close"].pct_change(20)

    # ── Moving Averages ───────────────────────────────────────────────────────
    for window in [5, 10, 20, 50]:
        f[f"sma_{window}"] = df["Close"].rolling(window).mean()
        f[f"close_to_sma_{window}"] = df["Close"] / f[f"sma_{window}"] - 1

    f["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    f["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    f["ema_12_to_ema_26"] = f["ema_12"] / f["ema_26"] - 1

    # ── Technical Indicators ──────────────────────────────────────────────────
    f["rsi"] = compute_rsi(df["Close"], window=14)
    macd, macd_sig, macd_hist = compute_macd(df["Close"])
    f["macd"] = macd
    f["macd_signal"] = macd_sig
    f["macd_hist"] = macd_hist

    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(df["Close"])
    f["bb_upper"] = bb_upper
    f["bb_middle"] = bb_middle
    f["bb_lower"] = bb_lower
    f["bb_width"] = (bb_upper - bb_lower) / bb_middle  # Bollinger Band width
    f["bb_pct"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower)  # %B

    f["atr"] = compute_atr(df["High"], df["Low"], df["Close"])

    # ── Volatility ────────────────────────────────────────────────────────────
    f["volatility_5"] = df["Close"].pct_change().rolling(5).std()
    f["volatility_20"] = df["Close"].pct_change().rolling(20).std()
    f["hist_volatility"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)

    # ── Lag Features (shift(n) ensures no look-ahead) ─────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        f[f"close_lag_{lag}"] = df["Close"].shift(lag)
        f[f"return_lag_{lag}"] = f["daily_return"].shift(lag)

    # ── Volume Features ───────────────────────────────────────────────────────
    f["volume_change"] = df["Volume"].pct_change()
    f["volume_sma_20"] = df["Volume"].rolling(20).mean()
    f["volume_ratio"] = df["Volume"] / f["volume_sma_20"]

    # ── Price Position ────────────────────────────────────────────────────────
    f["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
    f["open_close_range"] = (df["Close"] - df["Open"]) / df["Open"]

    # ── 52-Week High/Low (uses up to 252 trading days) ─────────────────────────
    f["52w_high"] = df["High"].rolling(252, min_periods=20).max()
    f["52w_low"] = df["Low"].rolling(252, min_periods=20).min()
    f["pct_from_52w_high"] = df["Close"] / f["52w_high"] - 1
    f["pct_from_52w_low"] = df["Close"] / f["52w_low"] - 1

    # ── Sentiment (optional) ──────────────────────────────────────────────────
    if include_sentiment and "sentiment_score" in df.columns:
        f["sentiment_score"] = df["sentiment_score"]
        f["sentiment_lag1"] = df["sentiment_score"].shift(1)
        f["sentiment_rolling3"] = df["sentiment_score"].rolling(3).mean()

    # ── Targets ───────────────────────────────────────────────────────────────
    # Regression target: next-period close (shifted by -1)
    # IMPORTANT: this creates NaN in the last row — drop before training
    f["target_close"] = df["Close"].shift(-1)
    f["target_return"] = df["Close"].pct_change().shift(-1)
    # Classification target: 1 if next close > current close, else 0 (NaN on last row)
    f["target_direction"] = np.where(
        f["target_close"].isna(),
        np.nan,
        (f["target_close"] > df["Close"]).astype(float),
    )

    # Drop the first rows where rolling features are not yet available
    f = f.dropna(subset=[c for c in f.columns if c not in ("target_close", "target_return", "target_direction")])

    return f


def get_feature_columns(include_sentiment: bool = False) -> list[str]:
    """Return the list of input feature column names (excludes targets)."""
    targets = {"target_close", "target_return", "target_direction"}
    dummy = build_features(
        pd.DataFrame(
            {
                "Open": [100.0] * 300,
                "High": [105.0] * 300,
                "Low": [95.0] * 300,
                "Close": [102.0] * 300,
                "Volume": [1_000_000] * 300,
            },
            index=pd.bdate_range("2020-01-01", periods=300),
        )
    )
    cols = [c for c in dummy.columns if c not in targets]
    if not include_sentiment:
        cols = [c for c in cols if not c.startswith("sentiment")]
    return cols


def split_time_series(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/validation/test split.

    WHY NO RANDOM SPLIT:
    Stock returns are ordered in time. Random splitting would allow the model
    to "see" future data during training (look-ahead bias / data leakage),
    inflating performance metrics and making the model appear better than it is.

    The split is strictly chronological:
        train : oldest  train_frac
        val   : next    val_frac
        test  : latest  (1 - train_frac - val_frac)
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test
