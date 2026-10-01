"""
Market Data Service — Provider-agnostic interface.

Supports:
  - yfinance  (free, no API key required — default)
  - alpha_vantage (requires ALPHA_VANTAGE_API_KEY)
  - polygon   (requires POLYGON_API_KEY)

If the primary provider fails, the service falls back to sample CSV data
located in data/sample/. Sample data is ALWAYS clearly marked as such.
The provider can be swapped by changing MARKET_DATA_PROVIDER in .env.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _safe_parse_date(val: Any) -> date:
    """
    Safely parse any date representation (ISO string, YYYY-MM-DD, epoch ms/s, date, datetime).
    Handles millisecond/second numeric string timestamps (e.g. '0632015937927').
    """
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()

    if isinstance(val, (int, float)):
        unit = "ms" if val > 1e11 else "s"
        return pd.to_datetime(val, unit=unit).date()

    val_str = str(val).strip()
    if val_str.isdigit():
        num_val = int(val_str)
        unit = "ms" if num_val > 1e11 else "s"
        return pd.to_datetime(num_val, unit=unit).date()

    try:
        dt = pd.to_datetime(val_str, format="mixed", errors="coerce")
        if pd.notna(dt):
            return dt.date()
    except Exception:
        pass

    dt = pd.to_datetime(val_str, errors="coerce")
    if pd.notna(dt):
        return dt.date()

    raise ValueError(f"Unable to parse date: {val}")


def _safe_to_datetime_index(index) -> pd.DatetimeIndex:
    """
    Safely converts an index or series into a timezone-naive DatetimeIndex,
    supporting ISO dates, mixed format strings, and numeric epoch timestamps.
    """
    s = pd.Series(index)
    try:
        dt_series = pd.to_datetime(s, format="mixed", errors="coerce")
    except Exception:
        dt_series = pd.to_datetime(s, errors="coerce")

    if dt_series.isna().any():
        num_s = pd.to_numeric(s, errors="coerce")
        mask = dt_series.isna() & num_s.notna()
        if mask.any():
            ms_mask = mask & (num_s > 1e11)
            s_mask = mask & (~ms_mask)
            if ms_mask.any():
                dt_series.loc[ms_mask] = pd.to_datetime(num_s[ms_mask], unit="ms", errors="coerce")
            if s_mask.any():
                dt_series.loc[s_mask] = pd.to_datetime(num_s[s_mask], unit="s", errors="coerce")

    valid_mask = dt_series.notna()
    dti = pd.DatetimeIndex(dt_series[valid_mask])
    if getattr(dti, "tz", None) is not None:
        dti = dti.tz_localize(None)
    elif hasattr(dti, "tz_convert"):
        try:
            dti = dti.tz_convert(None)
        except Exception:
            pass
    return dti


# ─── Abstract Interface ───────────────────────────────────────────────────────

class MarketDataProvider:
    """Base class — all providers must implement fetch_ohlcv."""

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
            Date (index, tz-naive), Open, High, Low, Close, Volume, Adj_Close
        Rows are sorted chronologically (oldest first).
        """
        raise NotImplementedError

    def fetch_info(self, symbol: str) -> dict:
        """Returns basic stock metadata. Return {} if unavailable."""
        return {}


# ─── yfinance Provider ───────────────────────────────────────────────────────

class YFinanceProvider(MarketDataProvider):
    """Free market data via yfinance or direct Yahoo Finance API fallback."""

    def fetch_ohlcv(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=True,
                actions=False,
            )
            if not df.empty:
                df.index = _safe_to_datetime_index(df.index)
                df = df[["Open", "High", "Low", "Close", "Volume"]].rename(
                    columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"}
                )
                df["Adj_Close"] = df["Close"]
                df.index.name = "Date"
                return df.sort_index()
        except ImportError:
            pass
        except Exception as e:
            logger.warning("yfinance package fetch failed, falling back to direct API", symbol=symbol, error=str(e))

        # Direct Yahoo Finance API fallback using standard requests (zero extra dependencies)
        import requests
        try:
            p1 = int(datetime.combine(start, datetime.min.time()).timestamp())
            p2 = int(datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp())
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d"
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
            data = res.json()
            results = data.get("chart", {}).get("result")
            if not results:
                raise ValueError(f"No data returned for {symbol}")

            item = results[0]
            timestamps = item.get("timestamp", [])
            quotes = item.get("indicators", {}).get("quote", [{}])[0]
            dates = [datetime.fromtimestamp(ts).date() for ts in timestamps]

            df = pd.DataFrame(
                {
                    "Open": quotes.get("open", []),
                    "High": quotes.get("high", []),
                    "Low": quotes.get("low", []),
                    "Close": quotes.get("close", []),
                    "Volume": quotes.get("volume", []),
                },
                index=pd.DatetimeIndex(dates),
            )
            df["Adj_Close"] = df["Close"]
            df.index.name = "Date"
            df = df.dropna().sort_index()
            if df.empty:
                raise ValueError(f"Empty data for {symbol}")
            return df
        except Exception as exc:
            raise ValueError(f"Failed to fetch market data for {symbol}: {exc}")

    def fetch_info(self, symbol: str) -> dict:
        try:
            import yfinance as yf
            info = yf.Ticker(symbol).info
            return {
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange"),
            }
        except Exception:
            pass

        try:
            import requests
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=1d"
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            meta = res.json()["chart"]["result"][0]["meta"]
            return {
                "name": meta.get("shortName") or meta.get("symbol"),
                "currency": meta.get("currency", "USD"),
                "exchange": meta.get("exchangeName"),
            }
        except Exception as exc:
            logger.warning("yfinance info fetch failed", symbol=symbol, error=str(exc))
            return {}


# ─── Alpha Vantage Provider ──────────────────────────────────────────────────

class AlphaVantageProvider(MarketDataProvider):
    """Alpha Vantage daily OHLCV. Requires ALPHA_VANTAGE_API_KEY."""

    BASE_URL = "https://www.alphavantage.co/query"

    def fetch_ohlcv(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        import requests

        api_key = settings.ALPHA_VANTAGE_API_KEY
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is not set")

        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        ts_key = "Time Series (Daily)"
        if ts_key not in data:
            raise ValueError(f"Alpha Vantage error for {symbol}: {data.get('Note') or data.get('Information') or data}")

        records = []
        for date_str, vals in data[ts_key].items():
            d = _safe_parse_date(date_str)
            if start <= d <= end:
                records.append({
                    "Date": d,
                    "Open": float(vals["1. open"]),
                    "High": float(vals["2. high"]),
                    "Low": float(vals["3. low"]),
                    "Close": float(vals["4. close"]),
                    "Adj_Close": float(vals["5. adjusted close"]),
                    "Volume": int(vals["6. volume"]),
                })

        if not records:
            raise ValueError(f"No data in range {start}–{end} for {symbol}")

        df = pd.DataFrame(records).set_index("Date").sort_index()
        return df


# ─── Polygon.io Provider ─────────────────────────────────────────────────────

class PolygonProvider(MarketDataProvider):
    """Polygon.io aggregates endpoint. Requires POLYGON_API_KEY."""

    BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

    def fetch_ohlcv(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        import requests

        api_key = settings.POLYGON_API_KEY
        if not api_key:
            raise ValueError("POLYGON_API_KEY is not set")

        url = f"{self.BASE_URL}/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        resp = requests.get(url, params={"apiKey": api_key, "limit": 50000}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") not in ("OK", "DELAYED"):
            raise ValueError(f"Polygon error for {symbol}: {data.get('error') or data}")

        results = data.get("results", [])
        if not results:
            raise ValueError(f"No data returned from Polygon for {symbol}")

        records = [
            {
                "Date": _safe_parse_date(r["t"]),
                "Open": r["o"],
                "High": r["h"],
                "Low": r["l"],
                "Close": r["c"],
                "Volume": int(r["v"]),
                "Adj_Close": r["c"],
            }
            for r in results
        ]
        df = pd.DataFrame(records).set_index("Date").sort_index()
        return df


# ─── Sample / Fallback Provider ──────────────────────────────────────────────

class SampleDataProvider(MarketDataProvider):
    """
    Reads pre-generated CSV files from data/sample/.
    ALWAYS returns is_sample=True so the UI can display a clear notice.
    This provider is only used when real providers fail or are not configured.
    """

    def fetch_ohlcv(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        sample_path = settings.sample_data_dir / f"{symbol.upper()}.csv"
        if not sample_path.exists():
            # Generate synthetic sample data if CSV not found
            return _generate_synthetic_ohlcv(symbol, start, end)

        df = pd.read_csv(sample_path, index_col="Date")
        df.index = _safe_to_datetime_index(df.index)
        df = df.sort_index()
        mask = (df.index.date >= start) & (df.index.date <= end)
        return df[mask]


def _generate_synthetic_ohlcv(symbol: str, start: date, end: date) -> pd.DataFrame:
    """
    Generates a realistic-looking synthetic OHLCV time series using
    geometric Brownian motion. This is NOT real market data.
    Used ONLY as a development fallback.
    """
    import numpy as np

    dates = pd.bdate_range(start=start, end=end)  # business days only
    n = len(dates)
    if n == 0:
        raise ValueError(f"No business days in range {start}–{end}")

    rng = np.random.default_rng(seed=abs(hash(symbol)) % (2**31))
    mu = 0.0005  # daily drift
    sigma = 0.015  # daily volatility
    initial_price = 150.0

    log_returns = rng.normal(mu, sigma, n)
    prices = initial_price * np.exp(np.cumsum(log_returns))

    high = prices * (1 + rng.uniform(0.001, 0.03, n))
    low = prices * (1 - rng.uniform(0.001, 0.03, n))
    open_ = prices * (1 + rng.uniform(-0.01, 0.01, n))
    volume = rng.integers(1_000_000, 10_000_000, n)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": prices,
            "Volume": volume,
            "Adj_Close": prices,
        },
        index=dates,
    )
    df.index.name = "Date"
    logger.warning(
        "Using SYNTHETIC sample data — NOT real market data",
        symbol=symbol,
        start=str(start),
        end=str(end),
    )
    return df


# ─── Factory ─────────────────────────────────────────────────────────────────

def get_market_data_provider() -> MarketDataProvider:
    """
    Returns the configured provider. Falls back to SampleDataProvider
    if the configured provider raises during initialization.
    """
    provider_name = settings.MARKET_DATA_PROVIDER
    providers = {
        "yfinance": YFinanceProvider,
        "alpha_vantage": AlphaVantageProvider,
        "polygon": PolygonProvider,
    }
    cls = providers.get(provider_name, YFinanceProvider)
    logger.info("Market data provider selected", provider=provider_name)
    return cls()


# ─── High-level Service ───────────────────────────────────────────────────────

class MarketDataService:
    """
    Orchestrates data fetching, caching in the database, and fallback logic.
    """

    def __init__(self):
        self.provider = get_market_data_provider()
        self._ensure_dirs()

    def _ensure_dirs(self):
        settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        settings.sample_data_dir.mkdir(parents=True, exist_ok=True)
        settings.MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> tuple[pd.DataFrame, bool]:
        """
        Returns (DataFrame, is_sample).
        is_sample=True means the data is synthetic/demo — not real market data.
        """
        if start is None:
            start = date.today() - timedelta(days=5 * 365)
        if end is None:
            end = date.today()

        symbol = symbol.upper()
        logger.info("Fetching market data", symbol=symbol, start=str(start), end=str(end))

        # Check for cached raw data
        cache_path = settings.raw_data_dir / f"{symbol}.csv"
        if cache_path.exists():
            cached_df = pd.read_csv(cache_path, index_col="Date")
            cached_df.index = _safe_to_datetime_index(cached_df.index)
            # Drop NaT rows — newer yfinance/pandas can produce them after tz_localize(None)
            cached_df = cached_df[cached_df.index.notna()]
            if cached_df.empty:
                cache_path.unlink(missing_ok=True)  # corrupt cache — delete and re-fetch
            else:
                cached_start = cached_df.index.date.min()
                cached_end = cached_df.index.date.max()

                if cached_start <= start and cached_end >= end:
                    logger.info("Cache hit — returning from disk", symbol=symbol)
                    mask = (cached_df.index.date >= start) & (cached_df.index.date <= end)
                    return cached_df[mask], False

        # Try configured provider
        try:
            df = self.provider.fetch_ohlcv(symbol, start, end)
            self._validate(df, symbol)
            df = self._clean(df)
            # Save to cache
            df.to_csv(cache_path)
            logger.info("Data fetched and cached", symbol=symbol, rows=len(df))
            return df, False
        except Exception as exc:
            logger.error(
                "Primary provider failed — falling back to sample data",
                symbol=symbol,
                error=str(exc),
            )
            fallback = SampleDataProvider()
            df = fallback.fetch_ohlcv(symbol, start, end)
            df = self._clean(df)
            return df, True

    def _validate(self, df: pd.DataFrame, symbol: str) -> None:
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} for {symbol}")
        if df.empty:
            raise ValueError(f"Empty DataFrame for {symbol}")
        if df.index.duplicated().any():
            raise ValueError(f"Duplicate dates in data for {symbol}")

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index()
        # Forward-fill small gaps (e.g., missing days due to holidays)
        df = df.ffill(limit=3)
        # Drop rows where Close is NaN or zero (invalid)
        df = df[df["Close"].notna() & (df["Close"] > 0)]
        df.index = _safe_to_datetime_index(df.index)
        return df

    def get_stock_info(self, symbol: str) -> dict:
        try:
            return self.provider.fetch_info(symbol)
        except Exception as exc:
            logger.warning("Failed to fetch stock info", symbol=symbol, error=str(exc))
            return {}
