"""
Generate sample OHLCV CSV files for AAPL, MSFT, GOOGL, TSLA.
Used as development fallback when no API keys are configured.
These are clearly labeled as SYNTHETIC data — not real market prices.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_ohlcv(symbol: str, n_days: int = 1260) -> pd.DataFrame:
    """Generate ~5 years of synthetic OHLCV data using geometric Brownian motion."""
    rng = np.random.default_rng(seed=abs(hash(symbol)) % (2**31))

    # Symbol-specific starting prices (rough 2019 levels)
    starting_prices = {"AAPL": 150, "MSFT": 130, "GOOGL": 1200, "TSLA": 200, "NVDA": 180}
    initial_price = starting_prices.get(symbol, 100)

    mu = 0.0004
    sigma = 0.018
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)

    log_returns = rng.normal(mu, sigma, n_days)
    prices = initial_price * np.exp(np.cumsum(log_returns))
    high = prices * (1 + rng.uniform(0.002, 0.025, n_days))
    low = prices * (1 - rng.uniform(0.002, 0.025, n_days))
    open_ = prices * (1 + rng.uniform(-0.008, 0.008, n_days))
    volume = rng.integers(5_000_000, 50_000_000, n_days)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": prices, "Volume": volume, "Adj_Close": prices},
        index=dates,
    )
    df.index.name = "Date"
    return df


if __name__ == "__main__":
    sample_dir = Path(__file__).parent.parent / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META"]
    for sym in symbols:
        df = generate_synthetic_ohlcv(sym)
        path = sample_dir / f"{sym}.csv"
        df.to_csv(path)
        print(f"[OK] Generated sample data: {path} ({len(df)} rows)")
    print("\n[WARNING]  These are SYNTHETIC files -- NOT real market data.")
    print("   They are labeled sample data and used only as development fallback.")
