"""
Data exploration utility — run as script or copy cells into Jupyter.
Demonstrates how to load and visualize the raw market data.
"""
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

# Load market data using the same service as the application
from app.services.market_data import MarketDataService

svc = MarketDataService()
symbol = "AAPL"
start = date.today() - timedelta(days=5 * 365)
df, is_sample = svc.fetch(symbol, start, date.today())

print(f"Symbol: {symbol}")
print(f"Is sample data: {is_sample}")
print(f"Rows: {len(df)}")
print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDescriptive statistics:")
print(df.describe())

# Basic visualizations
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

axes[0].plot(df.index, df['Close'], color='#6366f1', linewidth=1.5)
axes[0].set_title(f'{symbol} Closing Price', fontweight='bold')
axes[0].set_ylabel('Price ($)')
axes[0].grid(True, alpha=0.3)

axes[1].bar(df.index, df['Volume'], color='#818cf8', alpha=0.7)
axes[1].set_title('Volume')
axes[1].set_ylabel('Volume')
axes[1].grid(True, alpha=0.3)

returns = df['Close'].pct_change().dropna()
axes[2].hist(returns, bins=50, color='#6366f1', alpha=0.7, edgecolor='white')
axes[2].set_title('Daily Return Distribution')
axes[2].set_xlabel('Daily Return')
axes[2].set_ylabel('Frequency')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data/processed/01_data_exploration.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to data/processed/01_data_exploration.png")

if is_sample:
    print("\n⚠️  WARNING: Using synthetic sample data — not real market data")
    print("   Set MARKET_DATA_PROVIDER=yfinance in .env and run again for real data")
