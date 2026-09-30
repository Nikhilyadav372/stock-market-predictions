"""
Feature Engineering Demonstration
Shows all features computed by the pipeline and validates leakage prevention.
"""
import sys
import os

# FIX 1: Point to backend/ so that `app` package is importable,
# regardless of which directory you run the script from.
_notebooks_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.join(_notebooks_dir, '..', 'backend')
sys.path.insert(0, _backend_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

from app.services.market_data import MarketDataService
from app.ml.features import build_features, get_feature_columns, split_time_series

# Load data
svc = MarketDataService()
df, is_sample = svc.fetch("AAPL")
print(f"Raw data: {len(df)} rows {'(SAMPLE)' if is_sample else ''}")

# Build features
feat = build_features(df)
feature_cols = get_feature_columns()
print(f"\nFeature matrix: {feat.shape}")
print(f"Number of features: {len(feature_cols)}")
print("\nFeature columns:")
for col in sorted(feature_cols):
    print(f"  - {col}")

# Show target definition
print("\n=== TARGET DEFINITION ===")
print("Regression target: target_close = Close[i+1] (next period close)")
print("Classification target: target_direction = 1 if Close[i+1] > Close[i] else 0")
print(f"\nTarget distribution (direction):")
print(feat['target_direction'].value_counts())

# Verify no leakage in lag features
print("\n=== LEAKAGE VERIFICATION ===")
sample = feat[['close', 'close_lag_1', 'target_close']].dropna().head(5)
print("close_lag_1[i] should equal close[i-1]:")
print(sample)

# FIX 2: Use pd.testing.assert_series_equal instead of == on floats.
# The raw == operator raises a ValueError when the two Series have
# different lengths after dropna(), and silently fails for NaN values.
expected_lag1 = df['Close'].shift(1).reindex(feat.index).rename('close_lag_1')
pd.testing.assert_series_equal(
    feat['close_lag_1'].dropna(),
    expected_lag1.dropna(),
    check_names=False,
    rtol=1e-5,
)
print("[OK] No leakage in lag features")

# Chronological split
train, val, test = split_time_series(feat)
print(f"\n=== CHRONOLOGICAL SPLIT ===")
print(f"Train: {train.index[0].date()} -> {train.index[-1].date()} ({len(train)} rows)")
print(f"Val:   {val.index[0].date()} -> {val.index[-1].date()} ({len(val)} rows)")
print(f"Test:  {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} rows)")

# Plot a few key features
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

axes[0, 0].plot(feat.index[-200:], feat['rsi'].iloc[-200:], color='#f59e0b')
axes[0, 0].axhline(70, color='#ef4444', linestyle='--', alpha=0.7)
axes[0, 0].axhline(30, color='#22c55e', linestyle='--', alpha=0.7)
axes[0, 0].set_title('RSI (14)', fontweight='bold')

axes[0, 1].plot(feat.index[-200:], feat['macd'].iloc[-200:], color='#818cf8', label='MACD')
axes[0, 1].plot(feat.index[-200:], feat['macd_signal'].iloc[-200:], color='#f59e0b', label='Signal')
axes[0, 1].legend()
axes[0, 1].set_title('MACD', fontweight='bold')

axes[1, 0].plot(feat.index[-200:], feat['close'].iloc[-200:], color='#e2e8f0', label='Close')
axes[1, 0].plot(feat.index[-200:], feat['sma_20'].iloc[-200:], color='#6366f1', label='SMA20')
axes[1, 0].plot(feat.index[-200:], feat['bb_upper'].iloc[-200:], color='#22c55e', alpha=0.5, label='BB Upper')
axes[1, 0].plot(feat.index[-200:], feat['bb_lower'].iloc[-200:], color='#ef4444', alpha=0.5, label='BB Lower')
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_title('Bollinger Bands', fontweight='bold')

axes[1, 1].plot(feat.index[-200:], feat['volume_ratio'].iloc[-200:], color='#a78bfa')
axes[1, 1].axhline(1, color='white', linestyle='--', alpha=0.3)
axes[1, 1].set_title('Volume Ratio (vs 20d MA)', fontweight='bold')

axes[2, 0].plot(feat.index[-200:], feat['volatility_20'].iloc[-200:], color='#f59e0b')
axes[2, 0].set_title('20-Day Rolling Volatility', fontweight='bold')

axes[2, 1].plot(feat.index[-200:], feat['hist_volatility'].iloc[-200:], color='#ef4444')
axes[2, 1].set_title('Annualized Historical Volatility', fontweight='bold')

for ax in axes.flatten():
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#0f0f1a')

plt.tight_layout()

# FIX 3 & 4: Resolve output path relative to this script's own directory,
# not the current working directory. Also ensure the folder exists.
_out_dir = os.path.join(_notebooks_dir, '..', 'data', 'processed')
os.makedirs(_out_dir, exist_ok=True)
_out_path = os.path.normpath(os.path.join(_out_dir, '02_features.png'))
plt.savefig(_out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {_out_path}")
