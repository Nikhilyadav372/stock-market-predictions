# Data Leakage Prevention Guide

## What is Data Leakage?

Data leakage occurs when information that would not be available at prediction time
is used during model training. For time-series data like stock prices, this is
especially insidious because it produces dramatically inflated metrics that
completely vanish in production.

## Forms of Leakage in Stock Forecasting

### 1. Random Train/Test Split (Most Common Mistake)

❌ **Wrong:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
This randomly mixes future and past data — the model sees 2024 prices during training
on 2020 data.

✅ **Correct (this project):**
```python
# features.py — split_time_series()
train = df.iloc[:int(n * 0.70)]    # oldest 70%
val   = df.iloc[int(n * 0.70):int(n * 0.85)]
test  = df.iloc[int(n * 0.85):]   # newest 15%
```

### 2. Scaler Fitted on Full Dataset

❌ **Wrong:**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # uses future mean/std
X_train, X_test = X_scaled[:split], X_scaled[split:]
```

✅ **Correct (this project — train.py):**
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols])  # fit ONLY on train
X_val   = scaler.transform(val[feature_cols])        # transform only
X_test  = scaler.transform(test[feature_cols])       # transform only
```

### 3. Future-Looking Lag Features

❌ **Wrong:**
```python
df['next_close'] = df['Close'].shift(-1)  # ← future leakage if used as feature!
```

✅ **Correct (this project — features.py):**
```python
df['close_lag_1'] = df['Close'].shift(1)  # only past data
df['target_close'] = df['Close'].shift(-1)  # only used as TARGET, never as feature
```

### 4. Target Leakage

❌ **Wrong:** Including the target variable (or a transformation of it) as a feature.
Example: including tomorrow's return in the feature matrix.

✅ **Correct:** Target column is computed as `shift(-1)` and separated from features
before any scaling or model fitting.

### 5. Rolling Window Look-Ahead

❌ **Wrong:**
```python
df['rolling_mean'] = df['Close'].rolling(20, center=True).mean()
# center=True means the window is centered — uses future values!
```

✅ **Correct (this project):**
```python
df['sma_20'] = df['Close'].rolling(20).mean()
# Default: trailing window — only uses current and past values
```

### 6. LSTM Sequence Leakage

❌ **Wrong:** Building sequences BEFORE splitting, then splitting sequences.
This can create overlapping windows that span the split boundary.

✅ **Correct (this project — train.py):**
```python
# Step 1: Split features chronologically FIRST
X_train, X_val, X_test = scaler.transform(...)

# Step 2: Build sequences AFTER splitting
Xs_train, ys_train = build_sequences(X_train, y_train, lookback)
Xs_val, ys_val = build_sequences(X_val, y_val, lookback)
```

### 7. Backtesting Leakage

❌ **Wrong:** Using a model trained on the FULL dataset to backtest on a subset of that data.

✅ **Correct (this project — backtest.py):**
Rolling prediction: for each day d in the backtest period, we use only data
available up to day d to generate the prediction.
```python
for i in range(len(backtest_df)):
    available_data = df[df.index <= backtest_df.index[i]]
    predictions.append(predictor.predict_next_n(available_data, n=1))
```

## Audit Checklist

Before considering the pipeline complete, verify:

- [ ] `train_test_split` is NOT used anywhere in the main pipeline
- [ ] All scalers are fitted ONLY on training data
- [ ] No feature uses `shift(-n)` or `future` data
- [ ] Sequences for LSTM/GRU are built after the chronological split
- [ ] The target column is NOT in the feature column list
- [ ] Backtesting uses rolling predictions, not batch predictions on historical period
- [ ] `center=True` is NOT used in any rolling window calculation
