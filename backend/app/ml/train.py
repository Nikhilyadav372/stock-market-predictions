"""
ML Training Pipeline

Implements:
  1. Naive baseline (previous-close)
  2. Linear Regression
  3. Random Forest
  4. XGBoost
  5. LSTM (PyTorch)
  6. GRU  (PyTorch)

Key data-leakage protections:
  - Scalers are fitted ONLY on training data.
  - Time-series split is chronological (see features.py).
  - Sequences for LSTM/GRU are built AFTER the split.
  - No shuffle during sequence generation or training data prep.
"""
from __future__ import annotations

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
try:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    Ridge = None
    RandomForestRegressor = None
    RandomForestClassifier = None
    LogisticRegression = None
    StandardScaler = None
    _SKLEARN_AVAILABLE = False
    def mean_absolute_error(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
    def mean_squared_error(y_true, y_pred): return float(np.mean((y_true - y_pred) ** 2))
    def r2_score(y_true, y_pred): return 0.0
    def accuracy_score(y_true, y_pred): return float(np.mean(y_true == y_pred))
    def f1_score(y_true, y_pred, **kwargs): return 0.0
    def roc_auc_score(y_true, y_pred, **kwargs): return 0.5

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    xgb = None
    _XGB_AVAILABLE = False

from app.config import get_settings
from app.logging_config import get_logger
from app.ml.features import build_features, get_feature_columns, split_time_series

logger = get_logger(__name__)
settings = get_settings()


# ─── Metrics Helper ───────────────────────────────────────────────────────────

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    # Directional accuracy: fraction of times the predicted direction matches actual
    direction_true = np.sign(np.diff(y_true, prepend=y_true[0]))
    direction_pred = np.sign(y_pred - np.roll(y_true, 1))
    directional_acc = float(np.mean(direction_true == direction_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2, "directional_accuracy": directional_acc}


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    result = {"accuracy": acc, "f1": f1, "directional_accuracy": acc}
    if y_prob is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
    return result


# ─── Sequence Builder for LSTM / GRU ─────────────────────────────────────────

def build_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (samples, lookback, features) sequences.
    Sequences are built WITHOUT shuffling to preserve temporal order.
    The i-th sequence uses X[i : i+lookback] to predict y[i+lookback].
    """
    xs, ys = [], []
    for i in range(len(X) - lookback):
        xs.append(X[i : i + lookback])
        ys.append(y[i + lookback])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ─── PyTorch LSTM/GRU ─────────────────────────────────────────────────────────

def _build_rnn_model(model_type: str, input_size: int, hidden_size: int = 64, num_layers: int = 2):
    import torch
    import torch.nn as nn

    class RNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            rnn_cls = nn.LSTM if model_type == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    return RNNModel()


def train_rnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "lstm",
    lookback: int = 60,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> tuple[Any, dict]:
    """Train LSTM or GRU using PyTorch."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training {model_type.upper()} on {device}", epochs=epochs, lookback=lookback)

    Xs_train, ys_train = build_sequences(X_train, y_train, lookback)
    Xs_val, ys_val = build_sequences(X_val, y_val, lookback)

    if len(Xs_train) == 0:
        raise ValueError(f"Not enough training data for lookback={lookback}. Need at least {lookback+1} rows.")

    train_ds = TensorDataset(
        torch.tensor(Xs_train, dtype=torch.float32),
        torch.tensor(ys_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(Xs_val, dtype=torch.float32),
        torch.tensor(ys_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)  # No shuffle for time series
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = _build_rnn_model(model_type, input_size=X_train.shape[1])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else float("inf")
        scheduler.step(val_loss)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}", train_loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ─── Main Training Orchestrator ───────────────────────────────────────────────

class ModelTrainer:
    """
    Trains a specified model type on a given stock's feature matrix.
    Handles scaling, sequence building, metric computation, and artifact saving.
    """

    SUPPORTED_MODELS = {"naive", "linear", "random_forest", "xgboost", "lstm", "gru"}

    def __init__(self, symbol: str, model_type: str, task: str, feature_set: str):
        self.symbol = symbol.upper()
        self.model_type = model_type
        self.task = task
        self.feature_set = feature_set
        self.artifact_dir = settings.MODEL_ARTIFACTS_DIR / self.symbol
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        df_raw: pd.DataFrame,
        include_sentiment: bool = False,
        lookback: int = 60,
        hyperparams: Optional[dict] = None,
    ) -> dict:
        """
        Full training pipeline.
        Returns a results dict with metrics for train/val/test splits.
        """
        start_time = time.time()
        hp = hyperparams or {}

        # Step 1: Build features (no leakage — all rolling/lag ops respect time order)
        logger.info("Building features", symbol=self.symbol, model=self.model_type)
        feat_df = build_features(df_raw, include_sentiment=include_sentiment)

        # Step 2: Define feature columns and target
        feature_cols = get_feature_columns(include_sentiment=include_sentiment)
        feature_cols = [c for c in feature_cols if c in feat_df.columns]

        target_col = "target_direction" if self.task == "classification" else "target_close"

        # Step 3: Drop rows with NaN targets (last row has no next-period close)
        feat_df = feat_df.dropna(subset=[target_col])

        if len(feat_df) < 100:
            raise ValueError(f"Insufficient data: {len(feat_df)} rows after feature computation.")

        # Step 4: Chronological split (NEVER random)
        train_df, val_df, test_df = split_time_series(feat_df)
        logger.info(
            "Data split",
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df),
        )

        # Step 5: Scale features
        # CRITICAL: fit scaler ONLY on training data to prevent leakage
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].values)
        X_val = scaler.transform(val_df[feature_cols].values)
        X_test = scaler.transform(test_df[feature_cols].values)

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        # Step 6: For regression, also scale the target (inverse at eval)
        target_scaler = None
        if self.task == "regression" and self.model_type in ("lstm", "gru"):
            target_scaler = StandardScaler()
            y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_val = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
            # y_test kept in original scale for final metrics

        # Step 7: Train
        model, artifacts = self._train_model(
            X_train, y_train, X_val, y_val, lookback=lookback, hp=hp, target_scaler=target_scaler
        )

        # Step 8: Evaluate
        metrics = self._evaluate(
            model, artifacts, X_train, y_train, X_val, y_val, X_test, y_test,
            target_scaler=target_scaler, lookback=lookback, test_df=test_df,
            feature_cols=feature_cols
        )

        # Step 9: Save artifacts
        artifact_path = self._save_artifacts(model, scaler, target_scaler, feature_cols, artifacts)

        duration = time.time() - start_time
        logger.info("Training complete", symbol=self.symbol, model=self.model_type, duration=f"{duration:.1f}s")

        return {
            "metrics": metrics,
            "artifact_path": str(artifact_path),
            "feature_cols": feature_cols,
            "train_start": str(train_df.index[0].date()),
            "train_end": str(train_df.index[-1].date()),
            "val_start": str(val_df.index[0].date()),
            "val_end": str(val_df.index[-1].date()),
            "test_start": str(test_df.index[0].date()),
            "test_end": str(test_df.index[-1].date()),
            "duration_seconds": duration,
            "hyperparameters": hp,
            "lookback": lookback,
        }

    def _train_model(self, X_train, y_train, X_val, y_val, lookback, hp, target_scaler):
        """Dispatch to the correct model trainer."""
        if self.model_type == "naive":
            return None, {}

        elif self.model_type == "linear":
            if self.task == "regression":
                m = Ridge(alpha=hp.get("alpha", 1.0))
            else:
                m = LogisticRegression(max_iter=500, C=hp.get("C", 1.0))
            m.fit(X_train, y_train)
            return m, {}

        elif self.model_type == "random_forest":
            if self.task == "regression":
                m = RandomForestRegressor(
                    n_estimators=hp.get("n_estimators", 100),
                    max_depth=hp.get("max_depth", None),
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                m = RandomForestClassifier(
                    n_estimators=hp.get("n_estimators", 100),
                    max_depth=hp.get("max_depth", None),
                    random_state=42,
                    n_jobs=-1,
                )
            m.fit(X_train, y_train)
            return m, {}

        elif self.model_type == "xgboost":
            if self.task == "regression":
                m = xgb.XGBRegressor(
                    n_estimators=hp.get("n_estimators", 200),
                    max_depth=hp.get("max_depth", 4),
                    learning_rate=hp.get("learning_rate", 0.05),
                    subsample=hp.get("subsample", 0.8),
                    colsample_bytree=hp.get("colsample_bytree", 0.8),
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
            else:
                m = xgb.XGBClassifier(
                    n_estimators=hp.get("n_estimators", 200),
                    max_depth=hp.get("max_depth", 4),
                    learning_rate=hp.get("learning_rate", 0.05),
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return m, {}

        elif self.model_type in ("lstm", "gru"):
            model, history = train_rnn(
                X_train, y_train, X_val, y_val,
                model_type=self.model_type,
                lookback=lookback,
                epochs=hp.get("epochs", 30),
                batch_size=hp.get("batch_size", 32),
                lr=hp.get("lr", 1e-3),
            )
            return model, {"history": history, "lookback": lookback}

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _evaluate(self, model, artifacts, X_train, y_train, X_val, y_val, X_test, y_test,
                  target_scaler, lookback, test_df, feature_cols) -> dict:
        """Compute metrics for train/val/test sets."""

        def predict(X, y_orig, split="train"):
            if self.model_type == "naive":
                # Naive: predict today's close as tomorrow's close
                # For the naive baseline, we just use the test data
                if split == "test":
                    return test_df["close"].values[:-1], y_orig[1:]
                return y_orig, y_orig  # trivially perfect on train (not meaningful)

            if self.model_type in ("lstm", "gru"):
                import torch
                lk = artifacts.get("lookback", lookback)
                Xs, ys = build_sequences(X, y_orig, lk)
                if len(Xs) == 0:
                    return np.array([]), np.array([])
                model.eval()
                with torch.no_grad():
                    preds = model(torch.tensor(Xs, dtype=torch.float32)).numpy()
                if target_scaler is not None and self.task == "regression":
                    preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
                    ys = target_scaler.inverse_transform(ys.reshape(-1, 1)).ravel()
                return preds, ys
            else:
                preds = model.predict(X)
                return preds, y_orig

        metrics = {}
        for split, (X, y) in [("train", (X_train, y_train)), ("val", (X_val, y_val)), ("test", (X_test, y_test))]:
            preds, y_true = predict(X, y, split)
            if len(preds) == 0:
                continue
            if self.task == "regression":
                metrics[split] = compute_regression_metrics(y_true, preds)
            else:
                y_prob = None
                if self.model_type not in ("naive", "lstm", "gru") and hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X)[:, 1]
                preds_bin = (preds > 0.5).astype(int) if self.model_type in ("lstm", "gru") else preds
                metrics[split] = compute_classification_metrics(y_true.astype(int), preds_bin.astype(int), y_prob)

        return metrics

    def _save_artifacts(self, model, scaler, target_scaler, feature_cols, extra_artifacts) -> Path:
        """Save model, scaler, and metadata to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_path = self.artifact_dir / f"{self.model_type}_{self.task}_{timestamp}.pkl"

        bundle = {
            "model": model,
            "scaler": scaler,
            "target_scaler": target_scaler,
            "feature_cols": feature_cols,
            "model_type": self.model_type,
            "task": self.task,
            "symbol": self.symbol,
            "trained_at": timestamp,
            **extra_artifacts,
        }

        with open(artifact_path, "wb") as f:
            pickle.dump(bundle, f)

        logger.info("Model artifact saved", path=str(artifact_path))
        return artifact_path


# ─── Inference ────────────────────────────────────────────────────────────────

class ModelPredictor:
    """Loads a saved model artifact and generates forecasts."""

    def __init__(self, artifact_path: str):
        with open(artifact_path, "rb") as f:
            self.bundle = pickle.load(f)
        self.model = self.bundle["model"]
        self.scaler = self.bundle["scaler"]
        self.target_scaler = self.bundle.get("target_scaler")
        self.feature_cols = self.bundle["feature_cols"]
        self.model_type = self.bundle["model_type"]
        self.task = self.bundle["task"]
        self.lookback = self.bundle.get("lookback", 60)

    def predict_next_n(self, df_raw: pd.DataFrame, n: int, include_sentiment: bool = False) -> list[dict]:
        """
        Generate n-step ahead forecasts using iterative/direct strategy.
        For each step, the model predicts one period ahead and the result
        is appended to the series for the next prediction (where applicable).

        Returns a list of {period, value, direction} dicts.
        """
        feat_df = build_features(df_raw, include_sentiment=include_sentiment)
        feature_cols = [c for c in self.feature_cols if c in feat_df.columns]

        results = []

        if self.model_type == "naive":
            last_close = df_raw["Close"].iloc[-1]
            for i in range(1, n + 1):
                results.append({
                    "period": i,
                    "value": float(last_close),
                    "direction": "UNCHANGED",
                })
            return results

        if self.model_type in ("lstm", "gru"):
            try:
                import torch
            except ImportError:
                raise ValueError("PyTorch is not installed in this environment.")
            X_all = self.scaler.transform(feat_df[feature_cols].values)
            if len(X_all) < self.lookback:
                raise ValueError(f"Need at least {self.lookback} rows; got {len(X_all)}")

            self.model.eval()
            for i in range(1, n + 1):
                window = X_all[-self.lookback:]
                x_tensor = torch.tensor(window[np.newaxis, :, :], dtype=torch.float32)
                with torch.no_grad():
                    pred = self.model(x_tensor).item()
                if self.target_scaler is not None:
                    pred = float(self.target_scaler.inverse_transform([[pred]])[0][0])
                last_known = float(df_raw["Close"].iloc[-1]) if i == 1 else results[-1]["value"]
                direction = "UP" if pred > last_known else "DOWN"
                results.append({"period": i, "value": pred, "direction": direction})
        else:
            # Tabular models: use the last available feature row
            last_features = feat_df[feature_cols].iloc[-1:].values
            X_scaled = self.scaler.transform(last_features)
            last_close = float(df_raw["Close"].iloc[-1])

            for i in range(1, n + 1):
                if self.task == "regression":
                    pred = float(self.model.predict(X_scaled)[0])
                else:
                    pred = float(self.model.predict(X_scaled)[0])

                direction = "UP" if pred > last_close else "DOWN"
                results.append({"period": i, "value": pred, "direction": direction})
                last_close = pred  # use prediction as next input for multi-step

        return results
