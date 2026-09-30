r"""
Model Training & Evaluation Demonstration
==========================================
Trains all 4 sklearn/XGBoost models (Naive, Linear, Random Forest, XGBoost)
directly via the backend's ModelTrainer, evaluates them on the holdout test set,
and visualises comparative performance.

Run from the notebooks/ directory:
    ..\backend\venv\Scripts\python.exe 03_model_training.py
"""
import sys
import os

_notebooks_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.join(_notebooks_dir, "..", "backend")
sys.path.insert(0, _backend_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date, timedelta

from app.services.market_data import MarketDataService
from app.ml.features import build_features, split_time_series
from app.ml.train import ModelTrainer

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL = "AAPL"
TASK = "regression"           # "regression" or "classification"
FEATURE_SET = "market"        # "market" or "market+sentiment"
MODELS_TO_TRAIN = ["naive", "linear", "random_forest", "xgboost"]
OUTPUT_DIR = os.path.normpath(os.path.join(_notebooks_dir, "..", "data", "processed"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Fetch Data ─────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  03 — Model Training Demo  ({SYMBOL}, task={TASK})")
print("=" * 60)

svc = MarketDataService()
start = date.today() - timedelta(days=5 * 365)
df_raw, is_sample = svc.fetch(SYMBOL, start, date.today())
print(f"\n[Data] {len(df_raw)} rows fetched {'(SAMPLE DATA)' if is_sample else '(real data)'}")

# ── 2. Feature Engineering & Split (informational) ───────────────────────────
feat = build_features(df_raw)
train_df, val_df, test_df = split_time_series(feat)
print(f"\n[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"  Train period : {train_df.index[0].date()} -> {train_df.index[-1].date()}")
print(f"  Val   period : {val_df.index[0].date()} -> {val_df.index[-1].date()}")
print(f"  Test  period : {test_df.index[0].date()} -> {test_df.index[-1].date()}")

# ── 3. Train All Models ───────────────────────────────────────────────────────
results = {}
print(f"\n[Training] Running {len(MODELS_TO_TRAIN)} models ...\n")

for model_type in MODELS_TO_TRAIN:
    print(f"  -> {model_type.upper()} ...", end=" ", flush=True)
    trainer = ModelTrainer(
        symbol=SYMBOL,
        model_type=model_type,
        task=TASK,
        feature_set=FEATURE_SET,
    )
    try:
        res = trainer.train(
            df_raw=df_raw,
            include_sentiment=(FEATURE_SET == "market+sentiment"),
        )
        results[model_type] = res
        m = res["metrics"].get("test", {})
        if TASK == "regression":
            print(f"MAE={m.get('mae', float('nan')):.4f}  "
                  f"RMSE={m.get('rmse', float('nan')):.4f}  "
                  f"R2={m.get('r2', float('nan')):.4f}  "
                  f"DirAcc={m.get('directional_accuracy', float('nan')):.3f}")
        else:
            print(f"Acc={m.get('accuracy', float('nan')):.3f}  "
                  f"F1={m.get('f1', float('nan')):.3f}")
    except Exception as exc:
        print(f"FAILED: {exc}")
        results[model_type] = None

# ── 4. Summary Table ──────────────────────────────────────────────────────────
print("\n\n=== RESULTS SUMMARY (Test Set) ===\n")
if TASK == "regression":
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R2':>8} {'DirAcc':>10}")
    print("-" * 62)
    for m_type, res in results.items():
        if res is None:
            print(f"{m_type:<20} {'FAILED':>10}")
            continue
        m = res["metrics"].get("test", {})
        print(f"{m_type:<20} "
              f"{m.get('mae', float('nan')):>10.4f} "
              f"{m.get('rmse', float('nan')):>10.4f} "
              f"{m.get('r2', float('nan')):>8.4f} "
              f"{m.get('directional_accuracy', float('nan')):>10.3f}")
else:
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>8} {'ROC-AUC':>10}")
    print("-" * 52)
    for m_type, res in results.items():
        if res is None:
            print(f"{m_type:<20} {'FAILED':>10}")
            continue
        m = res["metrics"].get("test", {})
        print(f"{m_type:<20} "
              f"{m.get('accuracy', float('nan')):>10.3f} "
              f"{m.get('f1', float('nan')):>8.3f} "
              f"{m.get('roc_auc', float('nan')):>10.3f}")

# ── 5. Visualisation ──────────────────────────────────────────────────────────
valid_results = {k: v for k, v in results.items() if v is not None}

if not valid_results:
    print("\nNo models trained successfully — skipping plots.")
    sys.exit(0)

fig, axes = plt.subplots(1, 2 if TASK == "regression" else 1,
                         figsize=(14, 6))
if TASK == "regression":
    axes = axes if isinstance(axes, np.ndarray) else [axes]
else:
    axes = [axes]

fig.suptitle(
    f"Model Comparison — {SYMBOL} ({TASK.title()}, {FEATURE_SET})",
    fontsize=14, fontweight="bold", color="#e2e8f0",
    y=1.01,
)

COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#a78bfa", "#38bdf8"]
model_names = list(valid_results.keys())

if TASK == "regression":
    # Plot 1: MAE comparison
    maes = [valid_results[m]["metrics"]["test"].get("mae", 0) for m in model_names]
    bars = axes[0].barh(model_names, maes, color=COLORS[:len(model_names)], height=0.5)
    axes[0].set_title("Test MAE (lower is better)", color="#e2e8f0", fontweight="bold")
    axes[0].set_xlabel("Mean Absolute Error ($)", color="#94a3b8")
    for bar, val in zip(bars, maes):
        axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"${val:.2f}", va="center", color="#e2e8f0", fontsize=9)

    # Plot 2: R² comparison
    r2s = [valid_results[m]["metrics"]["test"].get("r2", 0) for m in model_names]
    bars2 = axes[1].barh(model_names, r2s, color=COLORS[:len(model_names)], height=0.5)
    axes[1].set_title("Test R² (higher is better)", color="#e2e8f0", fontweight="bold")
    axes[1].set_xlabel("R² Score", color="#94a3b8")
    axes[1].axvline(0, color="white", linestyle="--", alpha=0.3)
    for bar, val in zip(bars2, r2s):
        axes[1].text(max(bar.get_width() + 0.005, 0.01),
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", color="#e2e8f0", fontsize=9)
else:
    accs = [valid_results[m]["metrics"]["test"].get("accuracy", 0) for m in model_names]
    bars = axes[0].barh(model_names, accs, color=COLORS[:len(model_names)], height=0.5)
    axes[0].set_title("Test Accuracy (higher is better)", color="#e2e8f0", fontweight="bold")
    axes[0].set_xlabel("Accuracy", color="#94a3b8")
    axes[0].axvline(0.5, color="white", linestyle="--", alpha=0.5, label="Random baseline")
    axes[0].legend(facecolor="#141428", labelcolor="#94a3b8")
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", color="#e2e8f0", fontsize=9)

for ax in axes:
    ax.set_facecolor("#0f0f1a")
    ax.tick_params(colors="#94a3b8")
    ax.spines["bottom"].set_color("#333355")
    ax.spines["left"].set_color("#333355")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.patch.set_facecolor("#0a0a0f")
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "03_model_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n[OK] Chart saved to {out_path}")

# ── 6. Duration summary ───────────────────────────────────────────────────────
print("\n[Training times]")
for m_type, res in results.items():
    if res:
        print(f"  {m_type:<20} {res['duration_seconds']:.1f}s")

print("\n[NOTE] This is an educational demo. Predictions are NOT financial advice.")
