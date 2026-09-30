"""
ML model training, registry, and prediction API routes.
"""
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import MLModel, ModelRun, Prediction, Stock, ModelExplanation
from app.schemas import (
    ForecastPoint,
    MLModelResponse,
    PredictRequest,
    PredictionResponse,
    TrainRequest,
)
from app.services.market_data import MarketDataService
from app.ml.train import ModelTrainer, ModelPredictor
from app.ml.explainability import explain_prediction
from app.ml.features import build_features, get_feature_columns
from app.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])
predict_router = APIRouter(prefix="/api", tags=["predictions"])

_market_svc = MarketDataService()


def _get_or_create_stock(symbol: str, db: Session) -> Stock:
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if stock is None:
        stock = Stock(symbol=symbol)
        db.add(stock)
        db.flush()
    return stock


@router.post("/train")
def train_model(req: TrainRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Trigger model training. Training runs in the background.
    Returns the model_run_id immediately so the client can poll status.
    """
    symbol = req.symbol.upper()

    # Fetch data
    start = date.today() - timedelta(days=5 * 365)
    end = date.today()
    try:
        df_raw, is_sample = _market_svc.fetch(symbol, start, end)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Data fetch failed: {e}")

    if len(df_raw) < 120:
        raise HTTPException(status_code=422, detail="Insufficient data for training (need ≥120 rows).")

    # Create model record
    stock = _get_or_create_stock(symbol, db)
    ml_model = MLModel(
        name=f"{req.model_type.replace('_', ' ').title()} ({symbol})",
        model_type=req.model_type,
        stock_symbol=symbol,
        task=req.task,
        feature_set=req.feature_set,
        hyperparameters=req.hyperparameters,
    )
    db.add(ml_model)
    db.flush()

    model_run = ModelRun(model_id=ml_model.id, status="pending")
    db.add(model_run)
    db.commit()

    model_id = ml_model.id
    run_id = model_run.id

    # Run training in background
    background_tasks.add_task(
        _run_training,
        model_id=model_id,
        run_id=run_id,
        symbol=symbol,
        df_raw=df_raw,
        req=req,
    )

    return {
        "model_id": model_id,
        "run_id": run_id,
        "status": "training_started",
        "message": f"Training {req.model_type} for {symbol}. Poll /api/models/runs/{run_id} for status.",
    }


def _run_training(model_id: int, run_id: int, symbol: str, df_raw, req: TrainRequest):
    """Background task: train model and update DB."""
    from app.database import SessionLocal
    from datetime import datetime

    db = SessionLocal()
    try:
        run = db.query(ModelRun).get(run_id)
        run.status = "running"
        run.started_at = datetime.utcnow()
        db.commit()

        trainer = ModelTrainer(
            symbol=symbol,
            model_type=req.model_type,
            task=req.task,
            feature_set=req.feature_set,
        )
        results = trainer.train(
            df_raw=df_raw,
            include_sentiment=(req.feature_set == "market+sentiment"),
            lookback=req.lookback_window,
            hyperparams=req.hyperparameters,
        )

        # Update model record
        ml_model = db.query(MLModel).get(model_id)
        ml_model.artifact_path = results["artifact_path"]
        ml_model.feature_names = results["feature_cols"]
        ml_model.training_start_date = date.fromisoformat(results["train_start"])
        ml_model.training_end_date = date.fromisoformat(results["train_end"])

        # Update run record
        metrics = results["metrics"]
        run.status = "done"
        run.finished_at = datetime.utcnow()
        run.duration_seconds = results["duration_seconds"]

        if "train" in metrics:
            m = metrics["train"]
            run.train_mae = m.get("mae")
            run.train_rmse = m.get("rmse")
            run.train_r2 = m.get("r2")

        if "val" in metrics:
            m = metrics["val"]
            run.val_mae = m.get("mae")
            run.val_rmse = m.get("rmse")
            run.val_r2 = m.get("r2")

        if "test" in metrics:
            m = metrics["test"]
            run.test_mae = m.get("mae")
            run.test_rmse = m.get("rmse")
            run.test_r2 = m.get("r2")
            run.test_accuracy = m.get("accuracy")
            run.test_f1 = m.get("f1")
            run.test_roc_auc = m.get("roc_auc")
            run.directional_accuracy = m.get("directional_accuracy")

        db.commit()
        logger.info("Training background task complete", model_id=model_id, run_id=run_id)

    except Exception as exc:
        logger.error("Training failed", model_id=model_id, error=str(exc))
        run = db.query(ModelRun).get(run_id)
        run.status = "failed"
        run.error_message = str(exc)
        from datetime import datetime
        run.finished_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


@router.get("/runs/{run_id}")
def get_run_status(run_id: int, db: Session = Depends(get_db)):
    run = db.query(ModelRun).get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run.id,
        "model_id": run.model_id,
        "status": run.status,
        "train_mae": run.train_mae,
        "train_rmse": run.train_rmse,
        "val_mae": run.val_mae,
        "val_rmse": run.val_rmse,
        "test_mae": run.test_mae,
        "test_rmse": run.test_rmse,
        "test_r2": run.test_r2,
        "test_accuracy": run.test_accuracy,
        "test_f1": run.test_f1,
        "directional_accuracy": run.directional_accuracy,
        "duration_seconds": run.duration_seconds,
        "error_message": run.error_message,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


@router.get("", response_model=list[dict])
def list_models(symbol: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(MLModel)
    if symbol:
        q = q.filter(MLModel.stock_symbol == symbol.upper())
    models = q.order_by(MLModel.created_at.desc()).all()
    result = []
    for m in models:
        # Get latest run metrics
        latest_run = (
            db.query(ModelRun)
            .filter(ModelRun.model_id == m.id, ModelRun.status == "done")
            .order_by(ModelRun.created_at.desc())
            .first()
        )
        result.append({
            "id": m.id,
            "name": m.name,
            "model_type": m.model_type,
            "stock_symbol": m.stock_symbol,
            "task": m.task,
            "feature_set": m.feature_set,
            "is_active": m.is_active,
            "created_at": m.created_at.isoformat(),
            "has_artifact": bool(m.artifact_path and Path(m.artifact_path).exists()),
            "test_mae": latest_run.test_mae if latest_run else None,
            "test_rmse": latest_run.test_rmse if latest_run else None,
            "test_r2": latest_run.test_r2 if latest_run else None,
            "test_accuracy": latest_run.test_accuracy if latest_run else None,
            "directional_accuracy": latest_run.directional_accuracy if latest_run else None,
        })
    return result


@router.get("/{model_id}")
def get_model(model_id: int, db: Session = Depends(get_db)):
    m = db.query(MLModel).get(model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    runs = (
        db.query(ModelRun)
        .filter(ModelRun.model_id == model_id)
        .order_by(ModelRun.created_at.desc())
        .limit(5)
        .all()
    )
    return {
        "id": m.id,
        "name": m.name,
        "model_type": m.model_type,
        "stock_symbol": m.stock_symbol,
        "task": m.task,
        "feature_set": m.feature_set,
        "hyperparameters": m.hyperparameters,
        "feature_names": m.feature_names,
        "training_start_date": m.training_start_date.isoformat() if m.training_start_date else None,
        "training_end_date": m.training_end_date.isoformat() if m.training_end_date else None,
        "is_active": m.is_active,
        "created_at": m.created_at.isoformat(),
        "runs": [
            {
                "run_id": r.id,
                "status": r.status,
                "test_mae": r.test_mae,
                "test_rmse": r.test_rmse,
                "test_r2": r.test_r2,
                "test_accuracy": r.test_accuracy,
                "directional_accuracy": r.directional_accuracy,
                "duration_seconds": r.duration_seconds,
                "created_at": r.created_at.isoformat(),
            }
            for r in runs
        ],
    }


# ─── Predictions ─────────────────────────────────────────────────────────────

@predict_router.post("/predict")
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    ml_model = db.query(MLModel).get(req.model_id)
    if not ml_model:
        raise HTTPException(status_code=404, detail="Model not found")

    if not ml_model.artifact_path or not Path(ml_model.artifact_path).exists():
        raise HTTPException(status_code=422, detail="Model artifact not found. Please retrain the model.")

    # Fetch recent data for inference
    try:
        df_raw, _ = _market_svc.fetch(req.symbol.upper())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Generate forecast
    try:
        predictor = ModelPredictor(ml_model.artifact_path)
        raw_preds = predictor.predict_next_n(
            df_raw=df_raw,
            n=req.horizon,
            include_sentiment=(ml_model.feature_set == "market+sentiment"),
        )
    except Exception as exc:
        logger.error("Prediction failed", model_id=req.model_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # Build confidence intervals (naive ±2% for now — honest about uncertainty)
    forecast = []
    for p in raw_preds:
        val = p["value"]
        margin = abs(val) * 0.02  # ±2% uncertainty band
        forecast.append(
            ForecastPoint(
                period=p["period"],
                value=round(val, 4),
                lower_bound=round(val - margin, 4),
                upper_bound=round(val + margin, 4),
                direction=p.get("direction"),
            )
        )

    # Get latest metrics
    latest_run = (
        db.query(ModelRun)
        .filter(ModelRun.model_id == ml_model.id, ModelRun.status == "done")
        .order_by(ModelRun.created_at.desc())
        .first()
    )
    metrics = {}
    if latest_run:
        metrics = {
            "test_mae": latest_run.test_mae,
            "test_rmse": latest_run.test_rmse,
            "test_r2": latest_run.test_r2,
            "directional_accuracy": latest_run.directional_accuracy,
        }

    # Save prediction to DB
    stock = _get_or_create_stock(req.symbol.upper(), db)
    pred = Prediction(
        stock_id=stock.id,
        model_id=ml_model.id,
        task=ml_model.task,
        horizon=req.horizon,
        predicted_values=[f.model_dump() for f in forecast],
        model_metrics=metrics,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)

    # Generate SHAP explanation asynchronously
    try:
        _generate_explanation(pred.id, ml_model, df_raw, db)
    except Exception as exc:
        logger.warning("SHAP explanation skipped", error=str(exc))

    return {
        "id": pred.id,
        "symbol": req.symbol.upper(),
        "model_id": ml_model.id,
        "model_name": ml_model.name,
        "task": ml_model.task,
        "horizon": req.horizon,
        "forecast": [f.model_dump() for f in forecast],
        "model_metrics": metrics,
        "disclaimer": (
            "Predictions are experimental machine-learning forecasts for educational "
            "and research purposes only and are not financial advice."
        ),
        "created_at": pred.created_at.isoformat(),
    }


def _generate_explanation(prediction_id: int, ml_model: MLModel, df_raw, db: Session):
    """Compute and store SHAP explanation for the prediction."""
    if ml_model.model_type not in ("xgboost", "random_forest", "linear"):
        return

    if not ml_model.artifact_path or not Path(ml_model.artifact_path).exists():
        return

    import pickle
    with open(ml_model.artifact_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    feat_df = build_features(df_raw)
    feature_cols = [c for c in feature_cols if c in feat_df.columns]
    X = scaler.transform(feat_df[feature_cols].values)

    expl = explain_prediction(model, X, feature_cols, ml_model.model_type)

    explanation = ModelExplanation(
        prediction_id=prediction_id,
        shap_values=expl["shap_values"],
        feature_importance=expl["feature_importance"],
        top_positive_features=expl["top_positive_features"],
        top_negative_features=expl["top_negative_features"],
        explanation_text=expl["explanation_text"],
    )
    db.add(explanation)
    db.commit()


@predict_router.get("/predictions")
def get_predictions(
    symbol: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    q = db.query(Prediction)
    if symbol:
        stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if stock:
            q = q.filter(Prediction.stock_id == stock.id)
    preds = q.order_by(Prediction.created_at.desc()).limit(limit).all()
    result = []
    for p in preds:
        stock = db.query(Stock).get(p.stock_id)
        model = db.query(MLModel).get(p.model_id)
        result.append({
            "id": p.id,
            "symbol": stock.symbol if stock else "UNKNOWN",
            "model_name": model.name if model else "Unknown",
            "task": p.task,
            "horizon": p.horizon,
            "forecast": p.predicted_values,
            "model_metrics": p.model_metrics,
            "created_at": p.created_at.isoformat(),
        })
    return result


@predict_router.get("/explain/{prediction_id}")
def get_explanation(prediction_id: int, db: Session = Depends(get_db)):
    pred = db.query(Prediction).get(prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")

    expl = db.query(ModelExplanation).filter(ModelExplanation.prediction_id == prediction_id).first()
    if not expl:
        raise HTTPException(
            status_code=404,
            detail="No explanation available for this prediction. Explanations are generated for tree-based and linear models."
        )

    model = db.query(MLModel).get(pred.model_id)
    return {
        "prediction_id": prediction_id,
        "model_type": model.model_type if model else "unknown",
        "top_positive_features": expl.top_positive_features or [],
        "top_negative_features": expl.top_negative_features or [],
        "feature_importance": expl.feature_importance or {},
        "explanation_text": expl.explanation_text,
        "disclaimer": (
            "Feature importance shows correlation within the model, not causal relationships. "
            "SHAP values explain model behavior for this specific prediction."
        ),
    }
