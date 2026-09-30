"""
Backtesting API routes.
"""
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Backtest, MLModel, Stock
from app.schemas import BacktestRequest, BacktestResponse, EquityPoint
from app.services.market_data import MarketDataService
from app.ml.train import ModelPredictor
from app.backtesting.engine import run_backtest
from app.logging_config import get_logger

import pandas as pd
from pathlib import Path

logger = get_logger(__name__)
router = APIRouter(prefix="/api/backtest", tags=["backtesting"])

_market_svc = MarketDataService()


@router.post("", response_model=BacktestResponse)
def run_backtest_endpoint(req: BacktestRequest, db: Session = Depends(get_db)):
    """
    Run historical backtest for a given model and date range.
    Uses chronological data only — no future look-ahead.
    """
    symbol = req.symbol.upper()
    ml_model = db.query(MLModel).get(req.model_id)
    if not ml_model:
        raise HTTPException(status_code=404, detail="Model not found")

    if not ml_model.artifact_path or not Path(ml_model.artifact_path).exists():
        raise HTTPException(status_code=422, detail="Model artifact not found. Please train the model first.")

    # Fetch historical data for backtest period (add extra for feature warmup)
    warmup_start = req.start_date - timedelta(days=300)
    try:
        df_raw, _ = _market_svc.fetch(symbol, warmup_start, req.end_date)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Generate predictions for the backtest period using the trained model
    predictor = ModelPredictor(ml_model.artifact_path)

    # Get price series for the actual backtest period only
    backtest_df = df_raw[df_raw.index.date >= req.start_date]
    if len(backtest_df) < 10:
        raise HTTPException(status_code=422, detail="Not enough data in the selected backtest period.")

    # Generate one prediction per day using rolling windows
    # For each day, we use only data up to that day (no future leakage)
    directions = []
    for i in range(len(backtest_df)):
        # Use data up to (but not including) day i+1
        available_data = df_raw[df_raw.index <= backtest_df.index[i]]
        if len(available_data) < 70:
            directions.append("HOLD")
            continue
        try:
            preds = predictor.predict_next_n(available_data, n=1)
            directions.append(preds[0].get("direction", "HOLD") if preds else "HOLD")
        except Exception:
            directions.append("HOLD")

    # Map "HOLD" to "DOWN" (stay flat)
    mapped_directions = ["UP" if d == "UP" else "DOWN" for d in directions]

    # Run backtest engine
    price_series = backtest_df["Close"]
    results = run_backtest(price_series, mapped_directions)

    # Store in DB
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        stock = Stock(symbol=symbol)
        db.add(stock)
        db.flush()

    bt = Backtest(
        stock_symbol=symbol,
        model_id=req.model_id,
        start_date=req.start_date,
        end_date=req.end_date,
        strategy=req.strategy,
        total_trades=results["total_trades"],
        win_rate=results["win_rate"],
        cumulative_return=results["cumulative_return"],
        max_drawdown=results["max_drawdown"],
        sharpe_ratio=results["sharpe_ratio"],
        equity_curve=results["equity_curve"],
        trade_log=results["trade_log"][:100],  # cap to avoid huge JSON
    )
    db.add(bt)
    db.commit()
    db.refresh(bt)

    equity_pts = [
        EquityPoint(
            date=ep["date"],
            equity=ep["equity"],
            cumulative_return=ep["cumulative_return"],
        )
        for ep in results["equity_curve"]
    ]

    return BacktestResponse(
        id=bt.id,
        symbol=symbol,
        model_id=req.model_id,
        start_date=req.start_date,
        end_date=req.end_date,
        strategy=req.strategy,
        total_trades=results["total_trades"],
        win_rate=results["win_rate"],
        cumulative_return=results["cumulative_return"],
        max_drawdown=results["max_drawdown"],
        sharpe_ratio=results["sharpe_ratio"],
        equity_curve=equity_pts,
        created_at=bt.created_at,
    )


@router.get("/{backtest_id}")
def get_backtest(backtest_id: int, db: Session = Depends(get_db)):
    bt = db.query(Backtest).get(backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return {
        "id": bt.id,
        "symbol": bt.stock_symbol,
        "model_id": bt.model_id,
        "start_date": bt.start_date.isoformat(),
        "end_date": bt.end_date.isoformat(),
        "strategy": bt.strategy,
        "total_trades": bt.total_trades,
        "win_rate": bt.win_rate,
        "cumulative_return": bt.cumulative_return,
        "max_drawdown": bt.max_drawdown,
        "sharpe_ratio": bt.sharpe_ratio,
        "equity_curve": bt.equity_curve,
        "disclaimer": (
            "This is an educational backtest. It does not account for transaction costs, "
            "slippage, taxes, or real-world execution. Past performance does not indicate future results."
        ),
        "created_at": bt.created_at.isoformat(),
    }
