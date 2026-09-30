"""
Stock & market data API routes.
"""
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Stock, HistoricalPrice
from app.schemas import (
    DataRefreshRequest,
    DataRefreshResponse,
    IndicatorsResponse,
    IndicatorPoint,
    PriceHistoryResponse,
    PricePoint,
    StockResponse,
)
from app.services.market_data import MarketDataService
from app.ml.features import (
    build_features,
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
)
from app.logging_config import get_logger

import pandas as pd

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["stocks"])

_market_svc = MarketDataService()

POPULAR_STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corp."},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "TSLA", "name": "Tesla Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corp."},
    {"symbol": "META", "name": "Meta Platforms"},
    {"symbol": "JPM", "name": "JPMorgan Chase"},
    {"symbol": "BRK-B", "name": "Berkshire Hathaway"},
    {"symbol": "V", "name": "Visa Inc."},
]


@router.get("/health")
def health_check():
    from datetime import datetime
    from app.config import get_settings
    settings = get_settings()
    try:
        db = next(get_db())
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "unavailable"
    return {
        "status": "ok",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/stocks", response_model=list[dict])
def list_stocks(db: Session = Depends(get_db)):
    """Return list of tracked stocks plus popular defaults."""
    db_stocks = db.query(Stock).all()
    tracked = [{"symbol": s.symbol, "name": s.name, "tracked": True} for s in db_stocks]
    tracked_symbols = {s["symbol"] for s in tracked}
    extras = [s for s in POPULAR_STOCKS if s["symbol"] not in tracked_symbols]
    for e in extras:
        e["tracked"] = False
    return tracked + extras


@router.get("/stocks/{symbol}")
def get_stock(symbol: str, db: Session = Depends(get_db)):
    symbol = symbol.upper()
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()

    info = _market_svc.get_stock_info(symbol)
    if stock is None and not info:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

    if stock:
        return {
            "id": stock.id,
            "symbol": stock.symbol,
            "name": stock.name or info.get("name"),
            "sector": stock.sector or info.get("sector"),
            "industry": stock.industry or info.get("industry"),
            "market_cap": stock.market_cap or info.get("market_cap"),
            "currency": stock.currency,
            "exchange": stock.exchange or info.get("exchange"),
        }
    return {
        "symbol": symbol,
        **info,
        "tracked": False,
    }


@router.get("/stocks/{symbol}/history", response_model=PriceHistoryResponse)
def get_price_history(
    symbol: str,
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    db: Session = Depends(get_db),
):
    symbol = symbol.upper()
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    # Try DB first
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if stock:
        prices = (
            db.query(HistoricalPrice)
            .filter(
                HistoricalPrice.stock_id == stock.id,
                HistoricalPrice.date >= start_date,
                HistoricalPrice.date <= end_date,
            )
            .order_by(HistoricalPrice.date)
            .all()
        )
        if prices:
            return PriceHistoryResponse(
                symbol=symbol,
                prices=[
                    PricePoint(
                        date=p.date,
                        open=p.open,
                        high=p.high,
                        low=p.low,
                        close=p.close,
                        volume=p.volume,
                        adj_close=p.adj_close,
                    )
                    for p in prices
                ],
                count=len(prices),
                start_date=prices[0].date if prices else None,
                end_date=prices[-1].date if prices else None,
            )

    # Fallback to market data service
    try:
        df, is_sample = _market_svc.fetch(symbol, start_date, end_date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    price_points = [
        PricePoint(
            date=idx.date(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=int(row["Volume"]),
            adj_close=float(row.get("Adj_Close", row["Close"])),
        )
        for idx, row in df.iterrows()
    ]
    return PriceHistoryResponse(
        symbol=symbol,
        prices=price_points,
        count=len(price_points),
        start_date=price_points[0].date if price_points else None,
        end_date=price_points[-1].date if price_points else None,
    )


@router.get("/stocks/{symbol}/indicators", response_model=IndicatorsResponse)
def get_indicators(
    symbol: str,
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    db: Session = Depends(get_db),
):
    symbol = symbol.upper()
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    try:
        df, _ = _market_svc.fetch(symbol, start_date - timedelta(days=100), end_date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    feat = build_features(df)

    # Filter back to requested date range
    mask = (feat.index.date >= start_date) & (feat.index.date <= end_date)
    feat = feat[mask]

    indicators = []
    for idx, row in feat.iterrows():
        indicators.append(
            IndicatorPoint(
                date=idx.date(),
                close=float(row.get("close", 0)),
                sma_20=_safe_float(row.get("sma_20")),
                sma_50=_safe_float(row.get("sma_50")),
                ema_12=_safe_float(row.get("ema_12")),
                ema_26=_safe_float(row.get("ema_26")),
                rsi=_safe_float(row.get("rsi")),
                macd=_safe_float(row.get("macd")),
                macd_signal=_safe_float(row.get("macd_signal")),
                macd_hist=_safe_float(row.get("macd_hist")),
                bb_upper=_safe_float(row.get("bb_upper")),
                bb_lower=_safe_float(row.get("bb_lower")),
                bb_middle=_safe_float(row.get("bb_middle")),
                volatility=_safe_float(row.get("volatility_20")),
            )
        )

    return IndicatorsResponse(symbol=symbol, indicators=indicators)


@router.post("/data/refresh", response_model=DataRefreshResponse)
def refresh_data(req: DataRefreshRequest, db: Session = Depends(get_db)):
    """Download / update historical market data for a symbol."""
    symbol = req.symbol.upper()
    start = req.start_date or (date.today() - timedelta(days=5 * 365))
    end = req.end_date or date.today()

    # Clear cache if force refresh
    if req.force:
        from app.config import get_settings as gs
        cache_path = gs().raw_data_dir / f"{symbol}.csv"
        if cache_path.exists():
            cache_path.unlink()
            logger.info("Cache cleared for force refresh", symbol=symbol)

    try:
        df, is_sample = _market_svc.fetch(symbol, start, end)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Upsert stock record
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if stock is None:
        info = _market_svc.get_stock_info(symbol)
        stock = Stock(
            symbol=symbol,
            name=info.get("name"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=info.get("market_cap"),
            currency=info.get("currency", "USD"),
            exchange=info.get("exchange"),
        )
        db.add(stock)
        db.flush()

    # Upsert prices
    existing_dates = {
        p.date
        for p in db.query(HistoricalPrice.date)
        .filter(HistoricalPrice.stock_id == stock.id)
        .all()
    }

    new_records = 0
    for idx, row in df.iterrows():
        row_date = idx.date()
        if row_date not in existing_dates:
            db.add(
                HistoricalPrice(
                    stock_id=stock.id,
                    date=row_date,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                    adj_close=float(row.get("Adj_Close", row["Close"])),
                )
            )
            new_records += 1

    db.commit()
    total = db.query(HistoricalPrice).filter(HistoricalPrice.stock_id == stock.id).count()

    logger.info("Data refresh complete", symbol=symbol, new_records=new_records, is_sample=is_sample)
    return DataRefreshResponse(
        symbol=symbol,
        records_added=new_records,
        records_total=total,
        start_date=start,
        end_date=end,
        provider=type(_market_svc.provider).__name__,
        is_sample=is_sample,
    )


def _safe_float(v) -> Optional[float]:
    """Return float or None for NaN/None values."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return None
