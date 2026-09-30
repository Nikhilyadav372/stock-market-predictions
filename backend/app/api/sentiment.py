"""
Sentiment analysis API routes.
"""
from datetime import date, timedelta, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Stock, NewsArticle, SentimentScore
from app.schemas import HeadlineItem, SentimentResponse, DailySentiment, SentimentAnalyzeRequest
from app.sentiment.analyzer import fetch_news, compute_aggregate_sentiment, analyze_text
from app.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])


@router.post("/analyze", response_model=SentimentResponse)
def analyze_sentiment(req: SentimentAnalyzeRequest, db: Session = Depends(get_db)):
    """Fetch news and run sentiment analysis for a symbol."""
    symbol = req.symbol.upper()

    # Get or create stock
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if stock is None:
        stock = Stock(symbol=symbol)
        db.add(stock)
        db.flush()

    # Fetch news articles
    articles, is_sample = fetch_news(symbol, days=req.days)

    # Run sentiment analysis on each article
    aggregate = compute_aggregate_sentiment(articles)

    # Save articles to DB
    headlines = []
    for art in articles:
        # Store in DB
        news = NewsArticle(
            stock_id=stock.id,
            title=art["title"],
            description=art.get("description"),
            url=art.get("url"),
            source=art.get("source"),
            published_at=_parse_dt(art.get("published_at")),
            is_sample=art.get("is_sample", is_sample),
        )
        db.add(news)

        sent = art.get("sentiment", "neutral")
        score = art.get("sentiment_score", 0.0)
        headlines.append(
            HeadlineItem(
                title=art["title"],
                source=art.get("source"),
                published_at=_parse_dt(art.get("published_at")),
                sentiment=sent,
                score=round(score, 4),
                is_sample=art.get("is_sample", is_sample),
            )
        )

    # Save aggregate daily sentiment
    today = date.today()
    existing = (
        db.query(SentimentScore)
        .filter(SentimentScore.stock_id == stock.id, SentimentScore.score_date == today)
        .first()
    )
    if existing:
        existing.compound_score = aggregate["overall_score"]
        existing.positive = aggregate["positive"]
        existing.neutral = aggregate["neutral"]
        existing.negative = aggregate["negative"]
        existing.article_count = aggregate["analyzed_count"]
        existing.is_sample = is_sample
    else:
        db.add(
            SentimentScore(
                stock_id=stock.id,
                score_date=today,
                compound_score=aggregate["overall_score"],
                positive=aggregate["positive"],
                neutral=aggregate["neutral"],
                negative=aggregate["negative"],
                article_count=aggregate["analyzed_count"],
                is_sample=is_sample,
            )
        )

    db.commit()

    # Build daily sentiment trend from DB
    past_30 = date.today() - timedelta(days=30)
    daily_rows = (
        db.query(SentimentScore)
        .filter(SentimentScore.stock_id == stock.id, SentimentScore.score_date >= past_30)
        .order_by(SentimentScore.score_date)
        .all()
    )
    daily = [
        DailySentiment(
            date=row.score_date,
            compound_score=row.compound_score,
            positive=row.positive,
            neutral=row.neutral,
            negative=row.negative,
            article_count=row.article_count,
        )
        for row in daily_rows
    ]

    return SentimentResponse(
        symbol=symbol,
        overall_score=round(aggregate["overall_score"], 4),
        overall_label=aggregate["overall_label"],
        daily_sentiment=daily,
        headlines=headlines[:20],  # cap at 20
        is_sample=is_sample,
    )


@router.get("/{symbol}", response_model=SentimentResponse)
def get_sentiment(symbol: str, db: Session = Depends(get_db)):
    """Return cached sentiment data for a symbol."""
    symbol = symbol.upper()
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()

    if not stock:
        # Return empty response rather than 404
        return SentimentResponse(
            symbol=symbol,
            overall_score=0.0,
            overall_label="neutral",
            daily_sentiment=[],
            headlines=[],
            is_sample=True,
        )

    past_30 = date.today() - timedelta(days=30)
    daily_rows = (
        db.query(SentimentScore)
        .filter(SentimentScore.stock_id == stock.id, SentimentScore.score_date >= past_30)
        .order_by(SentimentScore.score_date)
        .all()
    )

    headlines_db = (
        db.query(NewsArticle)
        .filter(NewsArticle.stock_id == stock.id)
        .order_by(NewsArticle.published_at.desc())
        .limit(20)
        .all()
    )

    overall = 0.0
    if daily_rows:
        overall = sum(r.compound_score for r in daily_rows) / len(daily_rows)

    label = "positive" if overall > 0.05 else ("negative" if overall < -0.05 else "neutral")
    has_sample = any(r.is_sample for r in daily_rows)

    # Re-analyze headlines from DB
    headlines = []
    for art in headlines_db:
        text = f"{art.title} {art.description or ''}".strip()
        sent = analyze_text(text)
        headlines.append(
            HeadlineItem(
                title=art.title,
                source=art.source,
                published_at=art.published_at,
                sentiment=sent["label"],
                score=round(sent["compound"], 4),
                is_sample=art.is_sample,
            )
        )

    return SentimentResponse(
        symbol=symbol,
        overall_score=round(overall, 4),
        overall_label=label,
        daily_sentiment=[
            DailySentiment(
                date=r.score_date,
                compound_score=r.compound_score,
                positive=r.positive,
                neutral=r.neutral,
                negative=r.negative,
                article_count=r.article_count,
            )
            for r in daily_rows
        ],
        headlines=headlines,
        is_sample=has_sample,
    )


def _parse_dt(s) -> Optional[datetime]:
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
