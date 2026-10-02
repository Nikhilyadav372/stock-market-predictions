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

    # Save daily sentiment scores for each date found in articles
    articles_by_date = {}
    for art in articles:
        p_dt = _parse_dt(art.get("published_at"))
        d_val = p_dt.date() if p_dt else date.today()
        articles_by_date.setdefault(d_val, []).append(art)

    articles_by_date.setdefault(date.today(), [])

    for d_val, d_arts in articles_by_date.items():
        if d_arts:
            d_scores = [a.get("sentiment_score", 0.0) for a in d_arts]
            d_pos = [1 if a.get("sentiment") == "positive" else 0 for a in d_arts]
            d_neu = [1 if a.get("sentiment") == "neutral" else 0 for a in d_arts]
            d_neg = [1 if a.get("sentiment") == "negative" else 0 for a in d_arts]
            count = len(d_arts)
            d_comp = sum(d_scores) / count
            pos_pct = sum(d_pos) / count
            neu_pct = sum(d_neu) / count
            neg_pct = sum(d_neg) / count
        else:
            d_comp = aggregate["overall_score"]
            pos_pct = aggregate["positive"]
            neu_pct = aggregate["neutral"]
            neg_pct = aggregate["negative"]
            count = aggregate["analyzed_count"]

        existing = (
            db.query(SentimentScore)
            .filter(SentimentScore.stock_id == stock.id, SentimentScore.score_date == d_val)
            .first()
        )
        if existing:
            existing.compound_score = round(d_comp, 4)
            existing.positive = round(pos_pct, 4)
            existing.neutral = round(neu_pct, 4)
            existing.negative = round(neg_pct, 4)
            existing.article_count = count
            existing.is_sample = is_sample
        else:
            db.add(
                SentimentScore(
                    stock_id=stock.id,
                    score_date=d_val,
                    compound_score=round(d_comp, 4),
                    positive=round(pos_pct, 4),
                    neutral=round(neu_pct, 4),
                    negative=round(neg_pct, 4),
                    article_count=count,
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

    if headlines:
        overall = sum(h.score for h in headlines) / len(headlines)
    elif daily_rows:
        overall = sum(r.compound_score for r in daily_rows) / len(daily_rows)
    else:
        overall = 0.0

    label = "positive" if overall > 0.05 else ("negative" if overall < -0.05 else "neutral")
    has_sample = any(r.is_sample for r in daily_rows) if daily_rows else any(h.is_sample for h in headlines)

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
