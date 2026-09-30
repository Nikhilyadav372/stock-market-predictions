"""
Financial News Sentiment Analysis

Uses FinBERT (ProsusAI/finbert) — a BERT model fine-tuned on financial text
from Hugging Face. This provides domain-appropriate sentiment (positive/neutral/negative)
unlike general-purpose sentiment models.

If the Hugging Face model is unavailable (no internet, firewall),
a simple lexicon-based fallback is used. Both are clearly labelled.

News sources:
  - NewsAPI (newsapi.org) if NEWS_API_KEY is set
  - Labeled sample headlines if not configured (NEWS_PROVIDER=sample)
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Lazy-loaded model to avoid startup delay
_finbert_pipeline = None


def _load_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline

    try:
        from transformers import pipeline
        logger.info("Loading FinBERT sentiment model (this may take a moment on first run)...")
        _finbert_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,  # return all class scores
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded successfully")
        return _finbert_pipeline
    except Exception as exc:
        logger.warning("FinBERT unavailable — using lexicon fallback", error=str(exc))
        return None


# ─── Lexicon Fallback ─────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    "surge", "surged", "surges", "soar", "soared", "soars", "rally", "rallied",
    "beat", "beats", "exceed", "exceeded", "exceeds", "profit", "profits",
    "growth", "gain", "gains", "record", "high", "upgrade", "upgraded",
    "bullish", "strong", "positive", "optimistic", "rise", "rises", "rose",
    "jumped", "climbed", "breakthrough", "partnership", "revenue", "revenues",
    "outperform", "outperformed", "recover", "recovered", "expansion",
}
NEGATIVE_WORDS = {
    "fall", "falls", "fell", "drop", "drops", "dropped", "plunge", "plunged",
    "miss", "missed", "loss", "losses", "decline", "declined", "declines",
    "weak", "weakness", "bearish", "downgrade", "downgraded", "concern",
    "concerns", "risk", "risks", "crash", "crashed", "sell", "cut", "cuts",
    "warning", "warnings", "recession", "lawsuit", "investigation",
    "debt", "layoff", "layoffs", "bankruptcy", "volatile", "volatility",
    "massive", "plummeted", "tumbled", "slump", "slumped", "collapse",
    "collapsed", "default", "downfall", "disappointing", "missed",
}


def _lexicon_sentiment(text: str) -> dict:
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    pos_count = len(tokens & POSITIVE_WORDS)
    neg_count = len(tokens & NEGATIVE_WORDS)
    total = pos_count + neg_count + 1  # +1 to avoid div by zero
    pos_score = pos_count / total
    neg_score = neg_count / total
    neu_score = 1.0 - pos_score - neg_score
    if pos_score > neg_score and pos_score > neu_score:
        label = "positive"
    elif neg_score > pos_score and neg_score > neu_score:
        label = "negative"
    else:
        label = "neutral"
    compound = pos_score - neg_score
    return {"label": label, "positive": pos_score, "negative": neg_score, "neutral": neu_score, "compound": compound}


def analyze_text(text: str) -> dict:
    """
    Returns sentiment dict:
    {label, positive, negative, neutral, compound}
    Uses FinBERT if available, falls back to lexicon.
    """
    if not text or len(text.strip()) < 3:
        return {"label": "neutral", "positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}

    pipe = _load_finbert()
    if pipe is not None:
        try:
            result = pipe(text[:512])[0]  # list of {label, score} dicts
            scores = {r["label"].lower(): r["score"] for r in result}
            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral", 0.0)
            compound = pos - neg
            label = max(scores, key=scores.get)
            return {"label": label, "positive": pos, "negative": neg, "neutral": neu, "compound": compound}
        except Exception as exc:
            logger.warning("FinBERT inference failed", error=str(exc))

    return _lexicon_sentiment(text)


# ─── News Fetching ────────────────────────────────────────────────────────────

def fetch_news(symbol: str, days: int = 7) -> tuple[list[dict], bool]:
    """
    Returns (articles_list, is_sample).
    Each article: {title, description, url, source, published_at, is_sample}
    is_sample=True ALWAYS means the articles are not real live news.
    """
    if settings.NEWS_PROVIDER == "newsapi" and settings.NEWS_API_KEY:
        try:
            return _fetch_newsapi(symbol, days), False
        except Exception as exc:
            logger.warning("NewsAPI failed — using sample data", error=str(exc))

    logger.info("Using labeled sample news data", symbol=symbol)
    return _get_sample_news(symbol), True


def _fetch_newsapi(symbol: str, days: int) -> list[dict]:
    import requests

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": symbol,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": settings.NEWS_API_KEY,
            "pageSize": 30,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    articles = []
    for a in data.get("articles", []):
        articles.append({
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "url": a.get("url", ""),
            "source": a.get("source", {}).get("name", ""),
            "published_at": a.get("publishedAt"),
            "is_sample": False,
        })
    return articles


def _get_sample_news(symbol: str) -> list[dict]:
    """
    Explicitly labeled sample/demo headlines.
    These are NOT real news and are NEVER presented as real news.
    """
    now = datetime.utcnow()
    return [
        {
            "title": f"[SAMPLE] {symbol} reports strong quarterly earnings, beating estimates",
            "description": "Sample data: Company beat earnings expectations for the quarter.",
            "url": "#",
            "source": "SAMPLE DATA",
            "published_at": (now - timedelta(hours=2)).isoformat(),
            "is_sample": True,
        },
        {
            "title": f"[SAMPLE] Analysts upgrade {symbol} price target amid market optimism",
            "description": "Sample data: Multiple analysts raised their price targets.",
            "url": "#",
            "source": "SAMPLE DATA",
            "published_at": (now - timedelta(hours=8)).isoformat(),
            "is_sample": True,
        },
        {
            "title": f"[SAMPLE] {symbol} faces regulatory scrutiny in new market",
            "description": "Sample data: Regulatory concerns have emerged in new segment.",
            "url": "#",
            "source": "SAMPLE DATA",
            "published_at": (now - timedelta(days=1)).isoformat(),
            "is_sample": True,
        },
        {
            "title": f"[SAMPLE] {symbol} announces strategic partnership for AI integration",
            "description": "Sample data: New partnership announced for technology integration.",
            "url": "#",
            "source": "SAMPLE DATA",
            "published_at": (now - timedelta(days=2)).isoformat(),
            "is_sample": True,
        },
        {
            "title": f"[SAMPLE] Broader market volatility impacts {symbol} stock movement",
            "description": "Sample data: Market-wide volatility affected the stock.",
            "url": "#",
            "source": "SAMPLE DATA",
            "published_at": (now - timedelta(days=3)).isoformat(),
            "is_sample": True,
        },
    ]


# ─── Aggregate Sentiment ──────────────────────────────────────────────────────

def compute_aggregate_sentiment(articles: list[dict]) -> dict:
    """
    Run sentiment analysis on all articles and return aggregated scores.
    """
    if not articles:
        return {
            "overall_score": 0.0,
            "overall_label": "neutral",
            "positive": 0.0,
            "neutral": 1.0,
            "negative": 0.0,
            "analyzed_count": 0,
        }

    scores = []
    for art in articles:
        text = f"{art.get('title', '')} {art.get('description', '')}".strip()
        sent = analyze_text(text)
        art["sentiment"] = sent["label"]
        art["sentiment_score"] = sent["compound"]
        scores.append(sent)

    avg_pos = sum(s["positive"] for s in scores) / len(scores)
    avg_neg = sum(s["negative"] for s in scores) / len(scores)
    avg_neu = sum(s["neutral"] for s in scores) / len(scores)
    avg_compound = sum(s["compound"] for s in scores) / len(scores)

    if avg_compound > 0.05:
        label = "positive"
    elif avg_compound < -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "overall_score": avg_compound,
        "overall_label": label,
        "positive": avg_pos,
        "neutral": avg_neu,
        "negative": avg_neg,
        "analyzed_count": len(scores),
    }
