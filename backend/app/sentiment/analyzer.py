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
    Fetches real live news for US and Indian stocks without requiring an API key.
    """
    symbol = symbol.upper()

    # 1. NewsAPI (if configured with API key)
    if settings.NEWS_PROVIDER == "newsapi" and settings.NEWS_API_KEY:
        try:
            arts = _fetch_newsapi(symbol, days)
            if arts:
                return arts, False
        except Exception as exc:
            logger.warning("NewsAPI failed — falling back to live web news", error=str(exc))

    # 2. Live Free Financial News (Yahoo Finance Search & Google News RSS)
    try:
        live_arts = _fetch_live_financial_news(symbol, days)
        if live_arts:
            logger.info("Fetched live financial news", symbol=symbol, count=len(live_arts))
            return live_arts, False
    except Exception as exc:
        logger.warning("Live web news fetch failed — using dynamic sample data", symbol=symbol, error=str(exc))

    # 3. Dynamic fallback if offline
    return _get_sample_news(symbol), True


def _fetch_live_financial_news(symbol: str, days: int = 7) -> list[dict]:
    """
    Fetch free, real financial news headlines for US or Indian stocks.
    Uses Yahoo Finance Search API & Google News RSS with zero API keys required.
    """
    import requests
    import xml.etree.ElementTree as ET

    articles = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    clean_sym = symbol.split(".")[0].upper()

    # Try Yahoo Finance search news
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol}&newsCount=15"
        res = requests.get(url, headers=headers, timeout=6)
        if res.status_code == 200:
            items = res.json().get("news", [])
            for it in items:
                title = it.get("title")
                if title:
                    pub_ts = it.get("providerPublishTime")
                    pub_dt = datetime.fromtimestamp(pub_ts).isoformat() if pub_ts else datetime.utcnow().isoformat()
                    articles.append({
                        "title": title,
                        "description": it.get("summary") or it.get("publisher") or "",
                        "url": it.get("link") or "#",
                        "source": it.get("publisher") or "Yahoo Finance",
                        "published_at": pub_dt,
                        "is_sample": False,
                    })
    except Exception:
        pass

    # If fewer than 5 headlines (e.g. for Indian stocks or tickers with no Yahoo feed), use Google News RSS
    if len(articles) < 5:
        try:
            is_indian = symbol.endswith(".NS") or symbol.endswith(".BO")
            gl = "IN" if is_indian else "US"
            hl = "en-IN" if is_indian else "en-US"
            ceid = "IN:en" if is_indian else "US:en"
            query = f"{clean_sym}+share+price+stock" if is_indian else f"{clean_sym}+stock+news"
            rss_url = f"https://news.google.com/rss/search?q={query}&hl={hl}&gl={gl}&ceid={ceid}"
            res = requests.get(rss_url, headers=headers, timeout=6)
            if res.status_code == 200:
                root = ET.fromstring(res.content)
                for item in root.findall("./channel/item")[:15]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    source_elem = item.find("source")
                    source = source_elem.text if source_elem is not None else "Google News"
                    link = item.find("link").text if item.find("link") is not None else "#"
                    if title:
                        try:
                            import email.utils
                            parsed_tuple = email.utils.parsedate_to_datetime(pub_date)
                            iso_date = parsed_tuple.isoformat()
                        except Exception:
                            iso_date = datetime.utcnow().isoformat()
                        articles.append({
                            "title": title,
                            "description": "",
                            "url": link,
                            "source": source,
                            "published_at": iso_date,
                            "is_sample": False,
                        })
        except Exception:
            pass

    return articles


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
    Explicitly labeled sample/demo headlines tailored per symbol across multiple dates.
    """
    now = datetime.utcnow()
    clean_sym = symbol.split(".")[0].upper()
    return [
        {
            "title": f"[SAMPLE] {clean_sym} reports strong quarterly operating margins, beating Wall Street estimates",
            "description": f"Quarterly financial results for {clean_sym} show steady revenue growth and disciplined cost management.",
            "url": "#",
            "source": "Financial Digest",
            "published_at": (now - timedelta(hours=3)).isoformat(),
            "is_sample": True,
        },
        {
            "title": f"[SAMPLE] Market analysts adjust price target for {clean_sym} amid expanding enterprise adoption",
            "description": f"Brokerage note highlights strategic positioning for {clean_sym} in key market segments.",
            "url": "#",
            "source": "Market Watchers",
            "published_at": (now - timedelta(days=1)).isoformat(),
            "is_sample": True,
        },
        {
            "title": f"[SAMPLE] {clean_sym} encounters supply chain constraints and higher input costs in recent quarter",
            "description": f"Industry-wide logistics pressures pose temporary margin headwinds for {clean_sym}.",
            "url": "#",
            "source": "Industry Insights",
            "published_at": (now - timedelta(days=2)).isoformat(),
            "is_sample": True,
        },
        {
            "title": f"[SAMPLE] {clean_sym} unveils new product ecosystem and strategic technology integration",
            "description": f"New initiative aims to accelerate multi-year modernization and client retention for {clean_sym}.",
            "url": "#",
            "source": "Tech & Trade News",
            "published_at": (now - timedelta(days=3)).isoformat(),
            "is_sample": True,
        },
        {
            "title": f"[SAMPLE] Macro headwinds and sector-wide volatility trigger cautious near-term outlook for {clean_sym}",
            "description": f"Broader rate environment and sector rotation impact equity valuations across peer group including {clean_sym}.",
            "url": "#",
            "source": "Global Economic Review",
            "published_at": (now - timedelta(days=5)).isoformat(),
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
