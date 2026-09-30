"""
Unit tests for the sentiment analyzer.
Tests lexicon fallback and aggregation.
"""
from app.sentiment.analyzer import _lexicon_sentiment, analyze_text, compute_aggregate_sentiment


class TestLexiconSentiment:
    def test_positive_text(self):
        result = _lexicon_sentiment("The company reported record profit and strong revenue growth")
        assert result["label"] == "positive"
        assert result["compound"] > 0

    def test_negative_text(self):
        result = _lexicon_sentiment("The stock crashed and the company reported a massive loss")
        assert result["label"] == "negative"
        assert result["compound"] < 0

    def test_neutral_text(self):
        result = _lexicon_sentiment("The company released its quarterly report today")
        assert result["label"] in ("neutral", "positive", "negative")

    def test_scores_sum_to_one(self):
        result = _lexicon_sentiment("Stock market analysis")
        total = result["positive"] + result["negative"] + result["neutral"]
        assert abs(total - 1.0) < 0.01

    def test_empty_text(self):
        result = analyze_text("")
        assert result["label"] == "neutral"


class TestAggregatesSentiment:
    def test_empty_articles(self):
        result = compute_aggregate_sentiment([])
        assert result["overall_label"] == "neutral"
        assert result["analyzed_count"] == 0

    def test_single_article(self):
        articles = [{"title": "Company reports strong earnings growth", "description": ""}]
        result = compute_aggregate_sentiment(articles)
        assert "overall_score" in result
        assert result["analyzed_count"] == 1

    def test_multiple_articles(self):
        articles = [
            {"title": "Stock surges on strong earnings", "description": ""},
            {"title": "Company faces regulatory concerns", "description": ""},
            {"title": "Quarterly report released", "description": ""},
        ]
        result = compute_aggregate_sentiment(articles)
        assert result["analyzed_count"] == 3
        assert -1.0 <= result["overall_score"] <= 1.0
