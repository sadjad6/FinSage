"""
News Context Module for FinSage

This module defines the structure for financial news data including articles,
sources, and sentiment analysis results.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field, HttpUrl

class SentimentLevel(str, Enum):
    """Sentiment levels for news articles"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

class SentimentAnalysis(BaseModel):
    """Structure for sentiment analysis results"""
    sentiment: SentimentLevel = SentimentLevel.NEUTRAL
    sentiment_score: float = 0.0  # -1.0 to 1.0
    confidence: float = 0.0  # 0.0 to 1.0
    key_phrases: List[str] = Field(default_factory=list)
    entities: Dict[str, List[str]] = Field(default_factory=dict)

class NewsArticle(BaseModel):
    """Structure for individual news articles"""
    article_id: str
    title: str
    url: HttpUrl
    source: str
    author: Optional[str] = None
    published_at: datetime
    retrieved_at: datetime = Field(default_factory=datetime.now)
    content: Optional[str] = None
    summary: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    categories: List[str] = Field(default_factory=list)
    tickers: List[str] = Field(default_factory=list)
    sentiment: Optional[SentimentAnalysis] = None
    read_status: bool = False
    flagged: bool = False
    tags: List[str] = Field(default_factory=list)

class CategorySentiment(BaseModel):
    """Aggregated sentiment for a news category"""
    category: str
    article_count: int = 0
    avg_sentiment_score: float = 0.0
    sentiment_distribution: Dict[SentimentLevel, int] = Field(default_factory=dict)
    recent_articles: List[str] = Field(default_factory=list)  # List of article_ids

class TickerSentiment(BaseModel):
    """Aggregated sentiment for a specific ticker/symbol"""
    ticker: str
    article_count: int = 0
    avg_sentiment_score: float = 0.0
    sentiment_distribution: Dict[SentimentLevel, int] = Field(default_factory=dict)
    sentiment_trend: List[Dict[str, float]] = Field(default_factory=list)
    recent_articles: List[str] = Field(default_factory=list)  # List of article_ids

class NewsContextContent(BaseModel):
    """Content structure for news context"""
    last_updated: datetime = Field(default_factory=datetime.now)
    
    # News articles indexed by article_id
    articles: Dict[str, NewsArticle] = Field(default_factory=dict)
    
    # Recent articles by publication date
    recent_articles: List[str] = Field(default_factory=list)  # List of article_ids
    
    # Aggregated sentiment by category
    category_sentiment: Dict[str, CategorySentiment] = Field(default_factory=dict)
    
    # Aggregated sentiment by ticker
    ticker_sentiment: Dict[str, TickerSentiment] = Field(default_factory=dict)
    
    # Overall market sentiment derived from news
    overall_sentiment: float = 0.0  # -1.0 to 1.0
    
    # Breaking news flags
    breaking_news: List[str] = Field(default_factory=list)  # List of article_ids
    
    def add_article(self, article: NewsArticle) -> None:
        """Add a news article to the context and update related metrics"""
        # Store the article
        self.articles[article.article_id] = article
        
        # Update recent articles (keep only the most recent 100)
        self.recent_articles.insert(0, article.article_id)
        self.recent_articles = self.recent_articles[:100]
        
        # Update category sentiment
        for category in article.categories:
            if category not in self.category_sentiment:
                self.category_sentiment[category] = CategorySentiment(category=category)
            
            cat_sentiment = self.category_sentiment[category]
            cat_sentiment.article_count += 1
            
            # Update sentiment distribution if sentiment analysis is available
            if article.sentiment:
                if article.sentiment.sentiment not in cat_sentiment.sentiment_distribution:
                    cat_sentiment.sentiment_distribution[article.sentiment.sentiment] = 0
                
                cat_sentiment.sentiment_distribution[article.sentiment.sentiment] += 1
                
                # Update average sentiment score
                total_score = cat_sentiment.avg_sentiment_score * (cat_sentiment.article_count - 1)
                total_score += article.sentiment.sentiment_score
                cat_sentiment.avg_sentiment_score = total_score / cat_sentiment.article_count
            
            # Add to recent articles
            cat_sentiment.recent_articles.insert(0, article.article_id)
            cat_sentiment.recent_articles = cat_sentiment.recent_articles[:20]  # Keep only 20 most recent
        
        # Update ticker sentiment
        for ticker in article.tickers:
            if ticker not in self.ticker_sentiment:
                self.ticker_sentiment[ticker] = TickerSentiment(ticker=ticker)
            
            ticker_sent = self.ticker_sentiment[ticker]
            ticker_sent.article_count += 1
            
            # Update sentiment distribution if sentiment analysis is available
            if article.sentiment:
                if article.sentiment.sentiment not in ticker_sent.sentiment_distribution:
                    ticker_sent.sentiment_distribution[article.sentiment.sentiment] = 0
                
                ticker_sent.sentiment_distribution[article.sentiment.sentiment] += 1
                
                # Update average sentiment score
                total_score = ticker_sent.avg_sentiment_score * (ticker_sent.article_count - 1)
                total_score += article.sentiment.sentiment_score
                ticker_sent.avg_sentiment_score = total_score / ticker_sent.article_count
                
                # Add to sentiment trend
                ticker_sent.sentiment_trend.append({
                    "date": article.published_at.isoformat(),
                    "sentiment": article.sentiment.sentiment_score
                })
            
            # Add to recent articles
            ticker_sent.recent_articles.insert(0, article.article_id)
            ticker_sent.recent_articles = ticker_sent.recent_articles[:20]  # Keep only 20 most recent
        
        # Recalculate overall market sentiment
        self._recalculate_overall_sentiment()
        
        # Update last_updated timestamp
        self.last_updated = datetime.now()
    
    def _recalculate_overall_sentiment(self) -> None:
        """Recalculate the overall market sentiment based on recent articles"""
        if not self.recent_articles:
            self.overall_sentiment = 0.0
            return
        
        # Calculate weighted average of sentiment scores from recent articles
        total_score = 0.0
        total_weight = 0.0
        
        for i, article_id in enumerate(self.recent_articles[:50]):  # Consider up to 50 recent articles
            article = self.articles.get(article_id)
            if article and article.sentiment:
                # More recent articles have higher weight
                weight = 1.0 / (i + 1)
                total_score += article.sentiment.sentiment_score * weight
                total_weight += weight
        
        if total_weight > 0:
            self.overall_sentiment = total_score / total_weight
        else:
            self.overall_sentiment = 0.0
    
    def get_sentiment_for_ticker(self, ticker: str) -> Optional[TickerSentiment]:
        """Get sentiment information for a specific ticker"""
        return self.ticker_sentiment.get(ticker.upper())
    
    def get_sentiment_for_category(self, category: str) -> Optional[CategorySentiment]:
        """Get sentiment information for a specific news category"""
        return self.category_sentiment.get(category.lower())
    
    def get_recent_articles_for_ticker(self, ticker: str, limit: int = 10) -> List[NewsArticle]:
        """Get recent news articles for a specific ticker"""
        ticker_sent = self.ticker_sentiment.get(ticker.upper())
        if not ticker_sent:
            return []
        
        articles = []
        for article_id in ticker_sent.recent_articles[:limit]:
            if article_id in self.articles:
                articles.append(self.articles[article_id])
        
        return articles
