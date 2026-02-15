"""
News Sentiment Agent for FinSage

This agent is responsible for fetching financial news, analyzing sentiment,
and updating the news context with the latest information.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama

import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as pipeline_factory
import torch

from contexts.news_context import NewsContextContent, NewsArticle, CategorySentiment, TickerSentiment
from utils.mcp_utils import ContextWrapper, get_registry
from utils.api_clients import NewsAPIClient, YFinanceClient

# Sentiment pipeline for testing and internal use
# We initialize it as None and load it on demand to avoid errors during test collection
pipeline = None

# Configure logger
logger = logging.getLogger(__name__)

class NewsSentimentAgent:
    """Agent for analyzing financial news sentiment"""
    
    def __init__(self):
        """Initialize the news sentiment agent"""
        self.agent_name = "NewsSentimentAgent"
        self.model = ChatOllama(model="gemma3:4b")
        
        self.yfinance_client = YFinanceClient()
        
        global pipeline
        if pipeline is None:
            try:
                pipeline = pipeline_factory("sentiment-analysis")
            except Exception as e:
                logger.error(f"Error initializing sentiment pipeline: {e}")
        
        self.sentiment_analyzer = pipeline
        
        # Initialize sentiment model (FinBERT)
        try:
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.sentiment_labels = ["negative", "neutral", "positive"]
            self.sentiment_model_loaded = True
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            self.sentiment_model_loaded = False
        
        # Initialize or get the latest news context
        self.news_context = self._get_or_create_news_context()
        
        # Set up tools for the agent
        self.tools = self._create_tools()
        
        # Set up the agent executor
        self.agent_executor = self._create_agent_executor()
    
    def _get_or_create_news_context(self) -> ContextWrapper:
        """Get existing news context or create a new one"""
        registry = get_registry()
        context = registry.get_latest_context("news_context")
        
        if not context:
            # Create a new news context
            context_content = NewsContextContent()
            
            context = ContextWrapper.create(
                context_type="news_context",
                creator_agent=self.agent_name,
                content_model=NewsContextContent,
                content_data=context_content.dict()
            )
            
            # Register the new context
            registry.register_context(context)
        
        return context
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools and prompt"""
        # Set up the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial news sentiment analyst that helps users understand
            market sentiment by analyzing financial news. You can fetch the latest news,
            analyze sentiment, and identify trends in financial reporting.
            
            Your goal is to provide objective insights about market sentiment and news
            that might impact investment decisions.
            
            Always use data and analysis to support your conclusions.
            Format your responses in markdown for better readability.
            """  
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent
        agent = {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            )
        } | prompt | self.model | OpenAIFunctionsAgentOutputParser()
        
        # Create the agent executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def run(self, query: str) -> str:
        """Run the agent with a user query"""
        try:
            response = self.agent_executor.invoke({"input": query})
            return response["output"]
        except Exception as e:
            logger.error(f"Error running news sentiment agent: {e}")
            return f"Error analyzing news sentiment: {str(e)}"
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a text using FinBERT.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment score and label
        """
        if self.sentiment_analyzer:
            try:
                results = self.sentiment_analyzer(text)
                sentiment_data = results[0]
                sentiment_label = sentiment_data["label"].lower()
                sentiment_score = sentiment_data["score"]
                if sentiment_label == "negative":
                    sentiment_score = -sentiment_score
                elif sentiment_label == "neutral":
                    sentiment_score = 0
                    
                return {
                    "sentiment_score": float(sentiment_score),
                    "sentiment_label": sentiment_label,
                    "raw_scores": {
                        "negative": 1.0 if sentiment_label == "negative" else 0.0,
                        "neutral": 1.0 if sentiment_label == "neutral" else 0.0,
                        "positive": 1.0 if sentiment_label == "positive" else 0.0
                    }
                }
            except Exception as e:
                logger.error(f"Error using sentiment analyzer pipeline: {e}")

        if not self.sentiment_model_loaded:
            # Fallback to simple rule-based sentiment if model not loaded
            return self._simple_sentiment_analysis(text)
        
        try:
            # Preprocess text (truncate if needed to fit in model's max length)
            max_length = self.tokenizer.model_max_length
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            
            # Run inference
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
            
            # Get predicted sentiment
            sentiment_score = scores[2] - scores[0]  # positive - negative
            sentiment_label = self.sentiment_labels[scores.argmax()]
            
            return {
                "sentiment_score": float(sentiment_score),
                "sentiment_label": sentiment_label,
                "raw_scores": {
                    "negative": float(scores[0]),
                    "neutral": float(scores[1]),
                    "positive": float(scores[2])
                }
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._simple_sentiment_analysis(text)
    
    def _simple_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Simple rule-based sentiment analysis as fallback.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment score and label
        """
        # Simple keyword-based approach
        positive_words = [
            "growth", "profit", "bullish", "upside", "gain", "positive", "increase",
            "improved", "rally", "recovery", "opportunity", "beat", "exceeded", "up",
            "higher", "surged", "outperform", "strong", "success", "advantage"
        ]
        
        negative_words = [
            "decline", "loss", "bearish", "downside", "fall", "negative", "decrease",
            "downturn", "recession", "risk", "miss", "disappointing", "down", "lower",
            "dropped", "underperform", "weak", "failure", "disadvantage", "crash"
        ]
        
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        
        # Calculate sentiment
        total_count = positive_count + negative_count
        if total_count == 0:
            sentiment_score = 0.0
            sentiment_label = "neutral"
        else:
            sentiment_score = (positive_count - negative_count) / total_count
            if sentiment_score > 0.1:
                sentiment_label = "positive"
            elif sentiment_score < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "raw_scores": {
                "negative": negative_count / (total_count if total_count > 0 else 1),
                "neutral": 0,
                "positive": positive_count / (total_count if total_count > 0 else 1)
            }
        }
    
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for the agent to use"""
        tools = []
        
        @tool("fetch_latest_news")
        def fetch_latest_news(query: str = "", category: str = "business", max_results: int = 10) -> str:
            """
            Fetch the latest financial news based on query and category.
            
            Args:
                query: Search query for news (optional)
                category: News category (business, markets, economy, etc.)
                max_results: Maximum number of articles to fetch
            """
            try:
                # Fetch news from NewsAPI
                articles = self.news_client.get_news_for_query(
                    query=query,
                    category=category,
                    max_results=max_results
                )
                
                if not articles:
                    return f"No news articles found for query: {query} in category: {category}"
                
                # Format news report
                report = [f"## Latest Financial News: {query if query else category.title()}", ""]
                
                for i, article in enumerate(articles[:limit], 1):
                    # Add article to news context
                    article_id = f"{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    # Analyze sentiment if content is available
                    sentiment_data = {"sentiment_score": 0, "sentiment_label": "neutral"}
                    if article.get("content"):
                        sentiment_data = self.analyze_sentiment(article["content"])
                    elif article.get("description"):
                        sentiment_data = self.analyze_sentiment(article["description"])
                    
                    # Create article object for context
                    news_article = NewsArticle(
                        article_id=article_id,
                        title=article.get("title", "No Title"),
                        url=article.get("url", ""),
                        source=article.get("source", {}).get("name", "Unknown"),
                        published_at=article.get("publishedAt", datetime.now().isoformat()),
                        category=category,
                        content=article.get("content", ""),
                        description=article.get("description", ""),
                        sentiment_score=sentiment_data["sentiment_score"],
                        sentiment_label=sentiment_data["sentiment_label"],
                        tickers=[]  # Will be updated later
                    )
                    
                    # Add to news context
                    self.news_context.content.articles[article_id] = news_article
                    
                    # Format for report
                    source = article.get("source", {}).get("name", "Unknown Source")
                    published_at = article.get("publishedAt", "")
                    
                    if published_at:
                        try:
                            published_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                            published_at = published_date.strftime("%Y-%m-%d %H:%M")
                        except:
                            pass
                    
                    sentiment_emoji = "🟡"  # neutral
                    if sentiment_data["sentiment_label"] == "positive":
                        sentiment_emoji = "🟢"
                    elif sentiment_data["sentiment_label"] == "negative":
                        sentiment_emoji = "🔴"
                    
                    report.append(f"### {i}. {article.get('title', 'No Title')} {sentiment_emoji}")
                    report.append(f"**Source**: {source} | **Published**: {published_at}")
                    report.append(f"**Sentiment**: {sentiment_data['sentiment_label'].title()} ({sentiment_data['sentiment_score']:.2f})")
                    
                    if article.get("description"):
                        report.append(f"\n{article['description']}")
                    
                    report.append(f"\n[Read more]({article.get('url', '#')})\n")
                
                # Update the news context
                self._update_news_context()
                
                return "\n".join(report)
                
            except Exception as e:
                logger.error(f"Error fetching news: {e}")
                return f"Error fetching news: {str(e)}"
        
        @tool("analyze_news_sentiment")
        def analyze_news_sentiment(text: str) -> str:
            """
            Analyze the sentiment of a news article or text.
            
            Args:
                text: The text to analyze
            """
            try:
                sentiment_data = self.analyze_sentiment(text)
                
                # Format sentiment report
                sentiment_score = sentiment_data["sentiment_score"]
                sentiment_label = sentiment_data["sentiment_label"]
                
                report = ["## Sentiment Analysis Results", ""]
                
                # Overall sentiment
                sentiment_emoji = "🟡"  # neutral
                if sentiment_label == "positive":
                    sentiment_emoji = "🟢"
                elif sentiment_label == "negative":
                    sentiment_emoji = "🔴"
                
                report.append(f"**Overall Sentiment**: {sentiment_label.title()} {sentiment_emoji}")
                report.append(f"**Sentiment Score**: {sentiment_score:.2f} (-1.0 to 1.0 scale)")
                
                # Raw scores
                report.append("\n**Detailed Sentiment Breakdown**:")
                for label, score in sentiment_data["raw_scores"].items():
                    report.append(f"- {label.title()}: {score:.2f}")
                
                # Interpretation
                report.append("\n**Interpretation**:")
                if sentiment_label == "positive":
                    report.append("This text has a positive financial sentiment, suggesting optimistic or favorable market conditions, growth prospects, or positive company performance.")
                elif sentiment_label == "negative":
                    report.append("This text has a negative financial sentiment, suggesting pessimistic or unfavorable market conditions, decline, or negative company performance.")
                else:
                    report.append("This text has a neutral financial sentiment, suggesting balanced, factual reporting without strong positive or negative implications.")
                
                # Key sentiment indicators
                report.append("\n**Key Sentiment Indicators**:")
                
                # Simple keyword extraction for demonstration
                text_lower = text.lower()
                positive_indicators = [
                    "growth", "profit", "bullish", "upside", "gain", "increase",
                    "improved", "rally", "recovery", "beat", "exceeded", "outperform"
                ]
                
                negative_indicators = [
                    "decline", "loss", "bearish", "downside", "fall", "decrease",
                    "downturn", "recession", "risk", "miss", "disappointing", "underperform"
                ]
                
                found_positive = [word for word in positive_indicators if word in text_lower]
                found_negative = [word for word in negative_indicators if word in text_lower]
                
                if found_positive:
                    report.append(f"- Positive indicators: {', '.join(found_positive)}")
                
                if found_negative:
                    report.append(f"- Negative indicators: {', '.join(found_negative)}")
                
                if not found_positive and not found_negative:
                    report.append("- No strong sentiment indicators detected")
                
                return "\n".join(report)
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment: {e}")
                return f"Error analyzing sentiment: {str(e)}"
        
        @tool("get_market_sentiment_summary")
        def get_market_sentiment_summary() -> str:
            """
            Get a summary of current market sentiment based on recent news.
            """
            try:
                # Get news context
                news_context = self.news_context.content
                
                if not news_context.articles:
                    return "No news articles available for sentiment analysis. Try fetching latest news first."
                
                # Calculate overall market sentiment
                sentiment_scores = [article.sentiment_score for article in news_context.articles.values()]
                overall_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                
                # Get sentiment by category
                category_sentiments = {}
                for article in news_context.articles.values():
                    category = article.category.lower() if article.category else "uncategorized"
                    if category not in category_sentiments:
                        category_sentiments[category] = []
                    
                    category_sentiments[category].append(article.sentiment_score)
                
                # Get sentiment by ticker if available
                ticker_sentiments = {}
                for article in news_context.articles.values():
                    for ticker in article.tickers:
                        if ticker not in ticker_sentiments:
                            ticker_sentiments[ticker] = []
                        
                        ticker_sentiments[ticker].append(article.sentiment_score)
                
                # Format the report
                report = ["## Market Sentiment Summary", ""]
                
                # Overall sentiment
                sentiment_label = "Neutral"
                sentiment_emoji = "🟡"
                if overall_sentiment > 0.2:
                    sentiment_label = "Positive"
                    sentiment_emoji = "🟢"
                elif overall_sentiment < -0.2:
                    sentiment_label = "Negative"
                    sentiment_emoji = "🔴"
                
                report.append(f"**Overall Market Sentiment**: {sentiment_label} {sentiment_emoji}")
                report.append(f"**Sentiment Score**: {overall_sentiment:.2f} (-1.0 to 1.0 scale)")
                report.append(f"**Based on**: {len(news_context.articles)} news articles")
                
                # Sentiment by category
                if category_sentiments:
                    report.append("\n### Sentiment by Category")
                    
                    for category, scores in category_sentiments.items():
                        avg_score = sum(scores) / len(scores) if scores else 0
                        cat_sentiment = "Neutral"
                        cat_emoji = "🟡"
                        
                        if avg_score > 0.2:
                            cat_sentiment = "Positive"
                            cat_emoji = "🟢"
                        elif avg_score < -0.2:
                            cat_sentiment = "Negative"
                            cat_emoji = "🔴"
                        
                        report.append(f"- **{category.title()}**: {cat_sentiment} {cat_emoji} ({avg_score:.2f})")
                
                # Sentiment by ticker
                if ticker_sentiments:
                    report.append("\n### Sentiment by Ticker")
                    
                    # Sort tickers by sentiment score
                    sorted_tickers = sorted(
                        ticker_sentiments.items(),
                        key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0,
                        reverse=True
                    )
                    
                    for ticker, scores in sorted_tickers:
                        avg_score = sum(scores) / len(scores) if scores else 0
                        ticker_sentiment = "Neutral"
                        ticker_emoji = "🟡"
                        
                        if avg_score > 0.2:
                            ticker_sentiment = "Positive"
                            ticker_emoji = "🟢"
                        elif avg_score < -0.2:
                            ticker_sentiment = "Negative"
                            ticker_emoji = "🔴"
                        
                        report.append(f"- **{ticker}**: {ticker_sentiment} {ticker_emoji} ({avg_score:.2f})")
                
                # Market trends and insights
                report.append("\n### Market Insights")
                
                if overall_sentiment > 0.3:
                    report.append("- Market news is predominantly positive, suggesting optimistic investor sentiment")
                    report.append("- Consider monitoring for potential market exuberance or overvaluation")
                elif overall_sentiment > 0:
                    report.append("- Market news is moderately positive, indicating cautious optimism")
                    report.append("- Look for sectors with strong positive sentiment as potential opportunities")
                elif overall_sentiment > -0.3:
                    report.append("- Market news is balanced to slightly negative, suggesting cautious investor sentiment")
                    report.append("- Focus on quality assets and consider defensive positioning")
                else:
                    report.append("- Market news is predominantly negative, indicating pessimistic investor sentiment")
                    report.append("- High negativity can sometimes indicate potential overselling or contrarian opportunities")
                
                # Sentiment trend if available
                if hasattr(news_context, "sentiment_trend") and news_context.sentiment_trend:
                    trend_direction = "stable"
                    if news_context.sentiment_trend > 0.1:
                        trend_direction = "improving"
                    elif news_context.sentiment_trend < -0.1:
                        trend_direction = "deteriorating"
                    
                    report.append(f"\n**Sentiment Trend**: {trend_direction.title()} ({news_context.sentiment_trend:.2f})")
                
                # Update news context with summary
                self._update_news_context()
                
                return "\n".join(report)
                
            except Exception as e:
                logger.error(f"Error generating sentiment summary: {e}")
                return f"Error generating sentiment summary: {str(e)}"
        
        @tool("analyze_ticker_news")
        def analyze_ticker_news(ticker: str, days: int = 7) -> str:
            """
            Analyze news sentiment for a specific ticker symbol.
            
            Args:
                ticker: Stock ticker symbol (e.g., AAPL, MSFT)
                days: Number of days of news to analyze
            """
            try:
                # Get company info
                company_info = self.yfinance_client.get_ticker_info(ticker)
                company_name = company_info.get("shortName", ticker) if company_info else ticker
                
                # Fetch news for the ticker
                news_articles = self.news_client.get_news(
                    query=f"{company_name} OR {ticker}",
                    days=days,
                    limit=20
                )
                
                if not news_articles:
                    return f"No news articles found for {ticker} ({company_name}) in the last {days} days."
                
                # Analyze sentiment for each article
                ticker_articles = []
                
                for article in news_articles:
                    # Analyze sentiment
                    text_to_analyze = article.get("content", article.get("description", ""))
                    if not text_to_analyze:
                        continue
                    
                    sentiment_data = self.analyze_sentiment(text_to_analyze)
                    
                    # Create article object
                    article_id = f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(ticker_articles)}"
                    
                    news_article = NewsArticle(
                        article_id=article_id,
                        title=article.get("title", "No Title"),
                        url=article.get("url", ""),
                        source=article.get("source", {}).get("name", "Unknown"),
                        published_at=article.get("publishedAt", datetime.now().isoformat()),
                        category="company",
                        content=article.get("content", ""),
                        description=article.get("description", ""),
                        sentiment_score=sentiment_data["sentiment_score"],
                        sentiment_label=sentiment_data["sentiment_label"],
                        tickers=[ticker]
                    )
                    
                    ticker_articles.append(news_article)
                    
                    # Add to news context
                    self.news_context.content.articles[article_id] = news_article
                
                # Calculate average sentiment
                sentiment_scores = [article.sentiment_score for article in ticker_articles]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                
                # Determine sentiment label
                sentiment_label = "Neutral"
                sentiment_emoji = "🟡"
                if avg_sentiment > 0.2:
                    sentiment_label = "Positive"
                    sentiment_emoji = "🟢"
                elif avg_sentiment < -0.2:
                    sentiment_label = "Negative"
                    sentiment_emoji = "🔴"
                
                # Format the report
                report = [f"## News Sentiment Analysis: {ticker} ({company_name})", ""]
                
                # Overall sentiment
                report.append(f"**Overall Sentiment**: {sentiment_label} {sentiment_emoji}")
                report.append(f"**Sentiment Score**: {avg_sentiment:.2f} (-1.0 to 1.0 scale)")
                report.append(f"**Based on**: {len(ticker_articles)} news articles from the last {days} days")
                
                # Add ticker sentiment to news context
                ticker_sentiment = TickerSentiment(
                    ticker=ticker,
                    company_name=company_name,
                    sentiment_score=avg_sentiment,
                    sentiment_label=sentiment_label,
                    article_count=len(ticker_articles),
                    last_updated=datetime.now()
                )
                
                self.news_context.content.ticker_sentiments[ticker] = ticker_sentiment
                
                # Top positive and negative articles
                report.append("\n### Most Significant News")
                
                # Sort by absolute sentiment score to get most significant articles
                significant_articles = sorted(
                    ticker_articles,
                    key=lambda x: abs(x.sentiment_score),
                    reverse=True
                )[:5]
                
                for i, article in enumerate(significant_articles, 1):
                    sentiment_emoji = "🟡"  # neutral
                    if article.sentiment_label == "positive":
                        sentiment_emoji = "🟢"
                    elif article.sentiment_label == "negative":
                        sentiment_emoji = "🔴"
                    
                    published_at = article.published_at
                    if published_at:
                        try:
                            published_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                            published_at = published_date.strftime("%Y-%m-%d")
                        except:
                            pass
                    
                    report.append(f"### {i}. {article.title} {sentiment_emoji}")
                    report.append(f"**Source**: {article.source} | **Date**: {published_at}")
                    report.append(f"**Sentiment**: {article.sentiment_label.title()} ({article.sentiment_score:.2f})")
                    
                    if article.description:
                        report.append(f"\n{article.description}")
                    
                    report.append(f"\n[Read more]({article.url})\n")
                
                # Market implications
                report.append("### Market Implications")
                
                if avg_sentiment > 0.3:
                    report.append("- News sentiment for this company is strongly positive")
                    report.append("- This often correlates with positive price movement, though market efficiency may already price in public news")
                    report.append("- Consider analyzing volume and price action alongside this sentiment data")
                elif avg_sentiment > 0:
                    report.append("- News sentiment for this company is moderately positive")
                    report.append("- The company is receiving generally favorable coverage")
                    report.append("- Monitor for changes in sentiment direction as news develops")
                elif avg_sentiment > -0.3:
                    report.append("- News sentiment for this company is neutral to slightly negative")
                    report.append("- This may indicate mixed news or balanced reporting")
                    report.append("- Look for specific news catalysts that could shift sentiment more decisively")
                else:
                    report.append("- News sentiment for this company is strongly negative")
                    report.append("- Negative news may create short-term pressure on the stock")
                    report.append("- Consider whether negative news represents a temporary setback or fundamental issues")
                
                # Update news context
                self._update_news_context()
                
                return "\n".join(report)
                
            except Exception as e:
                logger.error(f"Error analyzing ticker news: {e}")
                return f"Error analyzing ticker news for {ticker}: {str(e)}"
        
        # Add all tools to the list
        tools.append(fetch_latest_news)
        tools.append(analyze_news_sentiment)
        tools.append(get_market_sentiment_summary)
        tools.append(analyze_ticker_news)
        
        return tools
    
    def _update_news_context(self):
        """Update the news context with the latest sentiment analysis"""
        try:
            # Calculate overall market sentiment
            sentiment_scores = [article.sentiment_score for article in self.news_context.content.articles.values()]
            self.news_context.content.market_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # Update category sentiments
            category_sentiments = {}
            for article in self.news_context.content.articles.values():
                category = article.category.lower() if article.category else "uncategorized"
                if category not in category_sentiments:
                    category_sentiments[category] = []
                
                category_sentiments[category].append(article.sentiment_score)
            
            for category, scores in category_sentiments.items():
                avg_score = sum(scores) / len(scores) if scores else 0
                sentiment_label = "neutral"
                if avg_score > 0.2:
                    sentiment_label = "positive"
                elif avg_score < -0.2:
                    sentiment_label = "negative"
                
                cat_sentiment = CategorySentiment(
                    category=category,
                    sentiment_score=avg_score,
                    sentiment_label=sentiment_label,
                    article_count=len(scores),
                    last_updated=datetime.now()
                )
                
                self.news_context.content.category_sentiments[category] = cat_sentiment
            
            # Calculate sentiment trend
            current_sentiment = self.news_context.content.market_sentiment
            previous_sentiment = getattr(self.news_context.content, "previous_market_sentiment", current_sentiment)
            
            self.news_context.content.sentiment_trend = current_sentiment - previous_sentiment
            self.news_context.content.previous_market_sentiment = current_sentiment
            
            # Remove old articles (keeping only last 100)
            if len(self.news_context.content.articles) > 100:
                # Sort by date and keep only most recent
                sorted_articles = sorted(
                    self.news_context.content.articles.items(),
                    key=lambda x: x[1].published_at if x[1].published_at else "",
                    reverse=True
                )
                
                self.news_context.content.articles = {k: v for k, v in sorted_articles[:100]}
            
            # Update the context in the registry
            self.news_context.update(
                updated_by=self.agent_name,
                content_updates=self.news_context.content.dict()
            )
            
            registry = get_registry()
            registry.register_context(self.news_context)
            
        except Exception as e:
            logger.error(f"Error updating news context: {e}")
