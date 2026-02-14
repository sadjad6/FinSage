"""
API Clients Utility for FinSage

This module provides clients for various financial and news APIs
used by the agents to gather real-time data.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json
import requests

import yfinance as yf
from pycoingecko import CoinGeckoAPI
from newsapi import NewsApiClient
import finnhub
from dotenv import load_dotenv

from contexts.market_context import AssetType, AssetData

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class FinancialDataClient:
    """Base client for financial data APIs"""
    
    def __init__(self):
        """Initialize the financial data client"""
        # Initialize API clients
        self.finnhub_client = None
        self.news_api_client = None
        self.coingecko_client = None
        
        # Initialize API keys
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
        # Set up clients if API keys are available
        if self.finnhub_api_key:
            self.finnhub_client = finnhub.Client(api_key=self.finnhub_api_key)
        
        if self.news_api_key:
            self.news_api_client = NewsApiClient(api_key=self.news_api_key)
        
        # CoinGecko doesn't require an API key for basic usage
        self.coingecko_client = CoinGeckoAPI()
    
    def get_stock_data(self, symbol: str) -> Optional[AssetData]:
        """
        Get current stock data using Yahoo Finance API
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            AssetData object with stock information or None if not found
        """
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # If no info is returned, the ticker may not exist
            if not info or "regularMarketPrice" not in info:
                logger.warning(f"No data found for stock symbol: {symbol}")
                return None
            
            # Extract relevant information
            current_price = info.get("regularMarketPrice", 0.0)
            previous_close = info.get("previousClose", current_price)
            
            # Create asset data object
            asset = AssetData(
                symbol=symbol,
                name=info.get("shortName", symbol),
                asset_type=AssetType.STOCK,
                current_price=current_price,
                previous_close=previous_close,
                volume=info.get("volume", None),
                market_cap=info.get("marketCap", None),
                pe_ratio=info.get("trailingPE", None),
                dividend_yield=info.get("dividendYield", None),
                beta=info.get("beta", None),
                high_52_week=info.get("fiftyTwoWeekHigh", None),
                low_52_week=info.get("fiftyTwoWeekLow", None),
                moving_averages={
                    "50day": info.get("fiftyDayAverage", None),
                    "200day": info.get("twoHundredDayAverage", None)
                },
                last_updated=datetime.now()
            )
            
            # Calculate changes
            asset.calculate_changes()
            
            return asset
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def get_etf_data(self, symbol: str) -> Optional[AssetData]:
        """
        Get current ETF data using Yahoo Finance API
        
        Args:
            symbol: ETF ticker symbol
            
        Returns:
            AssetData object with ETF information or None if not found
        """
        try:
            # ETF data can be fetched the same way as stocks
            asset = self.get_stock_data(symbol)
            
            if asset:
                asset.asset_type = AssetType.ETF
            
            return asset
            
        except Exception as e:
            logger.error(f"Error fetching ETF data for {symbol}: {e}")
            return None
    
    def get_crypto_data(self, symbol: str) -> Optional[AssetData]:
        """
        Get current cryptocurrency data using CoinGecko API
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'btc' for Bitcoin)
            
        Returns:
            AssetData object with cryptocurrency information or None if not found
        """
        try:
            # Convert symbol to lowercase and handle common mappings
            symbol = symbol.lower()
            
            # Map common ticker symbols to CoinGecko IDs
            crypto_map = {
                "btc": "bitcoin",
                "eth": "ethereum",
                "sol": "solana",
                "ada": "cardano",
                "xrp": "ripple",
                "dot": "polkadot",
                "doge": "dogecoin",
                "uni": "uniswap",
                "link": "chainlink",
                "ltc": "litecoin"
            }
            
            coin_id = crypto_map.get(symbol, symbol)
            
            # Try to get coin data by ID
            coin_data = self.coingecko_client.get_coin_by_id(
                id=coin_id,
                localization=False,
                tickers=False,
                market_data=True,
                community_data=False,
                developer_data=False
            )
            
            if not coin_data or "market_data" not in coin_data:
                logger.warning(f"No data found for cryptocurrency: {symbol}")
                return None
            
            market_data = coin_data["market_data"]
            
            # Create asset data object
            asset = AssetData(
                symbol=symbol.upper(),
                name=coin_data.get("name", symbol.upper()),
                asset_type=AssetType.CRYPTO,
                current_price=market_data["current_price"].get("usd", 0.0),
                previous_close=market_data["current_price"].get("usd", 0.0) / (1 + market_data["price_change_percentage_24h"] / 100) if "price_change_percentage_24h" in market_data else 0.0,
                volume=market_data.get("total_volume", {}).get("usd"),
                market_cap=market_data.get("market_cap", {}).get("usd"),
                high_52_week=market_data.get("ath", {}).get("usd"),
                additional_metrics={
                    "market_cap_rank": coin_data.get("market_cap_rank"),
                    "price_change_percentage_24h": market_data.get("price_change_percentage_24h"),
                    "price_change_percentage_7d": market_data.get("price_change_percentage_7d"),
                    "price_change_percentage_30d": market_data.get("price_change_percentage_30d")
                },
                last_updated=datetime.now()
            )
            
            # Calculate changes
            asset.calculate_changes()
            
            return asset
            
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return None
    
    def get_asset_data(self, symbol: str, asset_type: AssetType = None) -> Optional[AssetData]:
        """
        Get asset data based on symbol and type
        
        Args:
            symbol: Asset symbol
            asset_type: Type of asset (stock, etf, crypto)
            
        Returns:
            AssetData object with asset information or None if not found
        """
        # If asset type is specified, use the appropriate method
        if asset_type:
            if asset_type == AssetType.STOCK:
                return self.get_stock_data(symbol)
            elif asset_type == AssetType.ETF:
                return self.get_etf_data(symbol)
            elif asset_type == AssetType.CRYPTO:
                return self.get_crypto_data(symbol)
        
        # If asset type is not specified, try to guess based on symbol format
        if symbol.lower() in ["btc", "eth", "sol", "ada", "xrp", "dot", "doge", "uni", "link", "ltc"]:
            return self.get_crypto_data(symbol)
        
        # Try stock/ETF data first, then fallback to crypto if not found
        asset = self.get_stock_data(symbol)
        if not asset:
            asset = self.get_crypto_data(symbol)
        
        return asset
    
    def get_market_indices(self) -> Dict[str, float]:
        """
        Get current market indices data
        
        Returns:
            Dictionary with market indices values
        """
        try:
            # Define indices symbols
            indices = {
                "^GSPC": "sp500",
                "^DJI": "dow_jones",
                "^IXIC": "nasdaq",
                "^RUT": "russell_2000",
                "^VIX": "vix"
            }
            
            # Get data for all indices
            data = {}
            for symbol, name in indices.items():
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info and "regularMarketPrice" in info:
                    data[name] = info["regularMarketPrice"]
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market indices: {e}")
            return {}
    
    def get_financial_news(self, query: str = None, tickers: List[str] = None, category: str = "business", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get financial news articles
        
        Args:
            query: Search query for news
            tickers: List of ticker symbols to get news for
            category: News category
            limit: Maximum number of articles to return
            
        Returns:
            List of news article dictionaries
        """
        try:
            if not self.news_api_client:
                logger.warning("News API client not initialized. Set NEWS_API_KEY in .env file.")
                return []
            
            # Prepare query parameters
            params = {
                "language": "en",
                "category": category,
                "sortBy": "publishedAt"
            }
            
            # If query is provided, use it
            if query:
                params["q"] = query
            
            # If tickers are provided, add them to the query
            if tickers:
                ticker_query = " OR ".join([f"{ticker}" for ticker in tickers])
                if "q" in params:
                    params["q"] = f"{params['q']} AND ({ticker_query})"
                else:
                    params["q"] = ticker_query
            
            # Get news articles
            response = self.news_api_client.get_top_headlines(**params) if not query else self.news_api_client.get_everything(**params)
            
            # Process and limit articles
            articles = response.get("articles", [])[:limit]
            
            # Format articles
            formatted_articles = []
            for article in articles:
                formatted = {
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "author": article.get("author", None),
                    "published_at": article.get("publishedAt", datetime.now().isoformat()),
                    "content": article.get("content", None),
                    "description": article.get("description", None),
                    "image_url": article.get("urlToImage", None)
                }
                formatted_articles.append(formatted)
            
            return formatted_articles
            
        except Exception as e:
            logger.error(f"Error fetching financial news: {e}")
            return []
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information using Finnhub API
        
        Args:
            symbol: Company ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            if not self.finnhub_client:
                logger.warning("Finnhub client not initialized. Set FINNHUB_API_KEY in .env file.")
                return {}
            
            # Get company profile
            profile = self.finnhub_client.company_profile2(symbol=symbol)
            
            # Get basic financials
            financials = self.finnhub_client.company_basic_financials(symbol, 'all')
            
            # Combine data
            company_info = {
                "profile": profile,
                "financials": financials
            }
            
            return company_info
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return {}

class NewsAPIClient:
    """Class for fetching news from NewsAPI"""
    
    def __init__(self):
        self.client = FinancialDataClient()
    
    def get_news(self, query: str = "", category: str = "business", limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch news articles"""
        return self.client.get_financial_news(query=query, category=category, limit=limit)
    
    def get_news_for_query(self, query: str = "", category: str = "business", max_results: int = 10) -> List[Dict[str, Any]]:
        """Fetch news articles (alias for get_news as used in some tests)"""
        return self.get_news(query=query, category=category, limit=max_results)

class YFinanceClient:
    """Class for fetching data from Yahoo Finance"""
    
    def __init__(self):
        self.client = FinancialDataClient()
    
    def get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker information"""
        ticker = yf.Ticker(symbol)
        return ticker.info
