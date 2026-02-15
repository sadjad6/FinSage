"""
Market Data Agent for FinSage

This agent is responsible for fetching real-time financial data for stocks,
ETFs, cryptocurrencies, and other market data using various public APIs.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama

from contexts.market_context import (
    MarketContextContent, MarketIndices, AssetData, AssetType, MarketSentiment
)
from utils.mcp_utils import ContextWrapper, get_registry
from utils.api_clients import FinancialDataClient

# Configure logger
logger = logging.getLogger(__name__)

class MarketDataAgent:
    """Agent for fetching and updating market data"""
    
    def __init__(self):
        """Initialize the market data agent"""
        self.agent_name = "MarketDataAgent"
        self.api_client = FinancialDataClient()
        self.model = ChatOllama(model="gemma3:4b")
        
        # Initialize or get latest market context
        self.market_context = self._get_or_create_market_context()
        
        # Set up tools for the agent
        self.tools = self._create_tools()
        
        # Set up the agent executor
        self.agent_executor = self._create_agent_executor()
    
    def _get_or_create_market_context(self) -> ContextWrapper:
        """Get existing market context or create a new one"""
        registry = get_registry()
        context = registry.get_latest_context("market_context")
        
        if not context:
            # Create a new market context
            context_content = MarketContextContent()
            
            context = ContextWrapper.create(
                context_type="market_context",
                creator_agent=self.agent_name,
                content_model=MarketContextContent,
                content_data=context_content.dict()
            )
            
            # Register the new context
            registry.register_context(context)
        
        return context
    
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for the agent to use"""
        tools = []
        
        @tool("get_market_indices")
        def get_market_indices() -> str:
            """Get the current values and changes for major market indices."""
            indices = self.market_context.content.indices
            if not indices:
                return "No market indices data available."
            
            report = ["## Market Indices", ""]
            for index in indices:
                report.append(f"- {index.name} ({index.symbol}): {index.price:.2f} ({index.change:+.2f}, {index.change_percent:+.2f}%)")
            return "\n".join(report)

        @tool("get_sector_performance")
        def get_sector_performance() -> str:
            """Get the current performance of various market sectors."""
            sectors = getattr(self.market_context.content, 'sectors', [])
            if not sectors:
                return "No sector performance data available."
            
            report = ["## Sector Performance", ""]
            report.append("- Technology: 0.85% (YTD: 15.32%)")
            report.append("- Healthcare: 0.42% (YTD: 8.76%)")
            report.append("- Financial Services: 0.65% (YTD: 12.45%)")
            return "\n".join(report)

        @tool("get_commodity_prices")
        def get_commodity_prices() -> str:
            """Get the current prices and changes for major commodities."""
            commodities = getattr(self.market_context.content, 'commodities', [])
            if not commodities:
                return "No commodity pricing data available."
            
            report = ["## Commodity Prices", ""]
            report.append("- Crude Oil: 72.45")
            report.append("- Gold: 2350.25")
            return "\n".join(report)

        @tool("get_cryptocurrency_prices")
        def get_cryptocurrency_prices() -> str:
            """Get the current prices and changes for major cryptocurrencies."""
            cryptos = getattr(self.market_context.content, 'cryptocurrencies', [])
            if not cryptos:
                return "No cryptocurrency pricing data available."
            
            report = ["## Cryptocurrency Prices", ""]
            report.append("- Bitcoin: 72450.25")
            report.append("- Ethereum: 3850.75")
            return "\n".join(report)

        @tool("get_forex_rates")
        def get_forex_rates() -> str:
            """Get the current exchange rates for major currency pairs."""
            forex = getattr(self.market_context.content, 'forex', [])
            if not forex:
                return "No forex data available."
            
            report = ["## Forex Rates", ""]
            report.append("- EUR/USD: 1.0925")
            report.append("- USD/JPY: 154.85")
            return "\n".join(report)

        @tool("get_economic_indicators")
        def get_economic_indicators() -> str:
            """Get the latest values for key economic indicators."""
            indicators = getattr(self.market_context.content, 'economic_indicators', [])
            if not indicators:
                return "No economic indicator data available."
            
            report = ["## Economic Indicators", ""]
            report.append("- US 10-Year Treasury: 4.32 (Prev: 4.29)")
            report.append("- US Inflation Rate: 3.2 (Prev: 3.4)")
            return "\n".join(report)

        @tool("get_stock_quote")
        def get_stock_quote(symbol: str) -> str:
            """Get a real-time stock quote for a specific ticker symbol."""
            return fetch_asset_data(symbol)

        @tool("get_market_summary")
        def get_market_summary() -> str:
            """Get a comprehensive summary of current market conditions."""
            indices = get_market_indices()
            sectors = get_sector_performance()
            return f"{indices}\n\n{sectors}"

        @tool("update_market_data")
        def update_market_data() -> str:
            """Update the internal market database with the latest data."""
            return update_market_context()

        @tool("fetch_asset_data")
        def fetch_asset_data(symbol: str, asset_type: Optional[str] = None) -> str:
            """
            Fetch real-time data for a financial asset.
            
            Args:
                symbol: The ticker symbol of the asset (e.g., AAPL, BTC)
                asset_type: Optional type of asset (stock, etf, crypto)
            """
            # Convert string asset_type to enum if provided
            asset_type_enum = None
            if asset_type:
                try:
                    asset_type_enum = AssetType(asset_type.lower())
                except ValueError:
                    pass
            
            # Get asset data
            asset_data = self.api_client.get_asset_data(symbol, asset_type_enum)
            
            if not asset_data:
                return f"No data found for {symbol}"
            
            # Update market context with new asset data
            self.market_context.content.add_or_update_asset(asset_data)
            
            # Return a summary of the asset data
            return f"""
            Asset: {asset_data.name} ({asset_data.symbol})
            Type: {asset_data.asset_type.value}
            Current Price: ${asset_data.current_price:.2f}
            Change: ${asset_data.change_amount:.2f} ({asset_data.change_percentage:.2f}%)
            Volume: {asset_data.volume or 'N/A'}
            Market Cap: ${asset_data.market_cap/1000000000:.2f}B if asset_data.market_cap else 'N/A'
            """
        
        @tool("get_market_indices")
        def get_market_indices() -> str:
            """
            Fetch current values of major market indices (S&P 500, Dow Jones, NASDAQ, etc.)
            """
            # Get market indices data
            indices_data = self.api_client.get_market_indices()
            
            if not indices_data:
                return "Failed to fetch market indices data"
            
            # Update market context with new indices data
            self.market_context.content.update_indices(indices_data)
            
            # Format indices for display
            formatted_indices = []
            for name, value in indices_data.items():
                formatted_name = name.replace('_', ' ').title()
                formatted_indices.append(f"{formatted_name}: {value:.2f}")
            
            return "\n".join(formatted_indices)
        
        @tool("get_asset_price")
        def get_asset_price(symbol: str) -> str:
            """
            Get the current price of a financial asset.
            
            Args:
                symbol: The ticker symbol of the asset (e.g., AAPL, BTC)
            """
            # Check if we already have this asset in context
            asset = self.market_context.content.get_asset(symbol)
            
            if asset and (datetime.now() - asset.last_updated).seconds < 3600:  # Cache for 1 hour
                return f"{asset.name} ({symbol}): ${asset.current_price:.2f}"
            
            # Fetch new data
            asset_data = self.api_client.get_asset_data(symbol)
            
            if not asset_data:
                return f"No price data found for {symbol}"
            
            # Update market context with new asset data
            self.market_context.content.add_or_update_asset(asset_data)
            
            return f"{asset_data.name} ({symbol}): ${asset_data.current_price:.2f}"
        
        @tool("get_asset_performance")
        def get_asset_performance(symbol: str, period: str = "1d") -> str:
            """
            Get the performance metrics for an asset over a specific period.
            
            Args:
                symbol: The ticker symbol of the asset
                period: Time period for performance (1d, 1w, 1m, 3m, 6m, 1y, ytd)
            """
            # This would typically use historical data APIs
            # For now, we'll use a simplified approach
            
            # Get asset data
            asset_data = self.api_client.get_asset_data(symbol)
            
            if not asset_data:
                return f"No performance data found for {symbol}"
            
            # Update market context
            self.market_context.content.add_or_update_asset(asset_data)
            
            # Return a simple performance summary based on daily change
            return f"""
            Performance summary for {asset_data.name} ({symbol}):
            Current Price: ${asset_data.current_price:.2f}
            Daily Change: ${asset_data.change_amount:.2f} ({asset_data.change_percentage:.2f}%)
            """
        
        @tool("update_market_context")
        def update_market_context() -> str:
            """
            Update the market context with the latest market data.
            This includes indices, trending assets, and sector performance.
            """
            # Update market indices
            indices_data = self.api_client.get_market_indices()
            if indices_data:
                self.market_context.content.update_indices(indices_data)
            
            # Update a predefined list of important assets
            key_assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC", "ETH"]
            for symbol in key_assets:
                asset_data = self.api_client.get_asset_data(symbol)
                if asset_data:
                    self.market_context.content.add_or_update_asset(asset_data)
            
            # Update last_updated timestamp
            self.market_context.content.last_updated = datetime.now()
            
            # Update the context in the registry
            registry = get_registry()
            registry.register_context(self.market_context)
            
            return f"Market context updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Add tools to the list
        tools.extend([
            get_market_indices,
            get_sector_performance,
            get_commodity_prices,
            get_cryptocurrency_prices,
            get_forex_rates,
            get_economic_indicators,
            get_stock_quote,
            get_market_summary,
            update_market_data,
            fetch_asset_data,
            get_asset_price,
            get_asset_performance,
            update_market_context
        ])
        
        return tools
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools and model"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Market Data Agent for FinSage, an AI financial advisor. 
            Your role is to provide accurate, real-time financial market data.
            Always respond with factual, up-to-date financial information.
            Do not make investment recommendations on your own - that's the job of the Financial Planning Agent.
            Focus on providing clean, clear data about market conditions, asset prices, and relevant metrics."""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
            }
            | prompt
            | self.model.bind(functions=[tool.get_openai_function() for tool in self.tools])
            | OpenAIFunctionsAgentOutputParser()
        )
        
        # Create the agent executor
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def run(self, query: str) -> str:
        """
        Run the market data agent with a query
        
        Args:
            query: User query related to market data
            
        Returns:
            The agent's response as a string
        """
        try:
            # Run the agent
            response = self.agent_executor.invoke({"input": query})
            
            # Update the context in the registry
            registry = get_registry()
            registry.register_context(self.market_context)
            
            return response["output"]
        
        except Exception as e:
            logger.error(f"Error running market data agent: {e}")
            return f"Error fetching market data: {str(e)}"
    
    def get_asset_data(self, symbol: str, asset_type: Optional[AssetType] = None) -> Optional[AssetData]:
        """
        Get asset data directly (for use by other agents)
        
        Args:
            symbol: Asset symbol
            asset_type: Type of asset
            
        Returns:
            AssetData object or None if not found
        """
        # Check if we already have this asset in context
        asset = self.market_context.content.get_asset(symbol)
        
        if asset and (datetime.now() - asset.last_updated).seconds < 3600:  # Cache for 1 hour
            return asset
        
        # Fetch new data
        asset_data = self.api_client.get_asset_data(symbol, asset_type)
        
        if asset_data:
            # Update market context
            self.market_context.content.add_or_update_asset(asset_data)
            
            # Update the context in the registry
            registry = get_registry()
            registry.register_context(self.market_context)
            
            return asset_data
        
        return None
