"""
Tests for the MarketDataAgent class.
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the path so we can import the agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.market_data_agent import MarketDataAgent


@pytest.fixture
def sample_market_data():
    """Fixture to provide sample market data for testing."""
    return {
        "timestamp": "2025-06-01T16:00:00.000Z",
        "indices": [
            {
                "name": "S&P 500",
                "symbol": "^GSPC",
                "price": 5420.25,
                "change": 32.45,
                "change_percent": 0.60,
                "prev_close": 5387.80
            },
            {
                "name": "Dow Jones",
                "symbol": "^DJI",
                "price": 41250.75,
                "change": 150.35,
                "change_percent": 0.37,
                "prev_close": 41100.40
            },
            {
                "name": "NASDAQ",
                "symbol": "^IXIC",
                "price": 17345.65,
                "change": 95.25,
                "change_percent": 0.55,
                "prev_close": 17250.40
            }
        ],
        "sectors": [
            {
                "name": "Technology",
                "change_percent": 0.85,
                "ytd_change_percent": 15.32
            },
            {
                "name": "Healthcare",
                "change_percent": 0.42,
                "ytd_change_percent": 8.76
            },
            {
                "name": "Financial Services",
                "change_percent": 0.65,
                "ytd_change_percent": 12.45
            }
        ],
        "commodities": [
            {
                "name": "Crude Oil WTI",
                "symbol": "CL=F",
                "price": 72.45,
                "change": -0.85,
                "change_percent": -1.16,
                "unit": "USD/barrel"
            },
            {
                "name": "Gold",
                "symbol": "GC=F",
                "price": 2350.25,
                "change": 15.75,
                "change_percent": 0.67,
                "unit": "USD/oz"
            }
        ],
        "cryptocurrencies": [
            {
                "name": "Bitcoin",
                "symbol": "BTC-USD",
                "price": 72450.25,
                "change": 1245.85,
                "change_percent": 1.75,
                "market_cap": 1425000000000
            },
            {
                "name": "Ethereum",
                "symbol": "ETH-USD",
                "price": 3850.75,
                "change": 65.35,
                "change_percent": 1.72,
                "market_cap": 465000000000
            }
        ],
        "forex": [
            {
                "pair": "EUR/USD",
                "rate": 1.0925,
                "change": 0.0025,
                "change_percent": 0.23
            },
            {
                "pair": "USD/JPY",
                "rate": 154.85,
                "change": -0.35,
                "change_percent": -0.23
            }
        ],
        "economic_indicators": [
            {
                "name": "US 10-Year Treasury",
                "value": 4.32,
                "previous": 4.29,
                "change": 0.03
            },
            {
                "name": "US Inflation Rate",
                "value": 3.2,
                "previous": 3.4,
                "change": -0.2
            }
        ]
    }


@pytest.fixture
def mock_agent(sample_market_data):
    """Fixture to create a mock market data agent with mocked dependencies."""
    with patch("agents.market_data_agent.get_registry", autospec=True) as mock_get_registry, \
         patch("agents.market_data_agent.os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=json.dumps(sample_market_data))), \
         patch("json.load", return_value=sample_market_data), \
         patch("agents.market_data_agent.ChatOllama") as mock_chat, \
         patch("agents.market_data_agent.YFinanceClient") as mock_yfin:
        
        # Mock registry
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        # Set up mock for YFinance client
        mock_yfin_instance = MagicMock()
        mock_yfin.return_value = mock_yfin_instance
        mock_yfin_instance.get_ticker_data.return_value = {
            "AAPL": {
                "name": "Apple Inc.",
                "price": 171.50,
                "change": 2.35,
                "change_percent": 1.39,
                "volume": 58425632,
                "market_cap": 2650000000000,
                "pe_ratio": 28.5,
                "dividend_yield": 0.52
            }
        }
        
        # Set up the agent context
        mock_context = MagicMock()
        mock_registry.get_latest_context.return_value = mock_context
        mock_context.content = sample_market_data
        
        # Create the agent with mocked dependencies
        agent = MarketDataAgent()
        
        yield agent


class TestMarketDataAgent:
    """Test cases for the MarketDataAgent class."""
    
    def test_initialization(self, mock_agent):
        """Test that the agent initializes correctly."""
        assert mock_agent is not None
        assert mock_agent.market_data_file_path is not None
        assert mock_agent.tools is not None
        assert mock_agent.agent_executor is not None
    
    def test_get_market_indices(self, mock_agent, sample_market_data):
        """Test the get_market_indices tool."""
        result = mock_agent._create_tools()[0]()
        
        # Check that the result is a string containing index information
        assert isinstance(result, str)
        assert "S&P 500" in result
        assert "Dow Jones" in result
        assert "NASDAQ" in result
        
        # Check for values
        assert "5420.25" in result
        assert "41250.75" in result
        assert "17345.65" in result
    
    def test_get_sector_performance(self, mock_agent, sample_market_data):
        """Test the get_sector_performance tool."""
        result = mock_agent._create_tools()[1]()
        
        # Check that the result contains sector performance information
        assert isinstance(result, str)
        assert "Technology" in result
        assert "Healthcare" in result
        assert "Financial Services" in result
        
        # Check for percentages
        assert "0.85" in result or "0.85%" in result
        assert "15.32" in result or "15.32%" in result
    
    def test_get_commodity_prices(self, mock_agent, sample_market_data):
        """Test the get_commodity_prices tool."""
        result = mock_agent._create_tools()[2]()
        
        # Check that the result contains commodity information
        assert isinstance(result, str)
        assert "Crude Oil" in result
        assert "Gold" in result
        
        # Check for prices
        assert "72.45" in result
        assert "2350.25" in result
    
    def test_get_cryptocurrency_prices(self, mock_agent, sample_market_data):
        """Test the get_cryptocurrency_prices tool."""
        result = mock_agent._create_tools()[3]()
        
        # Check that the result contains cryptocurrency information
        assert isinstance(result, str)
        assert "Bitcoin" in result
        assert "Ethereum" in result
        
        # Check for prices
        assert "72450.25" in result
        assert "3850.75" in result
    
    def test_get_forex_rates(self, mock_agent, sample_market_data):
        """Test the get_forex_rates tool."""
        result = mock_agent._create_tools()[4]()
        
        # Check that the result contains forex information
        assert isinstance(result, str)
        assert "EUR/USD" in result
        assert "USD/JPY" in result
        
        # Check for rates
        assert "1.0925" in result
        assert "154.85" in result
    
    def test_get_economic_indicators(self, mock_agent, sample_market_data):
        """Test the get_economic_indicators tool."""
        result = mock_agent._create_tools()[5]()
        
        # Check that the result contains economic indicators
        assert isinstance(result, str)
        assert "US 10-Year Treasury" in result
        assert "US Inflation Rate" in result
        
        # Check for values
        assert "4.32" in result
        assert "3.2" in result
    
    def test_get_stock_quote(self, mock_agent):
        """Test the get_stock_quote tool."""
        result = mock_agent._create_tools()[6]("AAPL")
        
        # Check that the result contains stock information
        assert isinstance(result, str)
        assert "Apple" in result
        assert "171.50" in result
        assert "1.39" in result or "1.39%" in result
    
    def test_get_market_summary(self, mock_agent):
        """Test the get_market_summary tool."""
        result = mock_agent._create_tools()[7]()
        
        # Check that the result is a comprehensive summary
        assert isinstance(result, str)
        assert "S&P 500" in result
        assert "sectors" in result.lower()
        assert "commodities" in result.lower()
        assert "treasury" in result.lower() or "economic" in result.lower()
    
    def test_update_market_data(self, mock_agent):
        """Test the update_market_data tool."""
        with patch("json.dump") as mock_json_dump, \
             patch("builtins.open", mock_open()) as mock_file:
            
            result = mock_agent._create_tools()[8]()
            
            # Check that json.dump was called (saving updated data)
            mock_json_dump.assert_called_once()
            
            # Check that the result indicates success
            assert isinstance(result, str)
            assert "updated" in result.lower()
            assert "market data" in result.lower()
    
    @patch("agents.market_data_agent.ChatOllama")
    def test_run_method(self, mock_chat_ollama, mock_agent):
        """Test the run method processes queries correctly."""
        # Setup the mock executor to return a response
        mock_agent.agent_executor.invoke.return_value = {"output": "Market data summary generated"}
        
        # Call the run method
        result = mock_agent.run("Give me a summary of today's market")
        
        # Verify the executor was called with the query
        mock_agent.agent_executor.invoke.assert_called_once_with(
            {"input": "Give me a summary of today's market"}
        )
        
        # Verify the result is what we expect
        assert result == "Market data summary generated"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
