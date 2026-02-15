"""
Tests for the PortfolioAnalyzerAgent class.
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the path so we can import the agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.portfolio_analyzer_agent import PortfolioAnalyzerAgent


@pytest.fixture
def sample_portfolio_data():
    """Fixture to provide sample portfolio data for testing."""
    return {
        "portfolio_summary": {
            "total_value": 354750.25,
            "cash_balance": 25000.00,
            "total_return": 35475.25,
            "total_return_percent": 11.12
        },
        "holdings": [
            {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "asset_class": "Equity",
                "sector": "Technology",
                "quantity": 85,
                "purchase_price": 156.35,
                "current_price": 171.50,
                "market_value": 14577.50,
                "weight": 4.11,
                "gain_loss": 1287.75,
                "gain_loss_percent": 9.69
            },
            {
                "ticker": "VOO",
                "name": "Vanguard S&P 500 ETF",
                "asset_class": "ETF",
                "sector": "Blend",
                "quantity": 150,
                "purchase_price": 385.25,
                "current_price": 415.75,
                "market_value": 62362.50,
                "weight": 17.58,
                "gain_loss": 4575.00,
                "gain_loss_percent": 7.91
            }
        ],
        "asset_allocation": [
            {"asset_class": "Equity", "value": 142421.50, "weight": 40.15},
            {"asset_class": "ETF", "value": 150376.25, "weight": 42.39},
            {"asset_class": "Bond", "value": 38839.00, "weight": 10.95},
            {"asset_class": "Cryptocurrency", "value": 66584.31, "weight": 18.77},
            {"asset_class": "Commodity", "value": 13372.19, "weight": 3.77},
            {"asset_class": "Cash", "value": 25000.00, "weight": 7.05}
        ],
        "sector_allocation": [
            {"sector": "Technology", "value": 69774.25, "weight": 19.67},
            {"sector": "Financial Services", "value": 14630.00, "weight": 4.12},
            {"sector": "Healthcare", "value": 21555.00, "weight": 6.08},
            {"sector": "Consumer Cyclical", "value": 11882.25, "weight": 3.35},
            {"sector": "Blend", "value": 150376.25, "weight": 42.39},
            {"sector": "Energy", "value": 10157.25, "weight": 2.86},
            {"sector": "Communication Services", "value": 14423.00, "weight": 4.07},
            {"sector": "Cryptocurrency", "value": 66584.31, "weight": 18.77},
            {"sector": "Commodity", "value": 13372.19, "weight": 3.77},
            {"sector": "Cash", "value": 25000.00, "weight": 7.05}
        ],
        "performance": {
            "1d": 0.55,
            "1w": 1.20,
            "1m": 2.35,
            "3m": 4.58,
            "6m": 6.92,
            "ytd": 8.75,
            "1y": 11.12,
            "3y": 36.45,
            "5y": 62.18
        },
        "risk_metrics": {
            "volatility": 12.85,
            "sharpe_ratio": 1.25,
            "sortino_ratio": 1.45,
            "max_drawdown": 18.25,
            "beta": 0.95,
            "alpha": 2.15,
            "r_squared": 0.89
        }
    }


@pytest.fixture
def mock_agent(sample_portfolio_data):
    """Fixture to create a mock portfolio analyzer agent with mocked dependencies."""
    with patch("agents.portfolio_analyzer_agent.get_registry", autospec=True) as mock_get_registry, \
         patch("agents.portfolio_analyzer_agent.os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=json.dumps(sample_portfolio_data))), \
         patch("json.load", return_value=sample_portfolio_data), \
         patch("agents.portfolio_analyzer_agent.ChatOllama") as mock_chat:
        
        # Mock registry
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        # Create a mock for the visualization utility
        mock_visualizer = MagicMock()
        
        # Create the agent with mocked dependencies
        agent = PortfolioAnalyzerAgent(visualizer=mock_visualizer)
        
        # Set up the agent context
        mock_context = MagicMock()
        mock_registry.get_latest_context.return_value = mock_context
        mock_context.content = sample_portfolio_data
        
        yield agent


class TestPortfolioAnalyzerAgent:
    """Test cases for the PortfolioAnalyzerAgent class."""
    
    def test_initialization(self, mock_agent):
        """Test that the agent initializes correctly."""
        assert mock_agent is not None
        assert mock_agent.portfolio_file_path is not None
        assert mock_agent.tools is not None
        assert mock_agent.agent_executor is not None
    
    def test_get_portfolio_data(self, mock_agent, sample_portfolio_data):
        """Test the get_portfolio_data tool."""
        result = mock_agent._create_tools()[0]()
        
        # Check that the result is a string containing portfolio information
        assert isinstance(result, str)
        assert "total_value" in result
        assert str(sample_portfolio_data["portfolio_summary"]["total_value"]) in result
    
    def test_get_holdings(self, mock_agent, sample_portfolio_data):
        """Test the get_holdings tool."""
        result = mock_agent._create_tools()[1]()
        
        # Check that the result contains the expected holdings
        assert isinstance(result, str)
        assert "AAPL" in result
        assert "VOO" in result
        
        # Check that some key details are included
        assert "Apple Inc." in result
        assert "Vanguard S&P 500 ETF" in result
    
    def test_get_asset_allocation(self, mock_agent, sample_portfolio_data):
        """Test the get_asset_allocation tool."""
        result = mock_agent._create_tools()[2]()
        
        # Check that the result contains asset allocation information
        assert isinstance(result, str)
        assert "Equity" in result
        assert "ETF" in result
        assert "Bond" in result
        assert "Cryptocurrency" in result
        
        # Check for percentages
        assert "40.15" in result or "40.15%" in result
        assert "42.39" in result or "42.39%" in result
    
    def test_get_sector_allocation(self, mock_agent, sample_portfolio_data):
        """Test the get_sector_allocation tool."""
        result = mock_agent._create_tools()[3]()
        
        # Check that the result contains sector allocation information
        assert isinstance(result, str)
        assert "Technology" in result
        assert "Financial Services" in result
        assert "Healthcare" in result
        
        # Check for percentages
        assert "19.67" in result or "19.67%" in result
        assert "4.12" in result or "4.12%" in result
    
    def test_get_performance(self, mock_agent, sample_portfolio_data):
        """Test the get_performance tool."""
        result = mock_agent._create_tools()[4]()
        
        # Check that the result contains performance metrics
        assert isinstance(result, str)
        assert "1d" in result
        assert "1y" in result
        assert "5y" in result
        
        # Check for return percentages
        assert "11.12" in result or "11.12%" in result
        assert "62.18" in result or "62.18%" in result
    
    def test_get_risk_metrics(self, mock_agent, sample_portfolio_data):
        """Test the get_risk_metrics tool."""
        result = mock_agent._create_tools()[5]()
        
        # Check that the result contains risk metrics
        assert isinstance(result, str)
        assert "volatility" in result.lower()
        assert "sharpe ratio" in result.lower()
        assert "max drawdown" in result.lower()
        
        # Check for actual values
        assert "12.85" in result
        assert "1.25" in result
        assert "18.25" in result
    
    def test_analyze_portfolio(self, mock_agent):
        """Test the analyze_portfolio tool."""
        result = mock_agent._create_tools()[6]()
        
        # Check that the result is a comprehensive analysis
        assert isinstance(result, str)
        assert "analysis" in result.lower()
        assert "diversification" in result.lower() or "allocation" in result.lower()
        assert "performance" in result.lower()
        assert "risk" in result.lower()
    
    def test_generate_visualizations(self, mock_agent):
        """Test the generate_visualizations tool."""
        # Call the tool
        result = mock_agent._create_tools()[7]()
        
        # Check that the visualizer was called
        mock_agent.visualizer.generate_asset_allocation_chart.assert_called_once()
        mock_agent.visualizer.generate_sector_allocation_chart.assert_called_once()
        mock_agent.visualizer.generate_performance_chart.assert_called_once()
        mock_agent.visualizer.generate_top_holdings_chart.assert_called_once()
        
        # Check that the result indicates successful visualization
        assert isinstance(result, str)
        assert "generated" in result.lower()
        assert "visualization" in result.lower()
    
    @patch("agents.portfolio_analyzer_agent.ChatOllama")
    def test_run_method(self, mock_chat_ollama, mock_agent):
        """Test the run method processes queries correctly."""
        # Setup the mock executor to return a response
        mock_agent.agent_executor.invoke.return_value = {"output": "Portfolio analysis complete"}
        
        # Call the run method
        result = mock_agent.run("Analyze my portfolio diversification")
        
        # Verify the executor was called with the query
        mock_agent.agent_executor.invoke.assert_called_once_with(
            {"input": "Analyze my portfolio diversification"}
        )
        
        # Verify the result is what we expect
        assert result == "Portfolio analysis complete"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
