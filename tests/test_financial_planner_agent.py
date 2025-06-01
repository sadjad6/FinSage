"""
Tests for the FinancialPlannerAgent class.
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.financial_planner_agent import FinancialPlannerAgent


@pytest.fixture
def sample_user_profile():
    """Fixture to load a sample user profile for testing."""
    with open("data/sample_user_profile.json", "r") as f:
        return json.load(f)


@pytest.fixture
def mock_agent():
    """Fixture to create a mock financial planner agent with mocked dependencies."""
    with patch("agents.financial_planner_agent.MCP", autospec=True) as mock_mcp:
        # Mock context to return sample data
        mock_context = MagicMock()
        mock_mcp.get_or_create_context.return_value = mock_context
        
        # Create mock agents
        mock_market_data_agent = MagicMock()
        mock_portfolio_analyzer_agent = MagicMock()
        
        # Create the agent
        agent = FinancialPlannerAgent(
            market_data_agent=mock_market_data_agent,
            portfolio_analyzer_agent=mock_portfolio_analyzer_agent
        )
        
        yield agent


class TestFinancialPlannerAgent:
    """Test cases for the FinancialPlannerAgent class."""
    
    def test_initialization(self, mock_agent):
        """Test that the agent initializes correctly."""
        assert mock_agent is not None
        assert mock_agent.tools is not None
        assert mock_agent.agent_executor is not None
    
    def test_get_user_profile(self, mock_agent, sample_user_profile):
        """Test the get_user_profile tool."""
        # Setup the mock context to return the sample profile
        mock_agent.user_profile_context.get.return_value = sample_user_profile
        
        # Call the tool
        result = mock_agent._create_tools()[0]()
        
        # Verify the context was accessed
        mock_agent.user_profile_context.get.assert_called_once()
        
        # Verify the result contains expected user info
        assert "Alex Johnson" in result
    
    def test_recommend_asset_allocation(self, mock_agent, sample_user_profile):
        """Test the recommend_asset_allocation tool."""
        # Setup the mock context
        mock_agent.user_profile_context.get.return_value = sample_user_profile
        
        # Call the tool
        result = mock_agent._create_tools()[2]()
        
        # Verify we got a recommendation string back
        assert isinstance(result, str)
        assert "allocation" in result.lower()
    
    @patch("agents.financial_planner_agent.ChatOllama")
    def test_run_method(self, mock_chat_ollama, mock_agent):
        """Test the run method processes queries correctly."""
        # Setup the mock executor to return a response
        mock_agent.agent_executor.run.return_value = "Test financial advice"
        
        # Call the run method
        result = mock_agent.run("What should I invest in?")
        
        # Verify the executor was called with the query
        mock_agent.agent_executor.run.assert_called_once_with(
            input="What should I invest in?"
        )
        
        # Verify the result is what we expect
        assert result == "Test financial advice"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
