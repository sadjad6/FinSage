"""
Tests for the ComplianceAgent class.
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.compliance_agent import ComplianceAgent


@pytest.fixture
def sample_user_profile():
    """Fixture to load a sample user profile for testing."""
    with open("data/sample_user_profile.json", "r") as f:
        return json.load(f)


@pytest.fixture
def mock_agent():
    """Fixture to create a mock compliance agent with mocked dependencies."""
    with patch("agents.compliance_agent.MCP", autospec=True) as mock_mcp:
        # Mock context to return sample data
        mock_context = MagicMock()
        mock_mcp.get_or_create_context.return_value = mock_context
        
        # Create the agent
        agent = ComplianceAgent()
        
        yield agent


class TestComplianceAgent:
    """Test cases for the ComplianceAgent class."""
    
    def test_initialization(self, mock_agent):
        """Test that the agent initializes correctly."""
        assert mock_agent is not None
        assert mock_agent.tools is not None
        assert mock_agent.agent_executor is not None
    
    def test_check_risk_suitability(self, mock_agent, sample_user_profile):
        """Test the check_risk_suitability tool."""
        # Setup the mock context to return the sample profile
        mock_agent.user_profile_context.get.return_value = sample_user_profile
        
        # Call the tool
        result = mock_agent._create_tools()[0]("High-risk cryptocurrency investment recommendation")
        
        # Verify the result contains expected risk assessment
        assert "risk" in result.lower()
        assert "suitability" in result.lower()
    
    def test_check_required_disclosures(self, mock_agent):
        """Test the check_required_disclosures tool."""
        # Call the tool with text missing disclosures
        advice = "You should invest all your money in tech stocks for maximum returns."
        result = mock_agent._create_tools()[1](advice)
        
        # Verify the result identifies missing disclosures
        assert "disclosure" in result.lower()
        assert "missing" in result.lower()
        
        # Call the tool with text containing proper disclosures
        advice_with_disclosures = (
            "Based on your risk profile, you might consider tech stocks. "
            "Please note that all investments carry risk and past performance "
            "is not indicative of future results. This is not a recommendation "
            "to buy or sell any security."
        )
        result_with_disclosures = mock_agent._create_tools()[1](advice_with_disclosures)
        
        # Verify the result acknowledges proper disclosures
        assert "disclosure" in result_with_disclosures.lower()
    
    def test_detect_misleading_statements(self, mock_agent):
        """Test the detect_misleading_statements tool."""
        # Call the tool with potentially misleading text
        advice = "This investment guaranteed to double your money in a year."
        result = mock_agent._create_tools()[2](advice)
        
        # Verify the result identifies misleading statements
        assert "misleading" in result.lower()
        assert "guaranteed" in result.lower()
    
    @patch("agents.compliance_agent.ChatOllama")
    def test_run_method(self, mock_chat_ollama, mock_agent):
        """Test the run method processes queries correctly."""
        # Setup the mock executor to return a response
        mock_agent.agent_executor.run.return_value = "Compliance check completed"
        
        # Call the run method
        result = mock_agent.run("Check compliance of this advice: Buy tech stocks now.")
        
        # Verify the executor was called with the query
        mock_agent.agent_executor.run.assert_called_once_with(
            input="Check compliance of this advice: Buy tech stocks now."
        )
        
        # Verify the result is what we expect
        assert result == "Compliance check completed"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
