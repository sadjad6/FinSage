"""
Tests for the SchedulerAgent class.
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.scheduler_agent import SchedulerAgent


@pytest.fixture
def mock_agents():
    """Fixture to create mock agents for the scheduler to coordinate."""
    mock_market_data_agent = MagicMock()
    mock_market_data_agent.run.return_value = "Market data updated"
    
    mock_portfolio_analyzer_agent = MagicMock()
    mock_portfolio_analyzer_agent.run.return_value = "Portfolio analysis complete"
    
    mock_news_sentiment_agent = MagicMock()
    mock_news_sentiment_agent.run.return_value = "News sentiment analysis complete"
    
    mock_financial_planner_agent = MagicMock()
    mock_financial_planner_agent.run.return_value = "Financial planning complete"
    
    mock_compliance_agent = MagicMock()
    mock_compliance_agent.run.return_value = "Compliance check complete"
    
    return {
        "market_data_agent": mock_market_data_agent,
        "portfolio_analyzer_agent": mock_portfolio_analyzer_agent,
        "news_sentiment_agent": mock_news_sentiment_agent,
        "financial_planner_agent": mock_financial_planner_agent,
        "compliance_agent": mock_compliance_agent
    }


@pytest.fixture
def mock_agent(mock_agents):
    """Fixture to create a mock scheduler agent with mocked dependencies."""
    with patch("agents.scheduler_agent.get_registry", autospec=True) as mock_get_registry, \
         patch("agents.scheduler_agent.ChatOllama") as mock_chat:
        # Mock registry
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        # Create the agent with mocked agents
        scheduler = SchedulerAgent(
            market_data_agent=mock_agents["market_data_agent"],
            portfolio_analyzer_agent=mock_agents["portfolio_analyzer_agent"],
            news_sentiment_agent=mock_agents["news_sentiment_agent"],
            financial_planner_agent=mock_agents["financial_planner_agent"],
            compliance_agent=mock_agents["compliance_agent"]
        )
        
        yield scheduler


class TestSchedulerAgent:
    """Test cases for the SchedulerAgent class."""
    
    def test_initialization(self, mock_agent):
        """Test that the agent initializes correctly."""
        assert mock_agent is not None
        assert mock_agent.tools is not None
        assert mock_agent.agent_executor is not None
        assert mock_agent.scheduler is not None
        assert not mock_agent.is_running
    
    def test_schedule_daily_update(self, mock_agent, mock_agents):
        """Test the schedule_daily_update tool."""
        # Call the tool
        result = mock_agent._create_tools()[0](hour=16, minute=30)
        
        # Verify the scheduler was called
        assert len(mock_agent.scheduler.get_jobs()) > 0
        
        # Verify the result indicates scheduling success
        assert "scheduled" in result.lower()
        assert "16:30" in result
    
    def test_list_scheduled_tasks(self, mock_agent):
        """Test the list_scheduled_tasks tool."""
        # Add a job to the scheduler
        mock_agent.scheduler.add_job(
            lambda: None,
            'cron', 
            hour=16, 
            minute=30,
            id="test_job",
            name="Test Job"
        )
        
        # Call the tool
        result = mock_agent._create_tools()[1]()
        
        # Verify the result includes the test job
        assert "Test Job" in result
        assert "test_job" in result
    
    def test_cancel_scheduled_task(self, mock_agent):
        """Test the cancel_scheduled_task tool."""
        # Add a job to the scheduler
        mock_agent.scheduler.add_job(
            lambda: None,
            'cron', 
            hour=16, 
            minute=30,
            id="test_job",
            name="Test Job"
        )
        
        # Verify job exists
        assert len(mock_agent.scheduler.get_jobs()) == 1
        
        # Call the tool
        result = mock_agent._create_tools()[2]("test_job")
        
        # Verify the job was removed
        assert len(mock_agent.scheduler.get_jobs()) == 0
        
        # Verify the result indicates cancellation success
        assert "cancelled" in result.lower()
        assert "test_job" in result
    
    def test_start_scheduler(self, mock_agent):
        """Test the start_scheduler tool."""
        # Call the tool
        with patch.object(mock_agent.scheduler, 'start') as mock_start:
            result = mock_agent._create_tools()[3]()
            
            # Verify the scheduler was started
            mock_start.assert_called_once()
            
            # Verify is_running flag was set
            assert mock_agent.is_running
            
            # Verify the result indicates starting success
            assert "started" in result.lower()
    
    def test_stop_scheduler(self, mock_agent):
        """Test the stop_scheduler tool."""
        # Set is_running to True
        mock_agent.is_running = True
        
        # Call the tool
        with patch.object(mock_agent.scheduler, 'shutdown') as mock_shutdown:
            result = mock_agent._create_tools()[4]()
            
            # Verify the scheduler was stopped
            mock_shutdown.assert_called_once()
            
            # Verify is_running flag was cleared
            assert not mock_agent.is_running
            
            # Verify the result indicates stopping success
            assert "stopped" in result.lower()
    
    def test_generate_daily_summary(self, mock_agent, mock_agents):
        """Test the generate_daily_summary tool."""
        # Call the tool
        result = mock_agent._create_tools()[5]()
        
        # Verify each agent was called
        mock_agents["market_data_agent"].run.assert_called_once()
        mock_agents["portfolio_analyzer_agent"].run.assert_called_once()
        mock_agents["news_sentiment_agent"].run.assert_called_once()
        
        # Verify the result contains summary information
        assert "summary" in result.lower()
        assert "market" in result.lower()
        assert "portfolio" in result.lower()
        assert "news" in result.lower()
    
    @patch("agents.scheduler_agent.ChatOllama")
    def test_run_method(self, mock_chat_ollama, mock_agent):
        """Test the run method processes queries correctly."""
        # Setup the mock executor to return a response
        mock_agent.agent_executor.invoke.return_value = {"output": "Task scheduled successfully"}
        
        # Call the run method
        result = mock_agent.run("Schedule a daily update at 5:00 PM")
        
        # Verify the executor was called with the query
        mock_agent.agent_executor.invoke.assert_called_once_with(
            {"input": "Schedule a daily update at 5:00 PM"}
        )
        
        # Verify the result is what we expect
        assert result == "Task scheduled successfully"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
