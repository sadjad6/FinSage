"""
Tests for the NewsSentimentAgent class.
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.news_sentiment_agent import NewsSentimentAgent


@pytest.fixture
def sample_news_data():
    """Fixture to load sample news data for testing."""
    with open("data/sample_news.json", "r") as f:
        return json.load(f)


@pytest.fixture
def mock_agent():
    """Fixture to create a mock news sentiment agent with mocked dependencies."""
    with patch("agents.news_sentiment_agent.get_registry", autospec=True) as mock_get_registry, \
         patch("agents.news_sentiment_agent.ChatOllama") as mock_chat:
        # Mock registry
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        # Mock context to return sample data
        mock_context = MagicMock()
        mock_registry.get_latest_context.return_value = mock_context
        
        # Mock news client
        mock_news_client = MagicMock()
        
        # Create the agent with mocked dependencies
        with patch("agents.news_sentiment_agent.NewsAPIClient", return_value=mock_news_client):
            with patch("agents.news_sentiment_agent.pipeline", return_value=MagicMock()):
                agent = NewsSentimentAgent()
                
                yield agent


class TestNewsSentimentAgent:
    """Test cases for the NewsSentimentAgent class."""
    
    def test_initialization(self, mock_agent):
        """Test that the agent initializes correctly."""
        assert mock_agent is not None
        assert mock_agent.tools is not None
        assert mock_agent.agent_executor is not None
        assert mock_agent.news_client is not None
    
    def test_fetch_latest_news(self, mock_agent):
        """Test the fetch_latest_news tool."""
        # Setup mock news client to return sample articles
        mock_agent.news_client.get_news_for_query.return_value = [
            {
                "title": "Test Article",
                "url": "https://example.com/news/1",
                "source": "Test Source",
                "published_at": "2023-06-01T10:00:00Z",
                "content": "This is a test article about finance."
            }
        ]
        
        # Call the tool
        tool = next(t for t in mock_agent.tools if t.name == "fetch_latest_news")
        result = tool.run({"query": "tesla", "category": "company", "max_results": 1})
        
        # Verify the news client was called
        mock_agent.news_client.get_news_for_query.assert_called_once_with(
            query="tesla", category="company", max_results=1
        )
        
        # Verify the result contains expected news information
        assert "Test Article" in result
        assert "https://example.com/news/1" in result
    
    def test_analyze_news_sentiment(self, mock_agent):
        """Test the analyze_news_sentiment tool."""
        # Setup mock sentiment analyzer
        mock_agent.sentiment_analyzer.return_value = [{"label": "positive", "score": 0.75}]
        
        # Call the tool
        tool = next(t for t in mock_agent.tools if t.name == "analyze_news_sentiment")
        result = tool.run({"text": "Tesla announced record quarterly earnings today."})
        
        # Verify the sentiment analyzer was called
        mock_agent.sentiment_analyzer.assert_called_once()
        
        # Verify the result contains sentiment information
        assert "sentiment" in result.lower()
        assert "positive" in result
    
    def test_get_market_sentiment_summary(self, mock_agent, sample_news_data):
        """Test the get_market_sentiment_summary tool."""
        # Setup mock news context to return sample data
        mock_agent.news_context.get.return_value = sample_news_data
        
        # Call the tool
        tool = next(t for t in mock_agent.tools if t.name == "get_market_sentiment_summary")
        result = tool.run({})
        
        # Verify the context was accessed
        mock_agent.news_context.get.assert_called_once()
        
        # Verify the result contains sentiment summary information
        assert "sentiment" in result.lower()
        assert "market" in result.lower()
    
    @patch("agents.news_sentiment_agent.ChatOllama")
    def test_run_method(self, mock_chat_ollama, mock_agent):
        """Test the run method processes queries correctly."""
        # Setup the mock executor to return a response
        mock_agent.agent_executor.invoke.return_value = {"output": "News sentiment analysis completed"}
        
        # Call the run method
        result = mock_agent.run("What's the sentiment around Tesla stock?")
        
        # Verify the executor was called with the query
        mock_agent.agent_executor.invoke.assert_called_once_with(
            {"input": "What's the sentiment around Tesla stock?"}
        )
        
        # Verify the result is what we expect
        assert result == "News sentiment analysis completed"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
