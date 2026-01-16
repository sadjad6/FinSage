# 🧠 FinSage: AI-Powered Financial Advisor

FinSage is an autonomous multi-agent system for financial and investment advising, helping users make data-driven decisions across various asset classes including stocks, ETFs, bonds, and cryptocurrencies.

## 🌟 Features

- **Interactive Chat Interface**: User-friendly Gradio UI for natural language financial queries
- **Portfolio Analysis**: Upload and analyze your investment portfolio with visual charts
- **Real-time Market Data**: Access current market information for various assets
- **Personalized Recommendations**: Receive tailored financial advice based on your goals and risk profile
- **News Sentiment Analysis**: Get insights from financial news using FinBERT sentiment analysis
- **Scheduled Reports**: Receive automated end-of-day market summaries and periodic portfolio reviews
- **Compliance Checks**: Ensure recommendations follow regulatory guidelines and suit your risk profile
- **Data Visualization**: Portfolio allocation, sector breakdown, performance, and top holdings charts

## 🔧 System Architecture

FinSage implements a multi-agent architecture using LangChain and LangGraph with Ollama's Gemma model:

- **MarketDataAgent**: Fetches real-time financial data from various sources
- **PortfolioAnalyzerAgent**: Analyzes asset allocation, risk metrics, and performance
- **FinancialPlanningAgent**: Provides personalized recommendations and goal-based planning
- **ComplianceAgent**: Ensures recommendations follow guidelines and are suitable for user risk profiles
- **NewsSentimentAgent**: Analyzes financial news sentiment using FinBERT model
- **SchedulerAgent**: Coordinates scheduled reports and automated updates

## 💾 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fin_sage.git
   cd fin_sage
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file:
   ```
   # API Keys
   FINNHUB_API_KEY=your_finnhub_api_key
   NEWS_API_KEY=your_newsapi_key
   ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
   
   # Application Settings
   LOG_LEVEL=INFO
   VISUALIZATIONS_DIR=./visualizations
   
   # Scheduler Settings
   DAILY_UPDATE_TIME=16:30
   WEEKLY_REPORT_DAY=Friday
   
   # Ollama Model Settings
   OLLAMA_MODEL=gemma3:4b
   OLLAMA_URL=http://localhost:11434
   ```

4. Make sure you have Ollama installed and the Gemma model available:
   ```bash
   ollama pull gemma3:4b
   ```

5. Generate visualizations from sample data:
   ```bash
   python generate_visualizations.py
   ```

## 🚀 Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to the local Gradio interface (typically http://127.0.0.1:7860)

3. Start chatting with FinSage or upload your portfolio data for analysis

For detailed instructions, examples, and best practices, see the [Usage Guide](docs/usage_guide.md).

## 📊 Example Conversations

FinSage supports complex financial conversations. Here are some examples:

- **Portfolio Analysis**: "Analyze my portfolio for diversification issues"
- **Market Data**: "What's the current price of Tesla stock and how has it performed this month?"
- **Financial Planning**: "Create a retirement savings plan assuming I retire at 65"
- **News Analysis**: "What's the market sentiment around cryptocurrencies today?"
- **Scheduled Tasks**: "Schedule a daily market update at 5:00 PM"

For more comprehensive conversation examples, see [Conversation Examples](docs/conversation_examples.md).

## 📊 Visualizations

FinSage generates visualizations to help understand your portfolio:

- **Asset Allocation**: Pie chart showing distribution across asset classes
- **Sector Allocation**: Breakdown of investments by market sector
- **Performance Chart**: Historical performance across different time periods
- **Top Holdings**: Bar chart of your largest investments by value

Run `python generate_visualizations.py` to create these visualizations from sample data.

## 📁 Sample Data

The repository includes sample data files for testing and demonstration:

- `data/sample_portfolio.json`: Example investment portfolio with holdings, allocation, and performance
- `data/sample_news.json`: Mock financial news data with sentiment analysis
- `data/sample_market_data.json`: Sample market indices, sectors, commodities, and economic indicators
- `data/sample_user_profile.json`: Example user financial profile with goals and preferences

## 🧪 Testing

FinSage includes comprehensive unit tests for all agents.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_compliance_agent.py

# Run tests with coverage report
pytest --cov=agents --cov=utils --cov=contexts

# Run a specific test case
pytest tests/test_compliance_agent.py::TestComplianceAgent::test_initialization
```

### Test Structure

Each agent has a corresponding test module:

| Agent | Test File |
|-------|-----------|
| FinancialPlannerAgent | `tests/test_financial_planner_agent.py` |
| ComplianceAgent | `tests/test_compliance_agent.py` |
| NewsSentimentAgent | `tests/test_news_sentiment_agent.py` |
| SchedulerAgent | `tests/test_scheduler_agent.py` |
| PortfolioAnalyzerAgent | `tests/test_portfolio_analyzer_agent.py` |
| MarketDataAgent | `tests/test_market_data_agent.py` |

### Mocking Strategy

Tests use `unittest.mock` to mock external dependencies:

- **`get_registry`**: Mocked to provide a controlled context registry for testing agent tools
- **`ChatOllama`**: Mocked to avoid requiring a running Ollama instance during tests
- **API Clients**: External API clients (e.g., `YFinanceClient`, `NewsAPIClient`) are mocked to provide predictable test data

Example fixture pattern:
```python
@pytest.fixture
def mock_agent():
    with patch("agents.compliance_agent.get_registry") as mock_get_registry, \
         patch("agents.compliance_agent.ChatOllama") as mock_chat:
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        agent = ComplianceAgent()
        yield agent
```

## 🔄 CI/CD

The project includes GitHub Actions workflows in `.github/workflows/ci.yml` that:

1. Run automated tests on Python 3.9 and 3.10
2. Perform linting with flake8
3. Generate code coverage reports
4. Build and publish Docker image on merge to main branch

## 🐳 Docker

You can run FinSage in a Docker container:

```bash
# Build the Docker image
docker build -t finsage/advisor:latest .

# Run the container
docker run -p 7860:7860 finsage/advisor:latest
```

## 📁 Project Structure

```
fin_sage/
├── app.py                       # Gradio UI
├── agents/                      # Agent modules
│   ├── market_data_agent.py
│   ├── portfolio_analyzer_agent.py
│   ├── financial_planner_agent.py
│   ├── compliance_agent.py
│   ├── news_sentiment_agent.py
│   └── scheduler_agent.py
├── contexts/                    # MCP context modules
│   ├── user_profile_context.py
│   ├── market_context.py
│   ├── portfolio_context.py
│   └── news_context.py
├── utils/                       # Utility functions
│   ├── data_loader.py
│   ├── mcp_utils.py
│   ├── api_clients.py
│   └── visualization.py
├── data/                        # Sample data files
│   ├── sample_portfolio.json
│   ├── sample_news.json
│   ├── sample_market_data.json
│   └── sample_user_profile.json
├── tests/                       # Test modules
│   ├── test_financial_planner_agent.py
│   ├── test_compliance_agent.py
│   ├── test_news_sentiment_agent.py
│   ├── test_scheduler_agent.py
│   ├── test_portfolio_analyzer_agent.py
│   └── test_market_data_agent.py
├── docs/                        # Documentation
│   ├── usage_guide.md
│   └── conversation_examples.md
├── visualizations/              # Generated charts
├── .github/workflows/ci.yml     # CI/CD configuration
├── Dockerfile                   # Docker configuration
├── generate_visualizations.py   # Script to generate charts
├── .env                         # Environment variables
├── .gitignore                   # Git ignore file
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## 🔄 Contributing

Contributions to improve FinSage are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
