# FinSage Usage Guide

This guide provides detailed instructions for using the FinSage financial advisor system, including examples, best practices, and common use cases.

## Table of Contents
- [Getting Started](#getting-started)
- [Interacting with FinSage](#interacting-with-finsage)
- [Portfolio Management](#portfolio-management)
- [Financial Planning](#financial-planning)
- [Market Data & News](#market-data--news)
- [Compliance Checks](#compliance-checks)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

Make sure you have Python 3.9+ installed, then:

```bash
# Clone the repository
git clone https://github.com/yourusername/fin_sage.git
cd fin_sage

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file with your API keys

# Install Ollama and download the Gemma model
ollama pull gemma3:4b
```

### Initial Setup

1. Start the application:

```bash
python app.py
```

2. Open your browser to the Gradio interface at: http://127.0.0.1:7860

3. If this is your first time, FinSage will use sample data. For real use, upload your portfolio data using the interface.

## Interacting with FinSage

### Chat Interface

The main way to interact with FinSage is through the chat interface. Here are some example queries:

**Market Information:**
- "What's the current price of Apple stock?"
- "How has the S&P 500 performed this week?"
- "Show me the performance of technology sector ETFs"

**Portfolio Analysis:**
- "Analyze my portfolio for diversification"
- "What's my current asset allocation?"
- "Which of my holdings has the highest dividend yield?"
- "Calculate my portfolio's risk metrics"

**Financial Planning:**
- "Create a retirement plan based on my profile"
- "How much should I be saving monthly to reach $1 million in 20 years?"
- "Suggest an asset allocation based on my risk tolerance"
- "What tax-efficient investment strategies would work for me?"

**News & Sentiment:**
- "What's the latest news about Tesla?"
- "What's the market sentiment around cryptocurrencies today?"
- "Show me the top financial news stories"
- "How is the sentiment trending for healthcare stocks?"

### Using the Sidebar

The sidebar contains tabs for:

1. **Portfolio Overview:** Visualizations of your portfolio
2. **User Profile:** Your financial information and goals
3. **Market Data:** Current market indices and sector performance
4. **News:** Latest financial news with sentiment analysis

Use the refresh buttons in each tab to get the most up-to-date information.

## Portfolio Management

### Uploading Your Portfolio

To use your actual investment data:

1. Prepare a JSON file in the format shown in `data/sample_portfolio.json`
2. Use the "Upload Portfolio" button in the interface
3. FinSage will analyze and visualize your data

### Portfolio Analysis

FinSage provides several portfolio analytics:

- **Asset Allocation:** Breakdown by asset class with pie chart
- **Sector Allocation:** Exposure to different market sectors
- **Performance Metrics:** Returns over various time periods
- **Risk Analysis:** Volatility, Sharpe ratio, max drawdown
- **Top Holdings:** Your largest positions by value

## Financial Planning

### Setting Financial Goals

Use these example queries to set and track goals:

- "Create a new financial goal to save $50,000 for a house down payment in 5 years"
- "Update my retirement goal target amount to $2 million"
- "What's my progress towards my college savings goal?"

### Retirement Planning

For retirement planning:

- "Create a retirement plan assuming I retire at 65"
- "How much monthly income can I expect in retirement based on my current savings?"
- "What if I increase my 401(k) contribution by 2%?"
- "Show me a withdrawal strategy for retirement"

## Market Data & News

### Getting Market Insights

FinSage provides comprehensive market data:

- Current values of major indices (S&P 500, Dow, NASDAQ)
- Sector performance
- Commodity prices
- Forex rates
- Cryptocurrency prices
- Key economic indicators

### News Analysis

The news sentiment analysis features:

- Aggregates news from multiple sources
- Analyzes sentiment using FinBERT model
- Tracks sentiment trends over time
- Provides ticker-specific news sentiment
- Categorizes news by sector/topic

## Compliance Checks

FinSage includes compliance features to ensure advice is appropriate:

- **Risk Suitability:** Checks if recommendations match your risk profile
- **Required Disclosures:** Ensures proper disclaimers are included
- **Factual Accuracy:** Verifies statements against data
- **Compliance Reports:** Generates detailed compliance summaries

## Advanced Features

### Scheduled Reports

FinSage can provide regular updates:

- "Schedule daily market updates at 5:00 PM"
- "Set up weekly portfolio performance reports"
- "Schedule monthly financial goal progress updates"

To manage schedules:

- "List all scheduled tasks"
- "Cancel the daily market update task"
- "Start the scheduler"
- "Stop all scheduled tasks"

### Custom Analyses

For more sophisticated analyses:

- "Compare my portfolio performance against the S&P 500"
- "Simulate the impact of a market correction on my portfolio"
- "Analyze the correlation between my holdings"
- "Generate an optimal portfolio based on my risk profile"

## Troubleshooting

### Common Issues

**Problem:** FinSage doesn't respond to queries
**Solution:** Check if the Ollama service is running (`ollama serve`)

**Problem:** Financial data seems outdated
**Solution:** Use the refresh buttons in the sidebar or ask "Update market data"

**Problem:** Portfolio visualizations don't appear
**Solution:** Run `python generate_visualizations.py` or check the `visualizations` folder permissions

**Problem:** API calls failing
**Solution:** Verify API keys in your `.env` file and check API usage limits

### Getting Help

For additional assistance:

- Check the GitHub repository issues
- Review the code documentation
- Run FinSage with debug logging: `LOG_LEVEL=DEBUG python app.py`

---

We hope this guide helps you make the most of FinSage! For further questions or suggestions, please create an issue on the GitHub repository.
