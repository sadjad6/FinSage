"""
FinSage - Autonomous Multi-Agent Financial Advisor

Main application with Gradio UI for user interaction.
"""

import os
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from agents.financial_planner_agent import FinancialPlannerAgent
from agents.compliance_agent import ComplianceAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.scheduler_agent import SchedulerAgent
from agents.market_data_agent import MarketDataAgent
from agents.portfolio_analyzer_agent import PortfolioAnalyzerAgent

from contexts.user_profile_context import UserProfileContent, RiskTolerance, IncomeType, TimeHorizon
from contexts.portfolio_context import PortfolioContextContent
from contexts.news_context import NewsContextContent
from utils.mcp_utils import get_registry, ContextWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("finsage.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

class FinSageApp:
    """Main application class for FinSage"""
    
    def __init__(self):
        """Initialize the FinSage application"""
        self.initialize_agents()
        self.load_sample_data()
        self.chat_history = []
    
    def initialize_agents(self):
        """Initialize all agents"""
        logger.info("Initializing agents...")
        try:
            # Initialize agents in order of dependency
            self.market_data_agent = MarketDataAgent()
            self.portfolio_analyzer_agent = PortfolioAnalyzerAgent(
                market_data_agent=self.market_data_agent
            )
            self.financial_planner_agent = FinancialPlannerAgent(
                market_data_agent=self.market_data_agent,
                portfolio_analyzer_agent=self.portfolio_analyzer_agent
            )
            self.compliance_agent = ComplianceAgent()
            self.news_sentiment_agent = NewsSentimentAgent()
            
            # Initialize scheduler agent last as it depends on all other agents
            self.scheduler_agent = SchedulerAgent(
                financial_planner_agent=self.financial_planner_agent,
                compliance_agent=self.compliance_agent,
                news_sentiment_agent=self.news_sentiment_agent,
                market_data_agent=self.market_data_agent,
                portfolio_analyzer_agent=self.portfolio_analyzer_agent
            )
            
            logger.info("All agents initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def load_sample_data(self):
        """Load sample data if it doesn't exist in the context"""
        logger.info("Checking for sample data...")
        registry = get_registry()
        
        # Check if user profile exists
        user_profile_context = registry.get_latest_context("user_profile_context")
        if not user_profile_context:
            logger.info("Creating sample user profile")
            self._create_sample_user_profile()
        
        # Check if portfolio exists
        portfolio_context = registry.get_latest_context("portfolio_context")
        if not portfolio_context:
            logger.info("Creating sample portfolio")
            self._create_sample_portfolio()
        
        logger.info("Sample data check complete")
    
    def _create_sample_user_profile(self):
        """Create a sample user profile"""
        try:
            # Create sample user profile
            user_profile = UserProfileContent(
                name="John Doe",
                age=35,
                income=85000,
                income_type=IncomeType.SALARY,
                tax_bracket=24,
                risk_tolerance=RiskTolerance.MODERATE,
                time_horizon=TimeHorizon.MEDIUM,
                investment_goals=["Retirement", "Home Purchase"],
                financial_goals=[
                    {
                        "name": "Retirement",
                        "target_amount": 1500000,
                        "current_amount": 125000,
                        "target_date": "2050-01-01",
                        "priority": 1,
                        "description": "Build retirement nest egg"
                    },
                    {
                        "name": "Home Down Payment",
                        "target_amount": 100000,
                        "current_amount": 35000,
                        "target_date": "2025-06-01",
                        "priority": 2,
                        "description": "Save for 20% down payment on a home"
                    }
                ],
                emergency_fund_months=3,
                retirement_accounts={
                    "401k": 85000,
                    "IRA": 40000
                },
                debt={
                    "Student Loans": {
                        "balance": 25000,
                        "interest_rate": 4.5,
                        "minimum_payment": 350
                    },
                    "Car Loan": {
                        "balance": 15000,
                        "interest_rate": 3.9,
                        "minimum_payment": 450
                    }
                }
            )
            
            # Create and register the context
            context = ContextWrapper.create(
                context_type="user_profile_context",
                creator_agent="FinSageApp",
                content_model=UserProfileContent,
                content_data=user_profile.dict()
            )
            
            registry = get_registry()
            registry.register_context(context)
            
            # Save to file for reference
            with open(DATA_DIR / "sample_user_profile.json", "w") as f:
                json.dump(user_profile.dict(), f, indent=2, default=str)
            
            logger.info("Sample user profile created")
            
        except Exception as e:
            logger.error(f"Error creating sample user profile: {e}")
            logger.error(traceback.format_exc())
    
    def _create_sample_portfolio(self):
        """Create a sample portfolio"""
        try:
            # Create sample portfolio
            portfolio = PortfolioContextContent(
                total_value=325000,
                cash_balance=15000,
                asset_allocation={
                    "Stocks": 65.0,
                    "Bonds": 20.0,
                    "Cash": 10.0,
                    "Alternatives": 5.0
                },
                holdings={
                    "AAPL": {
                        "name": "Apple Inc.",
                        "shares": 50,
                        "purchase_price": 150.75,
                        "current_price": 175.25,
                        "market_value": 8762.5,
                        "weight": 2.7,
                        "sector": "Technology",
                        "asset_class": "Stocks",
                        "daily_change_pct": 1.25
                    },
                    "MSFT": {
                        "name": "Microsoft Corp.",
                        "shares": 30,
                        "purchase_price": 280.50,
                        "current_price": 310.75,
                        "market_value": 9322.5,
                        "weight": 2.9,
                        "sector": "Technology",
                        "asset_class": "Stocks",
                        "daily_change_pct": 0.85
                    },
                    "AMZN": {
                        "name": "Amazon.com Inc.",
                        "shares": 15,
                        "purchase_price": 135.25,
                        "current_price": 142.50,
                        "market_value": 2137.5,
                        "weight": 0.7,
                        "sector": "Consumer Discretionary",
                        "asset_class": "Stocks",
                        "daily_change_pct": -0.35
                    },
                    "VTI": {
                        "name": "Vanguard Total Stock Market ETF",
                        "shares": 200,
                        "purchase_price": 210.50,
                        "current_price": 235.75,
                        "market_value": 47150.0,
                        "weight": 14.5,
                        "sector": "Diversified",
                        "asset_class": "Stocks",
                        "daily_change_pct": 0.45
                    },
                    "VXUS": {
                        "name": "Vanguard Total International Stock ETF",
                        "shares": 250,
                        "purchase_price": 55.25,
                        "current_price": 57.50,
                        "market_value": 14375.0,
                        "weight": 4.4,
                        "sector": "International",
                        "asset_class": "Stocks",
                        "daily_change_pct": -0.15
                    },
                    "VIG": {
                        "name": "Vanguard Dividend Appreciation ETF",
                        "shares": 150,
                        "purchase_price": 160.75,
                        "current_price": 172.25,
                        "market_value": 25837.5,
                        "weight": 8.0,
                        "sector": "Dividend",
                        "asset_class": "Stocks",
                        "daily_change_pct": 0.30
                    },
                    "QQQ": {
                        "name": "Invesco QQQ Trust",
                        "shares": 75,
                        "purchase_price": 350.25,
                        "current_price": 375.50,
                        "market_value": 28162.5,
                        "weight": 8.7,
                        "sector": "Technology",
                        "asset_class": "Stocks",
                        "daily_change_pct": 0.95
                    },
                    "IWM": {
                        "name": "iShares Russell 2000 ETF",
                        "shares": 80,
                        "purchase_price": 185.50,
                        "current_price": 192.75,
                        "market_value": 15420.0,
                        "weight": 4.7,
                        "sector": "Small Cap",
                        "asset_class": "Stocks",
                        "daily_change_pct": -0.25
                    },
                    "BND": {
                        "name": "Vanguard Total Bond Market ETF",
                        "shares": 300,
                        "purchase_price": 82.25,
                        "current_price": 81.50,
                        "market_value": 24450.0,
                        "weight": 7.5,
                        "sector": "Fixed Income",
                        "asset_class": "Bonds",
                        "daily_change_pct": 0.10
                    },
                    "BNDX": {
                        "name": "Vanguard Total International Bond ETF",
                        "shares": 250,
                        "purchase_price": 53.75,
                        "current_price": 54.25,
                        "market_value": 13562.5,
                        "weight": 4.2,
                        "sector": "International Fixed Income",
                        "asset_class": "Bonds",
                        "daily_change_pct": 0.05
                    },
                    "VTIP": {
                        "name": "Vanguard Short-Term Inflation-Protected Securities ETF",
                        "shares": 200,
                        "purchase_price": 51.25,
                        "current_price": 51.75,
                        "market_value": 10350.0,
                        "weight": 3.2,
                        "sector": "Fixed Income",
                        "asset_class": "Bonds",
                        "daily_change_pct": 0.15
                    },
                    "MUB": {
                        "name": "iShares National Muni Bond ETF",
                        "shares": 150,
                        "purchase_price": 106.75,
                        "current_price": 108.25,
                        "market_value": 16237.5,
                        "weight": 5.0,
                        "sector": "Municipal",
                        "asset_class": "Bonds",
                        "daily_change_pct": 0.08
                    },
                    "VGIT": {
                        "name": "Vanguard Intermediate-Term Treasury ETF",
                        "shares": 175,
                        "purchase_price": 65.50,
                        "current_price": 64.75,
                        "market_value": 11331.25,
                        "weight": 3.5,
                        "sector": "Government",
                        "asset_class": "Bonds",
                        "daily_change_pct": -0.05
                    },
                    "GLD": {
                        "name": "SPDR Gold Shares",
                        "shares": 40,
                        "purchase_price": 175.25,
                        "current_price": 182.50,
                        "market_value": 7300.0,
                        "weight": 2.2,
                        "sector": "Commodities",
                        "asset_class": "Alternatives",
                        "daily_change_pct": 0.65
                    },
                    "VNQ": {
                        "name": "Vanguard Real Estate ETF",
                        "shares": 60,
                        "purchase_price": 95.50,
                        "current_price": 92.75,
                        "market_value": 5565.0,
                        "weight": 1.7,
                        "sector": "Real Estate",
                        "asset_class": "Alternatives",
                        "daily_change_pct": -0.45
                    }
                },
                performance={
                    "1d": 0.25,
                    "1w": 0.75,
                    "1m": 1.35,
                    "3m": 2.85,
                    "6m": 5.25,
                    "ytd": 7.45,
                    "1y": 12.65,
                    "3y": 35.85,
                    "5y": 52.45,
                    "10y": 125.65
                },
                sector_allocation={
                    "Technology": 18.9,
                    "Consumer Discretionary": 0.7,
                    "Diversified": 14.5,
                    "International": 4.4,
                    "Dividend": 8.0,
                    "Small Cap": 4.7,
                    "Fixed Income": 10.7,
                    "International Fixed Income": 4.2,
                    "Municipal": 5.0,
                    "Government": 3.5,
                    "Commodities": 2.2,
                    "Real Estate": 1.7,
                    "Cash": 10.0
                },
                risk_metrics={
                    "sharpe_ratio": 0.85,
                    "sortino_ratio": 1.15,
                    "max_drawdown": -12.5,
                    "volatility": 9.75,
                    "beta": 0.92,
                    "alpha": 1.25
                },
                last_updated=datetime.now()
            )
            
            # Create and register the context
            context = ContextWrapper.create(
                context_type="portfolio_context",
                creator_agent="FinSageApp",
                content_model=PortfolioContextContent,
                content_data=portfolio.dict()
            )
            
            registry = get_registry()
            registry.register_context(context)
            
            # Save to file for reference
            with open(DATA_DIR / "sample_portfolio.json", "w") as f:
                json.dump(portfolio.dict(), f, indent=2, default=str)
            
            logger.info("Sample portfolio created")
            
        except Exception as e:
            logger.error(f"Error creating sample portfolio: {e}")
            logger.error(traceback.format_exc())
    
    def process_message(self, message, history):
        """Process a user message and generate a response"""
        logger.info(f"Processing message: {message}")
        
        try:
            # First, check for specific commands or keywords
            message_lower = message.lower()
            
            # Process command-based requests
            if "portfolio" in message_lower and any(keyword in message_lower for keyword in ["show", "view", "display", "summary"]):
                return self._handle_portfolio_request()
            
            elif "profile" in message_lower and any(keyword in message_lower for keyword in ["show", "view", "display", "summary"]):
                return self._handle_profile_request()
            
            elif "market" in message_lower and any(keyword in message_lower for keyword in ["data", "update", "info", "summary"]):
                return self._handle_market_request(message)
            
            elif "news" in message_lower and any(keyword in message_lower for keyword in ["latest", "update", "sentiment", "show"]):
                return self._handle_news_request(message)
            
            elif "schedule" in message_lower and any(keyword in message_lower for keyword in ["task", "update", "list", "daily"]):
                return self._handle_scheduler_request(message)
            
            # For financial advice and planning, use the financial planner agent
            elif any(keyword in message_lower for keyword in ["advice", "recommend", "suggest", "plan", "goal", "retire", "saving"]):
                response = self.financial_planner_agent.run(message)
                
                # Run compliance check on the advice
                compliance_review = self.compliance_agent.run(response, "advice")
                
                # If there are compliance issues, append the review
                if "COMPLIANCE ISSUE" in compliance_review:
                    response += f"\n\n## Compliance Review\n{compliance_review}"
                
                return response
            
            # Default to financial planner for general queries
            else:
                return self.financial_planner_agent.run(message)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(traceback.format_exc())
            return f"I'm sorry, I encountered an error processing your request: {str(e)}"
    
    def _handle_portfolio_request(self):
        """Handle a request to view portfolio information"""
        try:
            # Get the portfolio analysis from the portfolio analyzer
            response = self.portfolio_analyzer_agent.run("Generate a detailed portfolio summary with key metrics")
            
            # Create visualizations for the portfolio
            self._create_portfolio_visualizations()
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling portfolio request: {e}")
            return f"Error retrieving portfolio information: {str(e)}"
    
    def _handle_profile_request(self):
        """Handle a request to view user profile information"""
        try:
            # Get the user profile information
            response = self.financial_planner_agent.run("Get and format my complete user profile information")
            return response
            
        except Exception as e:
            logger.error(f"Error handling profile request: {e}")
            return f"Error retrieving user profile information: {str(e)}"
    
    def _handle_market_request(self, message):
        """Handle a request for market data"""
        try:
            # Get market data from the market data agent
            response = self.market_data_agent.run(message)
            return response
            
        except Exception as e:
            logger.error(f"Error handling market request: {e}")
            return f"Error retrieving market data: {str(e)}"
    
    def _handle_news_request(self, message):
        """Handle a request for news and sentiment"""
        try:
            # Get news and sentiment from the news sentiment agent
            response = self.news_sentiment_agent.run(message)
            return response
            
        except Exception as e:
            logger.error(f"Error handling news request: {e}")
            return f"Error retrieving news data: {str(e)}"
    
    def _handle_scheduler_request(self, message):
        """Handle a scheduler-related request"""
        try:
            # Forward to the scheduler agent
            response = self.scheduler_agent.run(message)
            return response
            
        except Exception as e:
            logger.error(f"Error handling scheduler request: {e}")
            return f"Error processing scheduler request: {str(e)}"
    
    def _create_portfolio_visualizations(self):
        """Create visualizations for the portfolio"""
        try:
            registry = get_registry()
            portfolio_context = registry.get_latest_context("portfolio_context")
            
            if not portfolio_context:
                logger.warning("No portfolio context found for visualizations")
                return
            
            portfolio = portfolio_context.content
            
            # Create asset allocation pie chart
            if portfolio.asset_allocation:
                fig_asset = px.pie(
                    names=list(portfolio.asset_allocation.keys()),
                    values=list(portfolio.asset_allocation.values()),
                    title="Asset Allocation",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.3
                )
                fig_asset.update_layout(margin=dict(t=40, b=40, l=40, r=40))
                fig_asset.write_image(DATA_DIR / "asset_allocation.png")
            
            # Create sector allocation pie chart
            if portfolio.sector_allocation:
                fig_sector = px.pie(
                    names=list(portfolio.sector_allocation.keys()),
                    values=list(portfolio.sector_allocation.values()),
                    title="Sector Allocation",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.3
                )
                fig_sector.update_layout(margin=dict(t=40, b=40, l=40, r=40))
                fig_sector.write_image(DATA_DIR / "sector_allocation.png")
            
            # Create performance bar chart
            if portfolio.performance:
                periods = list(portfolio.performance.keys())
                returns = list(portfolio.performance.values())
                
                fig_perf = px.bar(
                    x=periods,
                    y=returns,
                    title="Portfolio Performance",
                    labels={"x": "Time Period", "y": "Return (%)"},
                    color=returns,
                    color_continuous_scale="RdYlGn"
                )
                fig_perf.update_layout(margin=dict(t=40, b=40, l=40, r=40))
                fig_perf.write_image(DATA_DIR / "performance.png")
            
            # Create top holdings bar chart
            if portfolio.holdings:
                # Get top 10 holdings by market value
                top_holdings = sorted(
                    [(ticker, holding.market_value) for ticker, holding in portfolio.holdings.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                tickers = [item[0] for item in top_holdings]
                values = [item[1] for item in top_holdings]
                
                fig_holdings = px.bar(
                    x=tickers,
                    y=values,
                    title="Top 10 Holdings",
                    labels={"x": "Ticker", "y": "Market Value ($)"},
                    color=values,
                    color_continuous_scale="Blues"
                )
                fig_holdings.update_layout(margin=dict(t=40, b=40, l=40, r=40))
                fig_holdings.write_image(DATA_DIR / "top_holdings.png")
            
            logger.info("Portfolio visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating portfolio visualizations: {e}")
            logger.error(traceback.format_exc())

# Create the Gradio interface
def create_gradio_interface():
    """Create and launch the Gradio interface"""
    try:
        # Initialize the app
        app = FinSageApp()
        
        # Create logo and branding
        with gr.Blocks(title="FinSage - AI Financial Advisor", theme=gr.themes.Soft()) as demo:
            with gr.Row():
                gr.Markdown(
                    """
                    # FinSage: Autonomous Multi-Agent Financial Advisor
                    *Your personal AI-powered financial advisor and portfolio manager*
                    """
                )
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        value=[],
                        show_label=False,
                        height=600,
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask me about your finances, portfolio, or for financial advice...",
                            show_label=False,
                            scale=9
                        )
                        submit = gr.Button("Send", scale=1)
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            "Show me my portfolio summary",
                            "What's my current asset allocation?",
                            "How is the market performing today?",
                            "What's the latest financial news?",
                            "Can you recommend an investment strategy based on my profile?",
                            "How much should I be saving for retirement?",
                            "Create a financial goal for a home purchase in 5 years",
                            "What's my risk tolerance and what does it mean?",
                            "Schedule daily updates at 8:00 AM",
                            "Generate a daily financial summary"
                        ],
                        inputs=msg,
                        label="Example Queries"
                    )
                
                with gr.Column(scale=1):
                    # Sidebar with tabs for different views
                    with gr.Tabs():
                        with gr.TabItem("Portfolio"):
                            gr.Markdown("### Portfolio Overview")
                            with gr.Accordion("Asset Allocation", open=True):
                                asset_img = gr.Image(label="Asset Allocation", visible=True)
                            
                            with gr.Accordion("Sector Allocation", open=False):
                                sector_img = gr.Image(label="Sector Allocation", visible=True)
                            
                            with gr.Accordion("Performance", open=False):
                                perf_img = gr.Image(label="Performance", visible=True)
                            
                            with gr.Accordion("Top Holdings", open=False):
                                holdings_img = gr.Image(label="Top Holdings", visible=True)
                            
                            refresh_btn = gr.Button("Refresh Portfolio Data")
                        
                        with gr.TabItem("Profile"):
                            gr.Markdown("### User Profile")
                            profile_output = gr.Markdown()
                            
                            view_profile_btn = gr.Button("View Profile")
                        
                        with gr.TabItem("Market"):
                            gr.Markdown("### Market Data")
                            market_output = gr.Markdown()
                            
                            update_market_btn = gr.Button("Update Market Data")
                        
                        with gr.TabItem("News"):
                            gr.Markdown("### Latest Financial News")
                            news_output = gr.Markdown()
                            
                            update_news_btn = gr.Button("Get Latest News")
            
            # Set up event handlers
            def respond(message, chat_history):
                response = app.process_message(message, chat_history)
                chat_history.append((message, response))
                return "", chat_history
            
            def refresh_portfolio():
                app._create_portfolio_visualizations()
                
                # Load images
                asset_path = DATA_DIR / "asset_allocation.png"
                sector_path = DATA_DIR / "sector_allocation.png"
                perf_path = DATA_DIR / "performance.png"
                holdings_path = DATA_DIR / "top_holdings.png"
                
                images = []
                for path in [asset_path, sector_path, perf_path, holdings_path]:
                    if path.exists():
                        images.append(Image.open(path))
                    else:
                        images.append(None)
                
                return images
            
            def view_profile():
                response = app._handle_profile_request()
                return response
            
            def update_market():
                response = app._handle_market_request("Show me the latest market data summary")
                return response
            
            def update_news():
                response = app._handle_news_request("Show me the latest financial news and market sentiment")
                return response
            
            # Register event handlers
            submit.click(respond, [msg, chatbot], [msg, chatbot])
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            
            refresh_btn.click(refresh_portfolio, [], [asset_img, sector_img, perf_img, holdings_img])
            view_profile_btn.click(view_profile, [], [profile_output])
            update_market_btn.click(update_market, [], [market_output])
            update_news_btn.click(update_news, [], [news_output])
            
            # Initialize portfolio visualizations
            app._create_portfolio_visualizations()
            
            # Load initial images if they exist
            demo.load(
                refresh_portfolio, 
                [], 
                [asset_img, sector_img, perf_img, holdings_img]
            )
            
            # Load initial profile data
            demo.load(
                view_profile,
                [],
                [profile_output]
            )
        
        return demo
    
    except Exception as e:
        logger.error(f"Error creating Gradio interface: {e}")
        logger.error(traceback.format_exc())
        raise

# Run the application
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=False)
