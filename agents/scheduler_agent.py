"""
Scheduler Agent for FinSage

This agent is responsible for coordinating activities between agents,
scheduling regular tasks, and generating end-of-day reports.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from threading import Thread, Event
from apscheduler.schedulers.background import BackgroundScheduler
from typing import Dict, List, Optional, Any, Union, Tuple, Callable

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama

from utils.mcp_utils import ContextWrapper, get_registry
from contexts.user_profile_context import UserProfileContent
from contexts.portfolio_context import PortfolioContextContent
from contexts.news_context import NewsContextContent

# Configure logger
logger = logging.getLogger(__name__)

class SchedulerAgent:
    """Agent for scheduling tasks and coordinating between agents"""
    
    def __init__(
        self, 
        financial_planner_agent=None, 
        compliance_agent=None, 
        news_sentiment_agent=None,
        market_data_agent=None,
        portfolio_analyzer_agent=None
    ):
        """
        Initialize the scheduler agent
        
        Args:
            financial_planner_agent: FinancialPlannerAgent instance
            compliance_agent: ComplianceAgent instance
            news_sentiment_agent: NewsSentimentAgent instance
            market_data_agent: MarketDataAgent instance
            portfolio_analyzer_agent: PortfolioAnalyzerAgent instance
        """
        self.agent_name = "SchedulerAgent"
        self.model = ChatOllama(model="gemma3:4b")
        
        # Store references to other agents
        self.financial_planner_agent = financial_planner_agent
        self.compliance_agent = compliance_agent
        self.news_sentiment_agent = news_sentiment_agent
        self.market_data_agent = market_data_agent
        self.portfolio_analyzer_agent = portfolio_analyzer_agent
        
        # Scheduled tasks storage
        self.scheduled_tasks = {}
        self.stop_event = Event()
        self.scheduler_thread = None
        self.scheduler = BackgroundScheduler()
        
        # Set up tools for the agent
        self.tools = self._create_tools()
        
        # Set up the agent executor
        self.agent_executor = self._create_agent_executor()
    
    @property
    def is_running(self) -> bool:
        """Check if the scheduler is running"""
        return self.scheduler.running if self.scheduler else False
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools and prompt"""
        # Set up the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a scheduler agent responsible for coordinating activities 
            between financial advisor agents, scheduling regular tasks, and generating
            end-of-day reports. 
            
            Your responsibilities include:
            1. Scheduling regular market data updates
            2. Scheduling portfolio analysis
            3. Scheduling news sentiment analysis
            4. Generating daily summaries
            5. Coordinating tasks between agents
            
            You have access to multiple agents:
            - Financial Planner Agent: Provides financial planning and advice
            - Compliance Agent: Ensures compliance with regulations
            - News Sentiment Agent: Analyzes financial news sentiment
            - Market Data Agent: Fetches market data
            - Portfolio Analyzer Agent: Analyzes portfolio performance
            
            Format your responses in markdown for better readability.
            """  
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent
        agent = {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            )
        } | prompt | self.model | OpenAIFunctionsAgentOutputParser()
        
        # Create the agent executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def run(self, query: str) -> str:
        """Run the scheduler agent with a user query"""
        try:
            response = self.agent_executor.invoke({"input": query})
            return response["output"]
        except Exception as e:
            logger.error(f"Error running scheduler agent: {e}")
            return f"Error executing scheduler operation: {str(e)}"
    
    def start_scheduler(self):
        """Start the scheduler thread to execute scheduled tasks"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            return "Scheduler is already running"
        
        self.stop_event.clear()
        self.scheduler_thread = Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        return "Scheduler started successfully"
    
    def stop_scheduler(self):
        """Stop the scheduler thread"""
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            return "Scheduler is not running"
        
        self.stop_event.set()
        self.scheduler_thread.join(timeout=5)
        
        return "Scheduler stopped successfully"
    
    def _scheduler_loop(self):
        """Main scheduler loop to execute scheduled tasks"""
        logger.info("Scheduler loop started")
        
        while not self.stop_event.is_set():
            current_time = datetime.now(timezone.utc)
            
            # Check each scheduled task
            tasks_to_remove = []
            for task_id, task in self.scheduled_tasks.items():
                next_run = task.get("next_run")
                
                if next_run and current_time >= next_run:
                    try:
                        # Execute the task
                        logger.info(f"Executing scheduled task: {task['name']}")
                        task["function"]()
                        
                        # Update next run time for recurring tasks
                        if task.get("recurrence"):
                            task["next_run"] = current_time + task["recurrence"]
                            logger.info(f"Next run scheduled for: {task['next_run']}")
                        else:
                            # One-time task, mark for removal
                            tasks_to_remove.append(task_id)
                    except Exception as e:
                        logger.error(f"Error executing scheduled task {task['name']}: {e}")
            
            # Remove completed one-time tasks
            for task_id in tasks_to_remove:
                del self.scheduled_tasks[task_id]
            
            # Sleep for a short time to prevent CPU hogging
            time.sleep(10)  # Check every 10 seconds
    
    def schedule_task(
        self, 
        name: str, 
        function: Callable, 
        run_at: datetime = None, 
        recurrence: timedelta = None
    ) -> str:
        """
        Schedule a task to run at a specific time with optional recurrence
        
        Args:
            name: Name of the task
            function: Function to execute
            run_at: When to run the task (defaults to now if not specified)
            recurrence: How often to repeat the task (None for one-time tasks)
            
        Returns:
            task_id: ID of the scheduled task
        """
        if run_at is None:
            run_at = datetime.now(timezone.utc)
        
        # Ensure datetime has timezone info
        if run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=timezone.utc)
        
        task_id = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.scheduled_tasks[task_id] = {
            "name": name,
            "function": function,
            "next_run": run_at,
            "recurrence": recurrence,
            "created_at": datetime.now(timezone.utc)
        }
        
        logger.info(f"Scheduled task '{name}' with ID {task_id} to run at {run_at}")
        
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            bool: True if the task was cancelled, False if not found
        """
        if task_id in self.scheduled_tasks:
            task = self.scheduled_tasks[task_id]
            logger.info(f"Cancelling scheduled task: {task['name']} (ID: {task_id})")
            del self.scheduled_tasks[task_id]
            return True
        
        return False
    
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for the agent to use"""
        tools = []
        
        @tool("schedule_daily_updates")
        def schedule_daily_updates(hour: int = 8, minute: int = 0) -> str:
            """
            Schedule daily updates for market data, portfolio analysis, and news.
            
            Args:
                hour: Hour of day to run updates (0-23)
                minute: Minute of hour to run updates (0-59)
            """
            if not all([
                self.market_data_agent, 
                self.portfolio_analyzer_agent, 
                self.news_sentiment_agent
            ]):
                return "Cannot schedule updates: Missing required agent references"
            
            # Calculate the next run time
            now = datetime.now(timezone.utc)
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If the time for today has already passed, schedule for tomorrow
            if next_run <= now:
                next_run += timedelta(days=1)
            
            # Schedule market data update
            market_task_id = self.schedule_task(
                name="Daily Market Data Update",
                function=lambda: self.market_data_agent.run("Update all market data"),
                run_at=next_run,
                recurrence=timedelta(days=1)
            )
            
            # Schedule portfolio analysis (after market data update)
            portfolio_next_run = next_run + timedelta(minutes=15)
            portfolio_task_id = self.schedule_task(
                name="Daily Portfolio Analysis",
                function=lambda: self.portfolio_analyzer_agent.run("Run full portfolio analysis"),
                run_at=portfolio_next_run,
                recurrence=timedelta(days=1)
            )
            
            # Schedule news sentiment analysis
            news_next_run = next_run + timedelta(minutes=30)
            news_task_id = self.schedule_task(
                name="Daily News Sentiment Analysis",
                function=lambda: self.news_sentiment_agent.run("Fetch latest financial news and analyze market sentiment"),
                run_at=news_next_run,
                recurrence=timedelta(days=1)
            )
            
            # Schedule end-of-day summary
            summary_next_run = now.replace(hour=17, minute=0, second=0, microsecond=0)
            if summary_next_run <= now:
                summary_next_run += timedelta(days=1)
                
            summary_task_id = self.schedule_task(
                name="End-of-Day Summary",
                function=lambda: self.generate_daily_summary(),
                run_at=summary_next_run,
                recurrence=timedelta(days=1)
            )
            
            # Start the scheduler if not already running
            self.start_scheduler()
            
            return f"""
### Daily Updates Scheduled
- Market Data Update: Daily at {hour:02d}:{minute:02d} (Task ID: {market_task_id})
- Portfolio Analysis: Daily at {(portfolio_next_run.hour):02d}:{portfolio_next_run.minute:02d} (Task ID: {portfolio_task_id})
- News Sentiment Analysis: Daily at {(news_next_run.hour):02d}:{news_next_run.minute:02d} (Task ID: {news_task_id})
- End-of-Day Summary: Daily at 17:00 (Task ID: {summary_task_id})

Scheduler is now running. These tasks will execute at their scheduled times.
"""
        
        @tool("list_scheduled_tasks")
        def list_scheduled_tasks() -> str:
            """
            List all currently scheduled tasks
            """
            if not self.scheduled_tasks:
                return "No tasks are currently scheduled."
            
            now = datetime.now(timezone.utc)
            
            # Format task list
            tasks_list = ["## Currently Scheduled Tasks", ""]
            tasks_list.append("| Task Name | Next Run | Recurrence | Status |")
            tasks_list.append("|-----------|----------|------------|--------|")
            
            for task_id, task in self.scheduled_tasks.items():
                name = task.get("name", "Unnamed")
                next_run = task.get("next_run")
                
                if next_run:
                    next_run_str = next_run.strftime("%Y-%m-%d %H:%M")
                    time_diff = next_run - now
                    
                    if time_diff.total_seconds() < 0:
                        status = "Overdue"
                    elif time_diff.total_seconds() < 3600:  # Less than 1 hour
                        status = "Soon"
                    else:
                        status = "Scheduled"
                else:
                    next_run_str = "N/A"
                    status = "Unknown"
                
                recurrence = task.get("recurrence")
                if recurrence:
                    days = recurrence.days
                    hours = recurrence.seconds // 3600
                    minutes = (recurrence.seconds % 3600) // 60
                    
                    if days > 0:
                        recurrence_str = f"Every {days} day(s)"
                    elif hours > 0:
                        recurrence_str = f"Every {hours} hour(s)"
                    elif minutes > 0:
                        recurrence_str = f"Every {minutes} minute(s)"
                    else:
                        recurrence_str = f"Every {recurrence.seconds} second(s)"
                else:
                    recurrence_str = "One-time"
                
                tasks_list.append(f"| {name} | {next_run_str} | {recurrence_str} | {status} |")
            
            scheduler_status = "Running" if (self.scheduler_thread and self.scheduler_thread.is_alive()) else "Stopped"
            tasks_list.append(f"\n**Scheduler Status**: {scheduler_status}")
            
            return "\n".join(tasks_list)
        
        @tool("cancel_scheduled_task")
        def cancel_scheduled_task(task_id: str) -> str:
            """
            Cancel a scheduled task by ID
            
            Args:
                task_id: ID of the task to cancel
            """
            if task_id in self.scheduled_tasks:
                task_name = self.scheduled_tasks[task_id].get("name", "Unnamed task")
                if self.cancel_task(task_id):
                    return f"Successfully cancelled scheduled task: {task_name} (ID: {task_id})"
            
            return f"Task with ID '{task_id}' was not found"
        
        @tool("start_scheduler")
        def start_scheduler_tool() -> str:
            """
            Start the scheduler to execute scheduled tasks
            """
            return self.start_scheduler()
        
        @tool("stop_scheduler")
        def stop_scheduler_tool() -> str:
            """
            Stop the scheduler from executing scheduled tasks
            """
            return self.stop_scheduler()
        
        @tool("generate_daily_summary")
        def generate_daily_summary() -> str:
            """
            Generate a comprehensive daily summary of portfolio performance,
            market data, and news sentiment
            """
            try:
                summary = ["# Daily Financial Summary", ""]
                summary.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
                summary.append("---")
                
                # Get latest contexts
                registry = get_registry()
                user_profile_context = registry.get_latest_context("user_profile_context")
                portfolio_context = registry.get_latest_context("portfolio_context")
                news_context = registry.get_latest_context("news_context")
                market_context = registry.get_latest_context("market_context")
                
                # Add user profile summary if available
                if user_profile_context:
                    user_profile = user_profile_context.content
                    summary.append("## User Profile")
                    summary.append(f"**Name**: {user_profile.name}")
                    summary.append(f"**Risk Tolerance**: {user_profile.risk_tolerance.value.title()}")
                    
                    if user_profile.financial_goals:
                        summary.append("\n### Financial Goals")
                        for goal in user_profile.financial_goals:
                            progress = (goal.current_amount / goal.target_amount) * 100 if goal.target_amount > 0 else 0
                            summary.append(f"- **{goal.name}**: ${goal.current_amount:,.2f} / ${goal.target_amount:,.2f} ({progress:.1f}% complete)")
                    
                    summary.append("\n---")
                
                # Add portfolio summary if available
                if portfolio_context:
                    portfolio = portfolio_context.content
                    summary.append("## Portfolio Summary")
                    
                    total_value = portfolio.total_value
                    summary.append(f"**Total Value**: ${total_value:,.2f}")
                    
                    if hasattr(portfolio, "daily_change_pct") and portfolio.daily_change_pct is not None:
                        change_symbol = "↑" if portfolio.daily_change_pct >= 0 else "↓"
                        summary.append(f"**Daily Change**: {change_symbol} {abs(portfolio.daily_change_pct):.2f}%")
                    
                    if hasattr(portfolio, "total_gain_loss") and portfolio.total_gain_loss is not None:
                        gl_symbol = "+" if portfolio.total_gain_loss >= 0 else "-"
                        summary.append(f"**Total Gain/Loss**: {gl_symbol}${abs(portfolio.total_gain_loss):,.2f}")
                    
                    # Asset allocation
                    if portfolio.asset_allocation:
                        summary.append("\n### Asset Allocation")
                        for asset_class, percentage in portfolio.asset_allocation.items():
                            summary.append(f"- **{asset_class}**: {percentage:.1f}%")
                    
                    # Top holdings
                    if portfolio.holdings:
                        summary.append("\n### Top Holdings")
                        
                        # Sort holdings by value
                        sorted_holdings = sorted(
                            portfolio.holdings.items(),
                            key=lambda x: x[1].market_value if hasattr(x[1], "market_value") else 0,
                            reverse=True
                        )
                        
                        for i, (ticker, holding) in enumerate(sorted_holdings[:5], 1):
                            if hasattr(holding, "market_value") and hasattr(holding, "daily_change_pct"):
                                change_symbol = "↑" if holding.daily_change_pct >= 0 else "↓"
                                summary.append(f"- **{ticker}**: ${holding.market_value:,.2f} ({change_symbol} {abs(holding.daily_change_pct):.2f}%)")
                    
                    summary.append("\n---")
                
                # Add market summary if available
                if market_context:
                    market = market_context.content
                    summary.append("## Market Summary")
                    
                    # Market indices
                    if hasattr(market, "market_indices") and market.market_indices:
                        summary.append("### Market Indices")
                        for index_name, index_data in market.market_indices.items():
                            if hasattr(index_data, "price") and hasattr(index_data, "change_pct"):
                                change_symbol = "↑" if index_data.change_pct >= 0 else "↓"
                                summary.append(f"- **{index_name}**: {index_data.price:,.2f} ({change_symbol} {abs(index_data.change_pct):.2f}%)")
                    
                    # Economic indicators
                    if hasattr(market, "economic_indicators") and market.economic_indicators:
                        summary.append("\n### Economic Indicators")
                        for indicator, value in market.economic_indicators.items():
                            summary.append(f"- **{indicator}**: {value}")
                    
                    summary.append("\n---")
                
                # Add news sentiment summary if available
                if news_context:
                    news = news_context.content
                    summary.append("## News Sentiment")
                    
                    if hasattr(news, "market_sentiment"):
                        sentiment_label = "Neutral"
                        sentiment_emoji = "🟡"
                        
                        if news.market_sentiment > 0.2:
                            sentiment_label = "Positive"
                            sentiment_emoji = "🟢"
                        elif news.market_sentiment < -0.2:
                            sentiment_label = "Negative"
                            sentiment_emoji = "🔴"
                        
                        summary.append(f"**Overall Market Sentiment**: {sentiment_label} {sentiment_emoji} ({news.market_sentiment:.2f})")
                    
                    # Top news by category
                    if hasattr(news, "category_sentiments") and news.category_sentiments:
                        summary.append("\n### Sentiment by Category")
                        for category, cat_sentiment in news.category_sentiments.items():
                            cat_emoji = "🟡"
                            if cat_sentiment.sentiment_score > 0.2:
                                cat_emoji = "🟢"
                            elif cat_sentiment.sentiment_score < -0.2:
                                cat_emoji = "🔴"
                            
                            summary.append(f"- **{category.title()}**: {cat_sentiment.sentiment_label.title()} {cat_emoji} ({cat_sentiment.sentiment_score:.2f})")
                    
                    # Recent significant news
                    if hasattr(news, "articles") and news.articles:
                        summary.append("\n### Recent Significant News")
                        
                        # Sort articles by absolute sentiment score to get most impactful
                        significant_articles = sorted(
                            news.articles.values(),
                            key=lambda x: abs(x.sentiment_score),
                            reverse=True
                        )[:3]
                        
                        for i, article in enumerate(significant_articles, 1):
                            sentiment_emoji = "🟡"
                            if article.sentiment_label == "positive":
                                sentiment_emoji = "🟢"
                            elif article.sentiment_label == "negative":
                                sentiment_emoji = "🔴"
                            
                            summary.append(f"{i}. **{article.title}** {sentiment_emoji}")
                            summary.append(f"   Source: {article.source}")
                    
                    summary.append("\n---")
                
                # Add recommendations and next steps
                summary.append("## Recommendations and Next Steps")
                
                # Generate recommendations based on available data
                recommendations = []
                
                if portfolio_context and news_context:
                    portfolio = portfolio_context.content
                    news = news_context.content
                    
                    # Portfolio rebalancing recommendation
                    if hasattr(portfolio, "needs_rebalancing") and portfolio.needs_rebalancing:
                        recommendations.append("- **Portfolio Rebalancing**: Your portfolio may need rebalancing to maintain your target asset allocation.")
                    
                    # News-based recommendation
                    if hasattr(news, "market_sentiment"):
                        if news.market_sentiment < -0.3:
                            recommendations.append("- **Market Sentiment Alert**: Market sentiment is significantly negative. Consider reviewing your short-term investment strategy.")
                        elif news.market_sentiment > 0.3:
                            recommendations.append("- **Market Sentiment Alert**: Market sentiment is strongly positive. This may present opportunities in growth-oriented assets.")
                
                if portfolio_context:
                    portfolio = portfolio_context.content
                    
                    # Diversification recommendation
                    if hasattr(portfolio, "asset_allocation"):
                        largest_allocation = max(portfolio.asset_allocation.values()) if portfolio.asset_allocation else 0
                        if largest_allocation > 60:
                            for asset_class, percentage in portfolio.asset_allocation.items():
                                if percentage == largest_allocation:
                                    recommendations.append(f"- **Diversification Alert**: Your portfolio has a high concentration ({percentage:.1f}%) in {asset_class}. Consider diversifying.")
                                    break
                
                if market_context:
                    market = market_context.content
                    
                    # Interest rate recommendation
                    if hasattr(market, "economic_indicators") and "Federal Funds Rate" in market.economic_indicators:
                        rate = market.economic_indicators["Federal Funds Rate"]
                        if isinstance(rate, str):
                            try:
                                rate = float(rate.replace("%", ""))
                            except:
                                rate = None
                        
                        if rate is not None and rate > 3:
                            recommendations.append(f"- **Interest Rate Alert**: With the current Federal Funds Rate at {rate}%, consider the impact on debt and fixed-income investments.")
                
                if not recommendations:
                    recommendations.append("- No specific recommendations at this time. Continue to monitor your portfolio regularly.")
                
                summary.extend(recommendations)
                
                # Add disclaimer
                summary.append("\n---")
                summary.append("*This summary is for informational purposes only and does not constitute financial advice. Past performance is not indicative of future results.*")
                
                return "\n".join(summary)
            
            except Exception as e:
                logger.error(f"Error generating daily summary: {e}")
                return f"Error generating daily summary: {str(e)}"
        
        @tool("execute_cross_agent_task")
        def execute_cross_agent_task(task: str, agents: str) -> str:
            """
            Execute a task that requires coordination between multiple agents
            
            Args:
                task: The task to execute
                agents: Comma-separated list of agents to involve (financial_planner, compliance, news, market, portfolio)
            """
            try:
                agent_map = {
                    "financial_planner": self.financial_planner_agent,
                    "compliance": self.compliance_agent,
                    "news": self.news_sentiment_agent,
                    "market": self.market_data_agent,
                    "portfolio": self.portfolio_analyzer_agent
                }
                
                agents_list = [a.strip() for a in agents.split(",")]
                results = []
                
                for agent_name in agents_list:
                    if agent_name in agent_map and agent_map[agent_name] is not None:
                        agent = agent_map[agent_name]
                        logger.info(f"Executing task '{task}' with {agent_name} agent")
                        result = agent.run(task)
                        results.append(f"## {agent_name.title()} Agent Response\n\n{result}")
                    else:
                        results.append(f"## {agent_name.title()} Agent\n\nAgent not available or not found")
                
                return "\n\n".join(results)
            
            except Exception as e:
                logger.error(f"Error executing cross-agent task: {e}")
                return f"Error executing cross-agent task: {str(e)}"
        
        # Add all tools to the list
        tools.append(schedule_daily_updates)
        tools.append(list_scheduled_tasks)
        tools.append(cancel_scheduled_task)
        tools.append(start_scheduler_tool)
        tools.append(stop_scheduler_tool)
        tools.append(generate_daily_summary)
        tools.append(execute_cross_agent_task)
        
        return tools
