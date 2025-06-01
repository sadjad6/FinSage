"""
Financial Planning Agent for FinSage

This agent is responsible for providing financial recommendations and planning
advice based on user goals, risk profile, and portfolio status.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama

from contexts.user_profile_context import UserProfileContent, RiskTolerance, TimeHorizon, FinancialGoal
from contexts.portfolio_context import PortfolioContextContent
from contexts.market_context import AssetType
from utils.mcp_utils import ContextWrapper, get_registry
from agents.market_data_agent import MarketDataAgent
from agents.portfolio_analyzer_agent import PortfolioAnalyzerAgent

# Configure logger
logger = logging.getLogger(__name__)

class FinancialPlannerAgent:
    """Agent for providing financial planning recommendations"""
    
    def __init__(
        self, 
        market_data_agent: Optional[MarketDataAgent] = None,
        portfolio_analyzer_agent: Optional[PortfolioAnalyzerAgent] = None
    ):
        """Initialize the financial planning agent"""
        self.agent_name = "FinancialPlannerAgent"
        self.model = ChatOllama(model="gemma3:4b")
        
        # Initialize or link to other agents
        self.market_data_agent = market_data_agent if market_data_agent else MarketDataAgent()
        
        if portfolio_analyzer_agent:
            self.portfolio_analyzer_agent = portfolio_analyzer_agent
        else:
            self.portfolio_analyzer_agent = PortfolioAnalyzerAgent(self.market_data_agent)
        
        # Initialize or get latest user profile context
        self.user_profile_context = self._get_or_create_user_profile_context()
        
        # Set up tools for the agent
        self.tools = self._create_tools()
        
        # Set up the agent executor
        self.agent_executor = self._create_agent_executor()
    
    def _get_or_create_user_profile_context(self) -> ContextWrapper:
        """Get existing user profile context or create a new one"""
        registry = get_registry()
        context = registry.get_latest_context("user_profile_context")
        
        if not context:
            # Create a new user profile context
            context_content = UserProfileContent(
                user_id="default_user",
                name="Default User"
            )
            
            context = ContextWrapper.create(
                context_type="user_profile_context",
                creator_agent=self.agent_name,
                content_model=UserProfileContent,
                content_data=context_content.dict()
            )
            
            # Register the new context
            registry.register_context(context)
        
        return context
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools and prompt"""
        # Set up the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional financial planning assistant that helps users with financial goals, 
            retirement planning, and investment advice. You have access to the user's profile information, 
            current market data, and portfolio status.
            
            Your goal is to provide personalized financial advice based on the user's risk tolerance, 
            time horizon, and financial goals.
            
            Always explain your recommendations and provide context for your advice.
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
        """Run the agent with a user query"""
        try:
            response = self.agent_executor.invoke({"input": query})
            return response["output"]
        except Exception as e:
            logger.error(f"Error running financial planner agent: {e}")
            return f"Error generating financial planning advice: {str(e)}"
    
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for the agent to use"""
        tools = []
        
        @tool("get_user_profile")
        def get_user_profile() -> str:
            """
            Get the current user profile information including risk tolerance,
            financial goals, and preferences.
            """
            user_profile = self.user_profile_context.content
            
            # Format user profile report
            report = ["## User Profile", ""]
            
            # Basic information
            report.append("### Basic Information")
            report.append(f"- Name: {user_profile.name}")
            if user_profile.age:
                report.append(f"- Age: {user_profile.age}")
            if user_profile.annual_income:
                report.append(f"- Annual Income: ${user_profile.annual_income:,.2f}")
            if user_profile.total_net_worth:
                report.append(f"- Net Worth: ${user_profile.total_net_worth:,.2f}")
            
            # Investment preferences
            report.append("\n### Investment Preferences")
            report.append(f"- Risk Tolerance: {user_profile.risk_tolerance.value.title()}")
            report.append(f"- Time Horizon: {user_profile.time_horizon.value.replace('_', ' ').title()}")
            
            if user_profile.preferred_investment_types:
                report.append(f"- Preferred Investment Types: {', '.join(user_profile.preferred_investment_types)}")
            
            if user_profile.excluded_sectors:
                report.append(f"- Excluded Sectors: {', '.join(user_profile.excluded_sectors)}")
            
            if user_profile.esg_focus:
                report.append("- ESG Focus: Yes")
            
            # Financial goals
            if user_profile.financial_goals:
                report.append("\n### Financial Goals")
                
                sorted_goals = sorted(
                    user_profile.financial_goals.values(),
                    key=lambda g: g.priority
                )
                
                for goal in sorted_goals:
                    if goal.is_active:
                        progress = (goal.current_amount / goal.target_amount) * 100 if goal.target_amount > 0 else 0
                        target_date_str = goal.target_date.strftime("%Y-%m-%d")
                        
                        report.append(f"- **{goal.name}** (Priority: {goal.priority})")
                        if goal.description:
                            report.append(f"  - Description: {goal.description}")
                        report.append(f"  - Target: ${goal.target_amount:,.2f} by {target_date_str}")
                        report.append(f"  - Current Progress: ${goal.current_amount:,.2f} ({progress:.1f}%)")
            
            return "\n".join(report)
        
        @tool("set_user_profile")
        def set_user_profile(profile_data: str) -> str:
            """
            Set user profile data from a JSON string representation.
            This is mainly used to load user profile data from files or user input.
            
            Args:
                profile_data: JSON string with user profile data
            """
            try:
                # Parse JSON data
                import json
                data = json.loads(profile_data)
                
                # Extract financial goals if present
                financial_goals = {}
                if "financial_goals" in data and isinstance(data["financial_goals"], list):
                    for goal_data in data["financial_goals"]:
                        goal_id = goal_data.get("goal_id", str(len(financial_goals) + 1))
                        
                        # Parse target date
                        target_date_str = goal_data.get("target_date")
                        if target_date_str:
                            try:
                                target_date = datetime.fromisoformat(target_date_str)
                            except ValueError:
                                target_date = datetime.now()
                        else:
                            target_date = datetime.now()
                        
                        # Create financial goal
                        goal = FinancialGoal(
                            goal_id=goal_id,
                            name=goal_data.get("name", "Unnamed Goal"),
                            description=goal_data.get("description"),
                            target_amount=float(goal_data.get("target_amount", 0)),
                            target_date=target_date,
                            priority=int(goal_data.get("priority", 1)),
                            current_amount=float(goal_data.get("current_amount", 0)),
                            is_active=goal_data.get("is_active", True)
                        )
                        
                        financial_goals[goal_id] = goal
                
                # Remove financial_goals key from data to avoid duplication
                if "financial_goals" in data:
                    data_copy = data.copy()
                    del data_copy["financial_goals"]
                else:
                    data_copy = data
                
                # Create user profile
                user_profile = UserProfileContent(**data_copy)
                
                # Add financial goals
                user_profile.financial_goals = financial_goals
                
                # Create a new context wrapper
                self.user_profile_context = ContextWrapper.create(
                    context_type="user_profile_context",
                    creator_agent=self.agent_name,
                    content_model=UserProfileContent,
                    content_data=user_profile.dict()
                )
                
                # Register the new context
                registry = get_registry()
                registry.register_context(self.user_profile_context)
                
                return f"User profile updated for {user_profile.name}"
                
            except Exception as e:
                logger.error(f"Error setting user profile data: {e}")
                return f"Error setting user profile data: {str(e)}"
        
        @tool("recommend_asset_allocation")
        def recommend_asset_allocation() -> str:
            """
            Recommend an asset allocation strategy based on the user's risk tolerance,
            time horizon, and financial goals.
            """
            user_profile = self.user_profile_context.content
            
            # Define allocation strategies based on risk tolerance and time horizon
            allocation_strategies = {
                # Conservative allocations
                (RiskTolerance.CONSERVATIVE, TimeHorizon.SHORT_TERM): {
                    AssetType.STOCK: 20,
                    AssetType.BOND: 60,
                    AssetType.CASH: 20,
                    AssetType.CRYPTO: 0
                },
                (RiskTolerance.CONSERVATIVE, TimeHorizon.MEDIUM_TERM): {
                    AssetType.STOCK: 30,
                    AssetType.BOND: 60,
                    AssetType.CASH: 10,
                    AssetType.CRYPTO: 0
                },
                (RiskTolerance.CONSERVATIVE, TimeHorizon.LONG_TERM): {
                    AssetType.STOCK: 40,
                    AssetType.BOND: 50,
                    AssetType.CASH: 10,
                    AssetType.CRYPTO: 0
                },
                
                # Moderate allocations
                (RiskTolerance.MODERATE, TimeHorizon.SHORT_TERM): {
                    AssetType.STOCK: 40,
                    AssetType.BOND: 40,
                    AssetType.CASH: 15,
                    AssetType.CRYPTO: 5
                },
                (RiskTolerance.MODERATE, TimeHorizon.MEDIUM_TERM): {
                    AssetType.STOCK: 60,
                    AssetType.BOND: 30,
                    AssetType.CASH: 5,
                    AssetType.CRYPTO: 5
                },
                (RiskTolerance.MODERATE, TimeHorizon.LONG_TERM): {
                    AssetType.STOCK: 70,
                    AssetType.BOND: 20,
                    AssetType.CASH: 5,
                    AssetType.CRYPTO: 5
                },
                
                # Aggressive allocations
                (RiskTolerance.AGGRESSIVE, TimeHorizon.SHORT_TERM): {
                    AssetType.STOCK: 70,
                    AssetType.BOND: 15,
                    AssetType.CASH: 5,
                    AssetType.CRYPTO: 10
                },
                (RiskTolerance.AGGRESSIVE, TimeHorizon.MEDIUM_TERM): {
                    AssetType.STOCK: 75,
                    AssetType.BOND: 10,
                    AssetType.CASH: 5,
                    AssetType.CRYPTO: 10
                },
                (RiskTolerance.AGGRESSIVE, TimeHorizon.LONG_TERM): {
                    AssetType.STOCK: 80,
                    AssetType.BOND: 5,
                    AssetType.CASH: 5,
                    AssetType.CRYPTO: 10
                }
            }
            
            # Get the recommended allocation based on user profile
            risk_tolerance = user_profile.risk_tolerance
            time_horizon = user_profile.time_horizon
            
            # Get the allocation strategy
            allocation = allocation_strategies.get(
                (risk_tolerance, time_horizon),
                allocation_strategies[(RiskTolerance.MODERATE, TimeHorizon.MEDIUM_TERM)]  # Default
            )
            
            # Format asset allocation report
            report = ["## Recommended Asset Allocation", ""]
            
            # User profile summary
            report.append("### Based on Your Profile")
            report.append(f"- Risk Tolerance: {risk_tolerance.value.title()}")
            report.append(f"- Time Horizon: {time_horizon.value.replace('_', ' ').title()}")
            
            # Allocation recommendation
            report.append("\n### Recommended Allocation")
            for asset_type, percentage in allocation.items():
                report.append(f"- {asset_type.value.title()}: {percentage}%")
            
            # Add explanation
            report.append("\n### Explanation")
            
            if risk_tolerance == RiskTolerance.CONSERVATIVE:
                report.append("This conservative allocation prioritizes capital preservation and stable income over growth. ")
                report.append("The higher bond allocation provides more stability and income, while the limited stock exposure still offers some growth potential.")
            
            elif risk_tolerance == RiskTolerance.MODERATE:
                report.append("This balanced allocation seeks to provide a mix of growth and income. ")
                report.append("The stock component aims for long-term appreciation, while bonds and cash provide stability and income.")
            
            elif risk_tolerance == RiskTolerance.AGGRESSIVE:
                report.append("This growth-oriented allocation prioritizes capital appreciation over income and stability. ")
                report.append("The high stock allocation provides strong growth potential, while the limited bond and cash exposure offers some downside protection.")
            
            # Add time horizon explanation
            report.append("\nTime Horizon Impact:")
            if time_horizon == TimeHorizon.SHORT_TERM:
                report.append("- Your shorter time horizon requires more liquidity and less volatility")
                report.append("- This allocation has higher cash and bond allocations to protect against market volatility")
            
            elif time_horizon == TimeHorizon.MEDIUM_TERM:
                report.append("- Your medium-term time horizon allows for a more balanced approach")
                report.append("- This allocation provides a mix of growth potential and reasonable stability")
            
            elif time_horizon == TimeHorizon.LONG_TERM:
                report.append("- Your long-term time horizon allows for higher risk tolerance")
                report.append("- This allocation maximizes growth potential while still providing some diversification")
            
            # Implementation suggestions
            report.append("\n### Implementation Suggestions")
            report.append("- **Stocks**: Consider a mix of domestic and international stocks across different sectors and market caps")
            report.append("- **Bonds**: Include a combination of government, municipal, and corporate bonds with appropriate durations")
            report.append("- **Cash**: Maintain an emergency fund in high-yield savings accounts or money market funds")
            
            if allocation[AssetType.CRYPTO] > 0:
                report.append("- **Cryptocurrency**: Limit to established cryptocurrencies and consider dollar-cost averaging")
            
            return "\n".join(report)
        
        @tool("create_financial_goal")
        def create_financial_goal(name: str, target_amount: float, target_date: str, priority: int = 1, description: str = None) -> str:
            """
            Create a new financial goal for the user.
            
            Args:
                name: Name of the financial goal
                target_amount: Target amount to achieve
                target_date: Target date in YYYY-MM-DD format
                priority: Priority level (1 is highest)
                description: Optional description of the goal
            """
            try:
                user_profile = self.user_profile_context.content
                
                # Parse target date
                try:
                    target_date_obj = datetime.fromisoformat(target_date)
                except ValueError:
                    return f"Invalid target date format: {target_date}. Use YYYY-MM-DD format."
                
                # Create a unique goal ID
                goal_id = str(len(user_profile.financial_goals) + 1)
                
                # Create the financial goal
                goal = FinancialGoal(
                    goal_id=goal_id,
                    name=name,
                    description=description,
                    target_amount=float(target_amount),
                    target_date=target_date_obj,
                    priority=int(priority),
                    current_amount=0.0,
                    is_active=True
                )
                
                # Add the goal to the user profile
                user_profile.financial_goals[goal_id] = goal
                
                # Update the context in the registry
                self.user_profile_context.update(
                    updated_by=self.agent_name,
                    content_updates={"financial_goals": user_profile.financial_goals}
                )
                
                registry = get_registry()
                registry.register_context(self.user_profile_context)
                
                return f"Financial goal '{name}' created with target amount ${target_amount:,.2f} by {target_date}"
                
            except Exception as e:
                logger.error(f"Error creating financial goal: {e}")
                return f"Error creating financial goal: {str(e)}"
        
        @tool("generate_retirement_plan")
        def generate_retirement_plan() -> str:
            """
            Generate a retirement plan based on the user's age, income, current savings,
            and retirement goals.
            """
            user_profile = self.user_profile_context.content
            
            # Determine if we have enough information
            if not user_profile.age or not user_profile.annual_income:
                return "Insufficient information to generate a retirement plan. Please update your profile with age and annual income."
            
            # Extract relevant data from user profile
            age = user_profile.age
            annual_income = user_profile.annual_income
            current_retirement_savings = 0
            retirement_age = 65
            retirement_income_goal_percent = 80  # Default is 80% of current income
            
            # Check for a retirement goal in the user's financial goals
            for goal in user_profile.financial_goals.values():
                if "retirement" in goal.name.lower() and goal.is_active:
                    current_retirement_savings = goal.current_amount
                    
                    # Extract retirement age from description if available
                    if goal.description and "age" in goal.description.lower():
                        try:
                            # Try to find a number in the description
                            import re
                            age_match = re.search(r'age\s*(\d+)', goal.description.lower())
                            if age_match:
                                retirement_age = int(age_match.group(1))
                        except:
                            pass
            
            # Calculate retirement needs
            years_to_retirement = retirement_age - age
            if years_to_retirement <= 0:
                return "You are already at or past retirement age. Please consult with a financial advisor for personalized advice."
            
            retirement_years = 95 - retirement_age  # Plan until age 95
            annual_retirement_income_needed = annual_income * (retirement_income_goal_percent / 100)
            total_retirement_savings_needed = annual_retirement_income_needed * retirement_years
            
            # Adjust for social security and other income (simplified)
            social_security_annual = 20000  # Rough estimate
            additional_savings_needed = total_retirement_savings_needed - (social_security_annual * retirement_years)
            
            # Calculate monthly savings needed
            # Using a simplified formula assuming 6% annual returns
            annual_return_rate = 0.06
            monthly_return_rate = annual_return_rate / 12
            num_months = years_to_retirement * 12
            
            # Future value of current savings
            future_value_current_savings = current_retirement_savings * ((1 + annual_return_rate) ** years_to_retirement)
            remaining_savings_needed = additional_savings_needed - future_value_current_savings
            
            # If already saved enough
            if remaining_savings_needed <= 0:
                monthly_savings_needed = 0
            else:
                # PMT formula: https://en.wikipedia.org/wiki/Time_value_of_money
                monthly_savings_needed = (remaining_savings_needed * monthly_return_rate) / ((1 + monthly_return_rate) ** num_months - 1)
            
            # Generate the report
            report = ["## Retirement Planning Analysis", ""]
            
            # Current situation
            report.append("### Your Current Situation")
            report.append(f"- Current Age: {age}")
            report.append(f"- Annual Income: ${annual_income:,.2f}")
            report.append(f"- Current Retirement Savings: ${current_retirement_savings:,.2f}")
            
            # Retirement goals
            report.append("\n### Retirement Goals")
            report.append(f"- Target Retirement Age: {retirement_age}")
            report.append(f"- Years Until Retirement: {years_to_retirement}")
            report.append(f"- Retirement Income Goal: ${annual_retirement_income_needed:,.2f}/year ({retirement_income_goal_percent}% of current income)")
            
            # Financial projections
            report.append("\n### Financial Projections")
            report.append(f"- Estimated Retirement Duration: {retirement_years} years")
            report.append(f"- Total Retirement Savings Needed: ${total_retirement_savings_needed:,.2f}")
            report.append(f"- Estimated Social Security Income: ${social_security_annual:,.2f}/year")
            report.append(f"- Additional Savings Needed: ${additional_savings_needed:,.2f}")
            report.append(f"- Projected Value of Current Savings at Retirement: ${future_value_current_savings:,.2f}")
            
            # Savings plan
            report.append("\n### Recommended Savings Plan")
            if monthly_savings_needed <= 0:
                report.append("**Congratulations!** Your current savings are on track to meet your retirement goals.")
                report.append("Consider increasing your retirement goal for additional security or an improved lifestyle.")
            else:
                report.append(f"- **Recommended Monthly Savings**: ${monthly_savings_needed:,.2f}")
                report.append(f"- Percentage of Monthly Income: {(monthly_savings_needed * 12 / annual_income) * 100:.1f}%")
            
            # Additional advice
            report.append("\n### Additional Recommendations")
            report.append("1. **Maximize tax-advantaged accounts** like 401(k) and IRA contributions")
            report.append("2. **Diversify investments** based on your risk tolerance and time horizon")
            report.append("3. **Review and adjust your plan annually** as your income and goals change")
            report.append("4. **Consider inflation** in your long-term planning")
            report.append("5. **Consult with a professional financial advisor** for personalized advice")
            
            return "\n".join(report)
        
        @tool("provide_financial_advice")
        def provide_financial_advice(query: str) -> str:
            """
            Provide personalized financial advice based on the user's query, profile,
            and current portfolio status.
            
            Args:
                query: The user's financial question or concern
            """
            user_profile = self.user_profile_context.content
            
            # Get portfolio context if available
            registry = get_registry()
            portfolio_context = registry.get_latest_context("portfolio_context")
            portfolio_data = None
            if portfolio_context:
                portfolio_data = portfolio_context.content
            
            # Get market context if available
            market_context = registry.get_latest_context("market_context")
            market_data = None
            if market_context:
                market_data = market_context.content
            
            # Prepare advice based on query and available data
            advice = [f"## Financial Advice: {query}", ""]
            
            # Add user profile context to the advice
            advice.append("Based on your profile:")
            advice.append(f"- Risk Tolerance: {user_profile.risk_tolerance.value.title()}")
            advice.append(f"- Time Horizon: {user_profile.time_horizon.value.replace('_', ' ').title()}")
            
            if user_profile.age:
                advice.append(f"- Age: {user_profile.age}")
            
            if user_profile.annual_income:
                advice.append(f"- Annual Income: ${user_profile.annual_income:,.2f}")
            
            # Generate different advice based on the query type
            # Categorize the query and provide relevant advice
            lower_query = query.lower()
            
            # For retirement-related queries
            if any(term in lower_query for term in ["retire", "retirement", "401k", "ira"]):
                advice.append("\n### Retirement Planning Advice")
                if user_profile.age:
                    if user_profile.age < 30:
                        advice.append("- Start early: You have the advantage of time and compound growth")
                        advice.append("- Consider allocating more to stocks for long-term growth")
                        advice.append("- Maximize tax-advantaged accounts like Roth IRA and 401(k)")
                    elif user_profile.age < 50:
                        advice.append("- Focus on increasing retirement contributions as your income grows")
                        advice.append("- Review asset allocation to ensure it aligns with your timeline")
                        advice.append("- Consider tax diversification strategies across different account types")
                    else:
                        advice.append("- Make catch-up contributions to retirement accounts if eligible")
                        advice.append("- Begin to shift towards more conservative investments")
                        advice.append("- Consider consulting with a financial advisor about retirement income strategies")
                else:
                    advice.append("- Consider setting up automatic contributions to retirement accounts")
                    advice.append("- Aim to save at least 15% of your income for retirement")
                    advice.append("- Take full advantage of any employer matching in retirement plans")
            
            # For debt-related queries
            elif any(term in lower_query for term in ["debt", "loan", "mortgage", "credit"]):
                advice.append("\n### Debt Management Advice")
                advice.append("- Prioritize paying off high-interest debt first (usually credit cards)")
                advice.append("- Consider refinancing options for mortgages or student loans if interest rates are favorable")
                advice.append("- Maintain an emergency fund even while paying down debt")
                advice.append("- Be cautious about taking on new debt, especially for depreciating assets")
            
            # For investment-related queries
            elif any(term in lower_query for term in ["invest", "stock", "bond", "portfolio", "etf", "mutual fund"]):
                advice.append("\n### Investment Advice")
                
                if portfolio_data:
                    # Add portfolio-specific advice if available
                    advice.append(f"Based on your current portfolio allocation:")
                    
                    # Check diversification
                    if portfolio_data.metrics and hasattr(portfolio_data.metrics, 'diversification_score'):
                        div_score = portfolio_data.metrics.diversification_score
                        if div_score < 0.4:
                            advice.append("- Your portfolio has low diversification. Consider adding more assets across different sectors and asset classes.")
                        elif div_score < 0.7:
                            advice.append("- Your portfolio has moderate diversification. Continue to explore additional asset classes for better risk management.")
                        else:
                            advice.append("- Your portfolio is well-diversified. Maintain this balance while rebalancing periodically.")
                    
                    # Check risk level vs user profile
                    if portfolio_data.metrics and hasattr(portfolio_data.metrics, 'risk_level'):
                        risk_level = portfolio_data.metrics.risk_level
                        if user_profile.risk_tolerance == RiskTolerance.CONSERVATIVE and risk_level > 0.5:
                            advice.append("- Your current portfolio risk level is higher than your conservative risk tolerance. Consider reducing exposure to volatile assets.")
                        elif user_profile.risk_tolerance == RiskTolerance.AGGRESSIVE and risk_level < 0.5:
                            advice.append("- Your current portfolio risk level is lower than your aggressive risk tolerance. You may be missing growth opportunities.")
                else:
                    # General investment advice
                    advice.append("- Maintain a diversified portfolio across different asset classes")
                    advice.append("- Consider low-cost index funds for core portfolio holdings")
                    advice.append("- Rebalance your portfolio periodically to maintain your target allocation")
                    advice.append("- Stay focused on long-term goals rather than reacting to short-term market movements")
            
            # For savings-related queries
            elif any(term in lower_query for term in ["save", "saving", "emergency fund", "budget"]):
                advice.append("\n### Savings Advice")
                advice.append("- Aim to maintain an emergency fund covering 3-6 months of expenses")
                advice.append("- Use high-yield savings accounts for short-term goals")
                advice.append("- Automate savings to ensure consistency")
                advice.append("- Consider tax-advantaged savings options like HSAs or 529 plans for specific goals")
            
            # For tax-related queries
            elif any(term in lower_query for term in ["tax", "taxes", "ira", "deduction"]):
                advice.append("\n### Tax Planning Advice")
                advice.append("- Maximize contributions to tax-advantaged accounts")
                advice.append("- Consider tax-loss harvesting for taxable investment accounts")
                advice.append("- Keep records of tax-deductible expenses throughout the year")
                advice.append("- Consult with a tax professional for personalized tax strategies")
            
            # Default advice for other queries
            else:
                advice.append("\n### General Financial Advice")
                advice.append("- Follow a structured budget to manage income and expenses")
                advice.append("- Prioritize building an emergency fund, then paying down debt, then investing")
                advice.append("- Review your insurance coverage to ensure adequate protection")
                advice.append("- Update your financial plan annually or when major life events occur")
            
            # Market conditions disclaimer if market data is available
            if market_data:
                advice.append("\n### Current Market Considerations")
                
                # Check market sentiment
                if hasattr(market_data, 'market_sentiment') and market_data.market_sentiment:
                    sentiment = market_data.market_sentiment
                    if sentiment < -0.3:
                        advice.append("- Current market sentiment is negative. Focus on quality investments and avoid emotional decisions.")
                    elif sentiment > 0.3:
                        advice.append("- Current market sentiment is positive, but remain disciplined and avoid FOMO-driven investments.")
                    else:
                        advice.append("- Current market sentiment is neutral. This is a good time for methodical, planned investing.")
            
            # Disclaimer
            advice.append("\n*This advice is general in nature and may not be suitable for your specific situation. Consider consulting with a professional financial advisor for personalized guidance.*")
            
            return "\n".join(advice)
        
        # Add all tools
        tools.append(get_user_profile)
        tools.append(set_user_profile)
        tools.append(recommend_asset_allocation)
        tools.append(create_financial_goal)
        tools.append(generate_retirement_plan)
        tools.append(provide_financial_advice)
        
        return tools
