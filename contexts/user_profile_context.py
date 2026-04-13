"""
User Profile Context Module for FinSage

This module defines the structure and validation for user profile information,
including demographics, financial goals, and risk tolerance.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator, model_validator

class RiskTolerance(str, Enum):
    """Risk tolerance levels for investment strategies"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class TimeHorizon(str, Enum):
    """Investment time horizons"""
    SHORT_TERM = "short_term"  # Less than 3 years
    MEDIUM_TERM = "medium_term"  # 3-10 years
    LONG_TERM = "long_term"  # 10+ years

class FinancialGoal(BaseModel):
    """Structure for financial goals"""
    goal_id: str
    name: str
    description: Optional[str] = None
    target_amount: float
    target_date: datetime
    priority: int = 1  # 1 is highest priority
    current_amount: float = 0
    is_active: bool = True

class UserProfileContent(BaseModel):
    """Content structure for user profile context"""
    user_id: str
    name: str
    age: Optional[int] = None
    annual_income: Optional[float] = None
    tax_bracket: Optional[float] = None
    total_net_worth: Optional[float] = None
    monthly_expenses: Optional[float] = None
    monthly_savings: Optional[float] = None
    
    # Investment preferences
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    preferred_investment_types: List[str] = Field(default_factory=list)
    excluded_sectors: List[str] = Field(default_factory=list)
    esg_focus: bool = False  # Environmental, Social, and Governance focus
    
    # Financial goals
    financial_goals: Dict[str, FinancialGoal] = Field(default_factory=dict)
    
    # Preference settings
    preferred_communication_frequency: str = "daily"
    notification_settings: Dict[str, bool] = Field(
        default_factory=lambda: {
            "market_alerts": True,
            "portfolio_changes": True,
            "goal_progress": True,
            "news_alerts": True
        }
    )
    
    @model_validator(mode='before')
    @classmethod
    def flatten_nested_data(cls, data: Any) -> Any:
        """Flatten nested data from sample JSON structure if present"""
        if not isinstance(data, dict):
            return data
            
        # Extract from personal_info
        personal = data.get("personal_info", {})
        if personal:
            if "name" in personal and "name" not in data:
                data["name"] = personal["name"]
            if "age" in personal and "age" not in data:
                data["age"] = personal["age"]
            if "annual_income" in personal and "annual_income" not in data:
                data["annual_income"] = personal["annual_income"]
            if "tax_bracket" in personal and "tax_bracket" not in data:
                # Handle "24%" string to float conversion if needed
                tb = personal["tax_bracket"]
                if isinstance(tb, str) and "%" in tb:
                    try:
                        data["tax_bracket"] = float(tb.replace("%", "")) / 100
                    except:
                        pass
                else:
                    data["tax_bracket"] = tb

        # Extract from financial_profile
        financial = data.get("financial_profile", {})
        if financial:
            if "monthly_expenses" in financial and "monthly_expenses" not in data:
                data["monthly_expenses"] = financial["monthly_expenses"]
            if "monthly_savings" in financial and "monthly_savings" not in data:
                data["monthly_savings"] = financial["monthly_savings"]
            if "net_worth" in financial and "total_net_worth" not in data:
                data["total_net_worth"] = financial["net_worth"]

        # Extract from risk_profile
        risk = data.get("risk_profile", {})
        if risk:
            if "risk_tolerance" in risk and "risk_tolerance" not in data:
                data["risk_tolerance"] = risk["risk_tolerance"].lower()
            if "investment_horizon" in risk and "time_horizon" not in data:
                horizon = risk["investment_horizon"].lower()
                if "long" in horizon:
                    data["time_horizon"] = "long_term"
                elif "medium" in horizon:
                    data["time_horizon"] = "medium_term"
                else:
                    data["time_horizon"] = "short_term"

        # Extract from preferences
        prefs = data.get("preferences", {})
        if prefs:
            inv_prefs = prefs.get("investment_preferences", {})
            if inv_prefs:
                if "excluded_sectors" in inv_prefs and "excluded_sectors" not in data:
                    data["excluded_sectors"] = inv_prefs["excluded_sectors"]
                if "esg_focus" in inv_prefs and "esg_focus" not in data:
                    data["esg_focus"] = bool(inv_prefs["esg_focus"])

        return data
    
    @validator('financial_goals', pre=True)
    def validate_goals(cls, goals):
        """Validate that financial goals have valid priorities"""
        if not goals:
            return {}
        
        # If it's a list, convert to dict with goal_id as key
        if isinstance(goals, list):
            goals_dict = {}
            for goal in goals:
                if isinstance(goal, dict) and "goal_id" in goal:
                    goals_dict[goal["goal_id"]] = goal
                elif hasattr(goal, "goal_id"):
                    goals_dict[goal.goal_id] = goal
            goals = goals_dict
        
        # Check that priorities are unique
        priorities = [goal.priority if hasattr(goal, "priority") else goal.get("priority") 
                     for goal in goals.values()]
        priorities = [p for p in priorities if p is not None]
        
        if len(priorities) != len(set(priorities)):
            raise ValueError("Financial goal priorities must be unique")
        
        return goals
