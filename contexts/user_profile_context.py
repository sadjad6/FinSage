"""
User Profile Context Module for FinSage

This module defines the structure and validation for user profile information,
including demographics, financial goals, and risk tolerance.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator

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
    
    @validator('financial_goals', pre=True)
    def validate_goals(cls, goals):
        """Validate that financial goals have valid priorities"""
        if not goals:
            return {}
        
        # Check that priorities are unique
        priorities = [goal.priority for goal in goals.values()]
        if len(priorities) != len(set(priorities)):
            raise ValueError("Financial goal priorities must be unique")
        
        return goals
