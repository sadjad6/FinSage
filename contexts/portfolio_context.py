"""
Portfolio Context Module for FinSage

This module defines the structure for user portfolio data including assets,
historical performance, and allocation metrics.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator

from .market_context import AssetType

class AssetHolding(BaseModel):
    """Structure for an asset holding in a portfolio"""
    symbol: str
    name: str
    asset_type: AssetType
    quantity: float
    purchase_price: float
    purchase_date: datetime
    current_price: float = 0.0
    current_value: float = 0.0
    cost_basis: float = 0.0
    gain_loss_amount: float = 0.0
    gain_loss_percentage: float = 0.0
    weight: float = 0.0  # Percentage of portfolio
    dividends_received: float = 0.0
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    def update_metrics(self, current_price: float, total_portfolio_value: float) -> None:
        """Update the metrics for this holding based on current price"""
        self.current_price = current_price
        self.current_value = self.quantity * current_price
        self.cost_basis = self.quantity * self.purchase_price
        
        # Calculate gain/loss
        self.gain_loss_amount = self.current_value - self.cost_basis
        if self.cost_basis > 0:
            self.gain_loss_percentage = (self.gain_loss_amount / self.cost_basis) * 100
        
        # Calculate weight in portfolio
        if total_portfolio_value > 0:
            self.weight = (self.current_value / total_portfolio_value) * 100

class HistoricalValue(BaseModel):
    """Structure for historical portfolio value data points"""
    date: datetime
    total_value: float
    cash_value: float
    invested_value: float
    deposits: float = 0.0
    withdrawals: float = 0.0
    daily_change_amount: float = 0.0
    daily_change_percentage: float = 0.0

class PortfolioMetrics(BaseModel):
    """Structure for portfolio performance metrics"""
    total_return_amount: float = 0.0
    total_return_percentage: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    volatility: Optional[float] = None  # Standard deviation of returns
    max_drawdown: Optional[float] = None  # Maximum observed loss
    dividend_yield: Optional[float] = None
    
    # Asset allocation
    allocation_by_asset_type: Dict[AssetType, float] = Field(default_factory=dict)
    allocation_by_sector: Dict[str, float] = Field(default_factory=dict)
    allocation_by_region: Dict[str, float] = Field(default_factory=dict)
    
    # Risk metrics
    risk_concentration: Dict[str, float] = Field(default_factory=dict)
    diversification_score: Optional[float] = None  # 0-100, higher is better

class PortfolioContextContent(BaseModel):
    """Content structure for portfolio context"""
    portfolio_id: str
    user_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    # Current portfolio state
    total_value: float = 0.0
    cash_value: float = 0.0
    invested_value: float = 0.0
    
    # Holdings
    holdings: Dict[str, AssetHolding] = Field(default_factory=dict)
    
    # Performance metrics
    metrics: PortfolioMetrics = Field(default_factory=PortfolioMetrics)
    
    # Historical data
    historical_values: List[HistoricalValue] = Field(default_factory=list)
    
    # Portfolio goals
    target_allocation: Dict[AssetType, float] = Field(default_factory=dict)
    rebalance_threshold: float = 5.0  # Percentage deviation to trigger rebalance
    
    # Transaction history
    transactions: List[Dict[str, Any]] = Field(default_factory=list)
    
    @validator('total_value')
    def validate_total_value(cls, value):
        """Ensure total value is non-negative"""
        if value < 0:
            raise ValueError("Total portfolio value cannot be negative")
        return value
    
    def add_holding(self, holding: AssetHolding) -> None:
        """Add a new holding to the portfolio"""
        self.holdings[holding.symbol] = holding
        self.recalculate_metrics()
    
    def update_holding(self, symbol: str, updates: Dict[str, Any]) -> None:
        """Update an existing holding"""
        if symbol in self.holdings:
            for key, value in updates.items():
                if hasattr(self.holdings[symbol], key):
                    setattr(self.holdings[symbol], key, value)
            self.recalculate_metrics()
    
    def remove_holding(self, symbol: str) -> None:
        """Remove a holding from the portfolio"""
        if symbol in self.holdings:
            del self.holdings[symbol]
            self.recalculate_metrics()
    
    def recalculate_metrics(self) -> None:
        """Recalculate portfolio metrics and total values"""
        # Calculate total invested value
        self.invested_value = sum(holding.current_value for holding in self.holdings.values())
        self.total_value = self.cash_value + self.invested_value
        
        # Update holding weights
        for holding in self.holdings.values():
            holding.update_metrics(holding.current_value, self.total_value)
        
        # Update asset allocation metrics
        allocation_by_type = {}
        for holding in self.holdings.values():
            asset_type = holding.asset_type
            if asset_type not in allocation_by_type:
                allocation_by_type[asset_type] = 0
            allocation_by_type[asset_type] += holding.weight
        
        self.metrics.allocation_by_asset_type = allocation_by_type
        
        # Update last_updated timestamp
        self.last_updated = datetime.now()
    
    def add_historical_value(self, value_entry: HistoricalValue) -> None:
        """Add a historical value data point"""
        self.historical_values.append(value_entry)
        
        # Keep historical values sorted by date
        self.historical_values.sort(key=lambda x: x.date)
    
    def calculate_returns(self) -> None:
        """Calculate portfolio return metrics"""
        if not self.historical_values or len(self.historical_values) < 2:
            return
        
        # Get initial and current values
        initial_value = self.historical_values[0].total_value
        current_value = self.total_value
        
        # Calculate deposits and withdrawals
        total_deposits = sum(hv.deposits for hv in self.historical_values)
        total_withdrawals = sum(hv.withdrawals for hv in self.historical_values)
        
        # Calculate total return
        adjusted_initial = initial_value + total_deposits - total_withdrawals
        if adjusted_initial > 0:
            self.metrics.total_return_amount = current_value - adjusted_initial
            self.metrics.total_return_percentage = (self.metrics.total_return_amount / adjusted_initial) * 100
