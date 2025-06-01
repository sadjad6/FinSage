"""
Market Context Module for FinSage

This module defines the structure for market data including stocks, ETFs,
cryptocurrencies, and overall market indicators.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

class AssetType(str, Enum):
    """Types of financial assets"""
    STOCK = "stock"
    ETF = "etf"
    CRYPTO = "cryptocurrency"
    BOND = "bond"
    FOREX = "forex"
    COMMODITY = "commodity"
    OTHER = "other"

class AssetData(BaseModel):
    """Structure for individual asset data"""
    symbol: str
    name: str
    asset_type: AssetType
    current_price: float
    previous_close: float
    change_amount: float = 0.0
    change_percentage: float = 0.0
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    high_52_week: Optional[float] = None
    low_52_week: Optional[float] = None
    moving_averages: Optional[Dict[str, float]] = None  # e.g., "50day": 150.25
    rsi: Optional[float] = None  # Relative Strength Index
    additional_metrics: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def calculate_changes(self) -> None:
        """Calculate change amount and percentage"""
        if self.previous_close > 0:
            self.change_amount = self.current_price - self.previous_close
            self.change_percentage = (self.change_amount / self.previous_close) * 100

class MarketIndices(BaseModel):
    """Structure for major market indices"""
    sp500: Optional[float] = None
    dow_jones: Optional[float] = None
    nasdaq: Optional[float] = None
    russell_2000: Optional[float] = None
    vix: Optional[float] = None  # Volatility index
    bitcoin_dominance: Optional[float] = None  # For crypto market
    
    sp500_change: Optional[float] = None
    dow_jones_change: Optional[float] = None
    nasdaq_change: Optional[float] = None
    russell_2000_change: Optional[float] = None
    vix_change: Optional[float] = None

class MarketSentiment(BaseModel):
    """Structure for overall market sentiment"""
    overall_sentiment: str = "neutral"  # positive, neutral, negative
    sentiment_score: float = 0.0  # -1.0 to 1.0
    fear_greed_index: Optional[int] = None  # 0-100
    bullish_percentage: Optional[float] = None
    bearish_percentage: Optional[float] = None
    sector_sentiment: Dict[str, float] = Field(default_factory=dict)

class MarketContextContent(BaseModel):
    """Content structure for market context"""
    last_updated: datetime = Field(default_factory=datetime.now)
    market_open: bool = True
    trading_day: datetime = Field(default_factory=lambda: datetime.now().date())
    
    # Market indices
    indices: MarketIndices = Field(default_factory=MarketIndices)
    
    # Overall market sentiment
    sentiment: MarketSentiment = Field(default_factory=MarketSentiment)
    
    # Data for tracked assets
    assets: Dict[str, AssetData] = Field(default_factory=dict)
    
    # Sector performance
    sector_performance: Dict[str, float] = Field(default_factory=dict)
    
    # Economic indicators
    interest_rates: Dict[str, float] = Field(default_factory=dict)
    inflation_rate: Optional[float] = None
    unemployment_rate: Optional[float] = None
    gdp_growth_rate: Optional[float] = None
    
    # Market trends
    trending_assets: List[str] = Field(default_factory=list)
    market_movers: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "gainers": [],
            "losers": [],
            "most_active": []
        }
    )
    
    def add_or_update_asset(self, asset_data: AssetData) -> None:
        """Add or update an asset in the market context"""
        asset_data.calculate_changes()
        self.assets[asset_data.symbol] = asset_data
        self.last_updated = datetime.now()
    
    def get_asset(self, symbol: str) -> Optional[AssetData]:
        """Get asset data by symbol"""
        return self.assets.get(symbol.upper())
    
    def update_indices(self, new_indices: Dict[str, float]) -> None:
        """Update market indices with new values"""
        current_indices = self.indices.dict()
        
        for key, value in new_indices.items():
            if hasattr(self.indices, key):
                old_value = getattr(self.indices, key)
                if old_value is not None and key.endswith('_change'):
                    continue  # Skip updating change fields directly
                
                setattr(self.indices, key, value)
                
                # Calculate change if this is a price field and not a change field
                if not key.endswith('_change') and old_value is not None:
                    change_key = f"{key}_change"
                    if hasattr(self.indices, change_key):
                        change_value = ((value - old_value) / old_value) * 100
                        setattr(self.indices, change_key, change_value)
        
        self.last_updated = datetime.now()
