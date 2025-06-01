"""
Data Loader Utility for FinSage

This module provides functions for loading and parsing portfolio data
from various file formats (CSV, JSON, etc.).
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pandas as pd

from contexts.portfolio_context import PortfolioContextContent, AssetHolding
from contexts.market_context import AssetType
from contexts.user_profile_context import UserProfileContent, FinancialGoal

# Configure logger
logger = logging.getLogger(__name__)


def load_portfolio_from_json(file_path: str) -> PortfolioContextContent:
    """
    Load portfolio data from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing portfolio data
        
    Returns:
        A PortfolioContextContent object populated with the portfolio data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract portfolio metadata
        portfolio_data = {
            "portfolio_id": data.get("portfolio_id", "default_portfolio"),
            "user_id": data.get("user_id", "default_user"),
            "name": data.get("name", "My Portfolio"),
            "description": data.get("description"),
            "cash_value": data.get("cash_value", 0.0),
        }
        
        # Create portfolio context
        portfolio = PortfolioContextContent(**portfolio_data)
        
        # Add holdings
        if "holdings" in data and isinstance(data["holdings"], list):
            for holding_data in data["holdings"]:
                # Parse purchase date
                purchase_date_str = holding_data.get("purchase_date")
                if purchase_date_str:
                    try:
                        purchase_date = datetime.fromisoformat(purchase_date_str)
                    except ValueError:
                        purchase_date = datetime.now()
                else:
                    purchase_date = datetime.now()
                
                # Create holding
                holding = AssetHolding(
                    symbol=holding_data.get("symbol", "").upper(),
                    name=holding_data.get("name", ""),
                    asset_type=holding_data.get("asset_type", AssetType.STOCK),
                    quantity=float(holding_data.get("quantity", 0)),
                    purchase_price=float(holding_data.get("purchase_price", 0)),
                    purchase_date=purchase_date,
                    current_price=float(holding_data.get("current_price", 0)),
                    notes=holding_data.get("notes", ""),
                    tags=holding_data.get("tags", [])
                )
                
                portfolio.add_holding(holding)
        
        # Update portfolio metrics
        portfolio.recalculate_metrics()
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Error loading portfolio data from JSON: {e}")
        # Return an empty portfolio as fallback
        return PortfolioContextContent(
            portfolio_id="error_portfolio",
            user_id="default_user",
            name="Error Portfolio"
        )


def load_portfolio_from_csv(file_path: str) -> PortfolioContextContent:
    """
    Load portfolio data from a CSV file.
    
    Expected CSV format:
    symbol,name,type,quantity,purchase_price,purchase_date
    
    Args:
        file_path: Path to the CSV file containing portfolio data
        
    Returns:
        A PortfolioContextContent object populated with the portfolio data
    """
    try:
        # Read CSV into pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Create portfolio with default values
        portfolio = PortfolioContextContent(
            portfolio_id=Path(file_path).stem,  # Use filename as portfolio ID
            user_id="default_user",
            name=f"Portfolio from {Path(file_path).name}",
            cash_value=0.0
        )
        
        # Process each row as a holding
        for _, row in df.iterrows():
            # Map asset type string to AssetType enum
            asset_type_str = str(row.get("type", "stock")).lower()
            try:
                asset_type = AssetType(asset_type_str)
            except ValueError:
                asset_type = AssetType.STOCK
            
            # Parse purchase date
            purchase_date_str = row.get("purchase_date")
            if purchase_date_str:
                try:
                    purchase_date = pd.to_datetime(purchase_date_str)
                except:
                    purchase_date = datetime.now()
            else:
                purchase_date = datetime.now()
            
            # Create holding
            holding = AssetHolding(
                symbol=str(row.get("symbol", "")).upper(),
                name=str(row.get("name", "")),
                asset_type=asset_type,
                quantity=float(row.get("quantity", 0)),
                purchase_price=float(row.get("purchase_price", 0)),
                purchase_date=purchase_date,
                current_price=float(row.get("current_price", 0)) if "current_price" in row else 0.0
            )
            
            portfolio.add_holding(holding)
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Error loading portfolio data from CSV: {e}")
        # Return an empty portfolio as fallback
        return PortfolioContextContent(
            portfolio_id="error_portfolio",
            user_id="default_user",
            name="Error Portfolio"
        )


def load_user_profile(file_path: str) -> UserProfileContent:
    """
    Load user profile data from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing user profile data
        
    Returns:
        A UserProfileContent object populated with the user profile data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
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
        
        return user_profile
        
    except Exception as e:
        logger.error(f"Error loading user profile data: {e}")
        # Return a default user profile as fallback
        return UserProfileContent(
            user_id="default_user",
            name="Default User"
        )


def save_portfolio_to_json(portfolio: PortfolioContextContent, file_path: str) -> bool:
    """
    Save portfolio data to a JSON file.
    
    Args:
        portfolio: PortfolioContextContent object to save
        file_path: Path to save the JSON file
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Convert portfolio to dictionary
        portfolio_dict = portfolio.dict()
        
        # Convert datetime objects to ISO format strings
        portfolio_dict["created_at"] = portfolio.created_at.isoformat()
        portfolio_dict["last_updated"] = portfolio.last_updated.isoformat()
        
        # Convert holdings
        holdings_list = []
        for symbol, holding in portfolio.holdings.items():
            holding_dict = holding.dict()
            holding_dict["purchase_date"] = holding.purchase_date.isoformat()
            holdings_list.append(holding_dict)
        
        portfolio_dict["holdings"] = holdings_list
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(portfolio_dict, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving portfolio data to JSON: {e}")
        return False
