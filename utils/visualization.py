"""
Visualization utilities for FinSage application.
Creates portfolio visualizations using Plotly and saves them as images.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioVisualizer:
    """Class to generate portfolio visualizations for the FinSage Gradio UI."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the portfolio visualizer.
        
        Args:
            output_dir (str): Directory to save visualization images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Portfolio visualizer initialized with output directory: {output_dir}")
    
    def load_portfolio_data(self, file_path: str) -> Dict[str, Any]:
        """
        Load portfolio data from JSON file.
        
        Args:
            file_path (str): Path to portfolio JSON file
            
        Returns:
            Dict[str, Any]: Portfolio data
        """
        try:
            with open(file_path, 'r') as f:
                portfolio_data = json.load(f)
            logger.info(f"Successfully loaded portfolio data from {file_path}")
            return portfolio_data
        except Exception as e:
            logger.error(f"Error loading portfolio data: {str(e)}")
            return {}
    
    def create_asset_allocation_chart(self, portfolio_data: Dict[str, Any], 
                                     filename: str = "asset_allocation.png") -> str:
        """
        Create asset allocation pie chart.
        
        Args:
            portfolio_data (Dict[str, Any]): Portfolio data
            filename (str): Output filename
            
        Returns:
            str: Path to saved chart image
        """
        try:
            # Extract asset allocation data
            allocation = portfolio_data.get("allocation", {}).get("by_asset_class", {})
            if not allocation:
                logger.warning("No asset allocation data found in portfolio")
                return ""
            
            # Create dataframe for pie chart
            df = pd.DataFrame({
                'Asset Class': list(allocation.keys()),
                'Percentage': list(allocation.values())
            })
            
            # Create pie chart
            fig = px.pie(df, values='Percentage', names='Asset Class', 
                        title='Portfolio Asset Allocation',
                        color_discrete_sequence=px.colors.qualitative.Plotly)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                title_font_size=24,
                font=dict(size=16),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            
            # Save chart as image
            output_path = os.path.join(self.output_dir, filename)
            fig.write_image(output_path, width=800, height=600, scale=2)
            logger.info(f"Asset allocation chart saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating asset allocation chart: {str(e)}")
            return ""
    
    def create_sector_allocation_chart(self, portfolio_data: Dict[str, Any],
                                      filename: str = "sector_allocation.png") -> str:
        """
        Create sector allocation pie chart.
        
        Args:
            portfolio_data (Dict[str, Any]): Portfolio data
            filename (str): Output filename
            
        Returns:
            str: Path to saved chart image
        """
        try:
            # Extract sector allocation data
            allocation = portfolio_data.get("allocation", {}).get("by_sector", {})
            if not allocation:
                logger.warning("No sector allocation data found in portfolio")
                return ""
            
            # Create dataframe for pie chart
            df = pd.DataFrame({
                'Sector': list(allocation.keys()),
                'Percentage': list(allocation.values())
            })
            
            # Create pie chart
            fig = px.pie(df, values='Percentage', names='Sector', 
                        title='Portfolio Sector Allocation',
                        color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                title_font_size=24,
                font=dict(size=16),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            
            # Save chart as image
            output_path = os.path.join(self.output_dir, filename)
            fig.write_image(output_path, width=800, height=600, scale=2)
            logger.info(f"Sector allocation chart saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating sector allocation chart: {str(e)}")
            return ""
    
    def create_performance_chart(self, portfolio_data: Dict[str, Any],
                               filename: str = "performance.png") -> str:
        """
        Create portfolio performance bar chart.
        
        Args:
            portfolio_data (Dict[str, Any]): Portfolio data
            filename (str): Output filename
            
        Returns:
            str: Path to saved chart image
        """
        try:
            # Extract performance data
            performance = portfolio_data.get("performance", {})
            if not performance:
                logger.warning("No performance data found in portfolio")
                return ""
            
            # Create dataframe for bar chart
            periods = ['YTD', '1 Year', '3 Year', '5 Year']
            returns = [
                performance.get('ytd_return_pct', 0),
                performance.get('1y_return_pct', 0),
                performance.get('3y_return_pct', 0),
                performance.get('5y_return_pct', 0)
            ]
            
            # Create bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=periods,
                y=returns,
                text=[f"{r}%" for r in returns],
                textposition='auto',
                marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
            ))
            fig.update_layout(
                title='Portfolio Performance',
                title_font_size=24,
                xaxis=dict(title='Time Period'),
                yaxis=dict(title='Return (%)'),
                font=dict(size=16)
            )
            
            # Save chart as image
            output_path = os.path.join(self.output_dir, filename)
            fig.write_image(output_path, width=800, height=600, scale=2)
            logger.info(f"Performance chart saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating performance chart: {str(e)}")
            return ""
    
    def create_top_holdings_chart(self, portfolio_data: Dict[str, Any],
                                 filename: str = "top_holdings.png",
                                 top_n: int = 10) -> str:
        """
        Create top holdings bar chart.
        
        Args:
            portfolio_data (Dict[str, Any]): Portfolio data
            filename (str): Output filename
            top_n (int): Number of top holdings to display
            
        Returns:
            str: Path to saved chart image
        """
        try:
            # Extract holdings data
            holdings = portfolio_data.get("holdings", [])
            if not holdings:
                logger.warning("No holdings data found in portfolio")
                return ""
            
            # Sort holdings by value and select top N
            sorted_holdings = sorted(holdings, key=lambda x: x.get('current_value', 0), reverse=True)
            top_holdings = sorted_holdings[:top_n]
            
            # Create dataframe for bar chart
            df = pd.DataFrame({
                'Ticker': [h.get('ticker', '') for h in top_holdings],
                'Value': [h.get('current_value', 0) for h in top_holdings],
                'Name': [h.get('name', '') for h in top_holdings]
            })
            
            # Create horizontal bar chart
            fig = px.bar(df, x='Value', y='Ticker', orientation='h',
                        title=f'Top {top_n} Holdings',
                        text='Value',
                        color='Value',
                        color_continuous_scale='Viridis',
                        hover_data=['Name'])
            fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
            fig.update_layout(
                title_font_size=24,
                font=dict(size=16),
                xaxis=dict(title='Value ($)'),
                yaxis=dict(title='')
            )
            
            # Save chart as image
            output_path = os.path.join(self.output_dir, filename)
            fig.write_image(output_path, width=800, height=600, scale=2)
            logger.info(f"Top holdings chart saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating top holdings chart: {str(e)}")
            return ""
    
    def generate_all_visualizations(self, portfolio_file: str) -> Dict[str, str]:
        """
        Generate all portfolio visualizations.
        
        Args:
            portfolio_file (str): Path to portfolio JSON file
            
        Returns:
            Dict[str, str]: Dictionary of visualization names and paths
        """
        try:
            # Load portfolio data
            portfolio_data = self.load_portfolio_data(portfolio_file)
            if not portfolio_data:
                logger.error("Failed to load portfolio data for visualizations")
                return {}
            
            # Generate all charts
            visualizations = {
                "asset_allocation": self.create_asset_allocation_chart(portfolio_data),
                "sector_allocation": self.create_sector_allocation_chart(portfolio_data),
                "performance": self.create_performance_chart(portfolio_data),
                "top_holdings": self.create_top_holdings_chart(portfolio_data)
            }
            
            logger.info(f"Successfully generated {len(visualizations)} visualizations")
            return visualizations
        except Exception as e:
            logger.error(f"Error generating all visualizations: {str(e)}")
            return {}


# Simple test function
def test_visualizer():
    """Test function to generate sample visualizations."""
    visualizer = PortfolioVisualizer()
    portfolio_file = "data/sample_portfolio.json"
    visualizations = visualizer.generate_all_visualizations(portfolio_file)
    print(f"Generated visualizations: {visualizations}")


if __name__ == "__main__":
    test_visualizer()
