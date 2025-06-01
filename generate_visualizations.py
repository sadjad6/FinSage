"""
Script to generate initial visualizations for the FinSage application.
Run this script to create portfolio visualizations for the Gradio UI.
"""

import os
import logging
from utils.visualization import PortfolioVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Generate all portfolio visualizations for the Gradio UI."""
    logger.info("Starting visualization generation")
    
    # Check if sample portfolio file exists
    portfolio_file = "data/sample_portfolio.json"
    if not os.path.exists(portfolio_file):
        logger.error(f"Portfolio file not found: {portfolio_file}")
        return
    
    # Create portfolio visualizer
    visualizer = PortfolioVisualizer(output_dir="visualizations")
    
    # Generate all visualizations
    visualizations = visualizer.generate_all_visualizations(portfolio_file)
    
    # Log results
    if visualizations:
        logger.info(f"Successfully generated {len(visualizations)} visualizations:")
        for name, path in visualizations.items():
            logger.info(f"- {name}: {path}")
    else:
        logger.error("Failed to generate visualizations")

if __name__ == "__main__":
    main()
