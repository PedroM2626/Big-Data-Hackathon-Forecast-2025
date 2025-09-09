#!/usr/bin/env python3
"""
Main script for the Big Data Hackathon Forecast 2025 project.
This script demonstrates the complete data processing and modeling pipeline.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import local modules
from src.data.load_data import DataLoader
from src.features.data_cleaning import DataCleaner, DEFAULT_CLEANING_CONFIG
from src.models.train import train_model
from src.models.evaluate import evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def load_and_prepare_data(data_dir: str = "../data") -> pd.DataFrame:
    """
    Load and prepare the sales data.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        pd.DataFrame: Processed DataFrame ready for modeling
    """
    logger.info("Loading and preparing data...")
    
    # Initialize data loader
    loader = DataLoader(data_dir)
    
    try:
        # Load raw data
        # Note: Adjust the filename based on your actual data file
        df = loader.load_csv("sales_2022.csv", raw=True)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Save a copy of the raw data
        loader.save_csv(df, "sales_raw.csv", raw=False)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def clean_and_transform_data(
    df: pd.DataFrame,
    config: dict = None
) -> pd.DataFrame:
    """
    Clean and transform the data.
    
    Args:
        df: Input DataFrame
        config: Configuration for data cleaning
        
    Returns:
        pd.DataFrame: Cleaned and transformed DataFrame
    """
    logger.info("Cleaning and transforming data...")
    
    # Use default config if none provided
    if config is None:
        config = DEFAULT_CLEANING_CONFIG
    
    try:
        # Initialize data cleaner
        cleaner = DataCleaner(config)
        
        # Clean data
        df_cleaned = cleaner.clean_data(df)
        
        # Add any additional transformations here
        # Example: Feature engineering, date parsing, etc.
        
        logger.info(f"Data cleaned. Shape after cleaning: {df_cleaned.shape}")
        return df_cleaned
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise

def train_and_evaluate(
    df: pd.DataFrame,
    model_type: str = 'xgb',
    target_col: str = 'sales_quantity',
    test_size: float = 0.2,
    output_dir: str = "../models"
) -> dict:
    """
    Train and evaluate the forecasting model.
    
    Args:
        df: Input DataFrame with features and target
        model_type: Type of model to train ('xgb' or 'rf')
        target_col: Name of the target column
        test_size: Proportion of data to use for testing
        output_dir: Directory to save the trained model
        
    Returns:
        dict: Training and evaluation results
    """
    logger.info("Training and evaluating model...")
    
    try:
        # Train model
        results = train_model(
            data_path=None,  # We'll pass data directly
            model_type=model_type,
            target_col=target_col,
            test_size=test_size,
            output_dir=output_dir
        )
        
        # Evaluate model
        # Note: This is a simplified example. In practice, you might want to
        # evaluate on a separate test set or use cross-validation.
        
        logger.info("Model training and evaluation completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description='Sales Forecasting Pipeline')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Directory containing the data files')
    parser.add_argument('--model-type', type=str, default='xgb',
                        choices=['xgb', 'rf'],
                        help='Type of model to train')
    parser.add_argument('--target-col', type=str, default='sales_quantity',
                        help='Name of the target column')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--output-dir', type=str, default='../models',
                        help='Directory to save the trained model')
    
    args = parser.parse_args()
    
    try:
        # 1. Load and prepare data
        df = load_and_prepare_data(args.data_dir)
        
        # 2. Clean and transform data
        df_cleaned = clean_and_transform_data(df)
        
        # Save cleaned data
        loader = DataLoader(args.data_dir)
        loader.save_csv(df_cleaned, "sales_cleaned.csv", raw=False)
        
        # 3. Train and evaluate model
        results = train_and_evaluate(
            df_cleaned,
            model_type=args.model_type,
            target_col=args.target_col,
            test_size=args.test_size,
            output_dir=args.output_dir
        )
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
