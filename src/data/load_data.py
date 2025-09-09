import os
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the DataLoader with the data directory.
        
        Args:
            data_dir (str): Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_csv(
        self, 
        filename: str, 
        raw: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load a CSV file from the data directory.
        
        Args:
            filename (str): Name of the CSV file (with or without .csv extension)
            raw (bool): If True, load from raw data directory, else from processed
            **kwargs: Additional arguments to pass to pandas.read_csv()
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        # Ensure .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Determine the directory
        directory = self.raw_data_dir if raw else self.processed_data_dir
        filepath = directory / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        return pd.read_csv(filepath, **kwargs)
    
    def save_csv(
        self, 
        df: pd.DataFrame, 
        filename: str, 
        raw: bool = False,
        **kwargs
    ) -> None:
        """
        Save a DataFrame to a CSV file in the data directory.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Name of the output file (with or without .csv extension)
            raw (bool): If True, save to raw data directory, else to processed
            **kwargs: Additional arguments to pass to DataFrame.to_csv()
        """
        # Ensure .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Determine the directory
        directory = self.raw_data_dir if raw else self.processed_data_dir
        filepath = directory / filename
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving data to {filepath}")
        df.to_csv(filepath, index=False, **kwargs)

# Example usage
if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Example: Load raw data
    # df = loader.load_csv("sales_2022.csv", raw=True)
    
    # Example: Save processed data
    # loader.save_csv(processed_df, "processed_sales.csv", raw=False)
