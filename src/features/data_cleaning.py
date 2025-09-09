import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataCleaner with optional configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with cleaning parameters
        """
        self.config = config or {}
        self.imputers = {}
        self.scalers = {}
        self.encoders = {}
        
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'mean',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            strategy (str): Strategy to handle missing values ('mean', 'median', 'most_frequent', 'constant')
            columns (list, optional): Columns to process. If None, processes all columns.
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        df = df.copy()
        if columns is None:
            columns = df.columns
            
        for col in columns:
            if df[col].isnull().any():
                if strategy == 'drop':
                    df = df.dropna(subset=[col])
                else:
                    imp = SimpleImputer(strategy=strategy)
                    df[col] = imp.fit_transform(df[[col]]).ravel()
                    self.imputers[col] = imp
                    logger.info(f"Applied {strategy} imputation to column: {col}")
                    
        return df
    
    def normalize_columns(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Normalize numerical columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (list): Columns to normalize
            method (str): Normalization method ('standard', 'minmax')
            
        Returns:
            pd.DataFrame: DataFrame with normalized columns
        """
        df = df.copy()
        
        for col in columns:
            if method == 'standard':
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]]).ravel()
                self.scalers[col] = scaler
                logger.info(f"Applied standard scaling to column: {col}")
            # Add more normalization methods as needed
                
        return df
    
    def encode_categorical(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """
        Encode categorical columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (list): Categorical columns to encode
            method (str): Encoding method ('onehot', 'label')
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical columns
        """
        df = df.copy()
        
        if method == 'onehot':
            for col in columns:
                if col in df.columns:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[col]])
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=[f"{col}_{int(i)}" for i in range(encoded.shape[1])]
                    )
                    df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
                    self.encoders[col] = encoder
                    logger.info(f"Applied one-hot encoding to column: {col}")
        # Add more encoding methods as needed
                
        return df
    
    def clean_data(
        self, 
        df: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Apply all cleaning steps based on configuration.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            config (dict, optional): Configuration dictionary
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        config = config or self.config
        df = df.copy()
        
        # Handle missing values
        if 'missing_values' in config:
            missing_config = config['missing_values']
            df = self.handle_missing_values(
                df, 
                strategy=missing_config.get('strategy', 'mean'),
                columns=missing_config.get('columns')
            )
        
        # Normalize numerical columns
        if 'normalize' in config:
            norm_config = config['normalize']
            df = self.normalize_columns(
                df,
                columns=norm_config.get('columns', []),
                method=norm_config.get('method', 'standard')
            )
        
        # Encode categorical columns
        if 'encode_categorical' in config:
            encode_config = config['encode_categorical']
            df = self.encode_categorical(
                df,
                columns=encode_config.get('columns', []),
                method=encode_config.get('method', 'onehot')
            )
        
        return df

# Example configuration
DEFAULT_CLEANING_CONFIG = {
    'missing_values': {
        'strategy': 'mean',
        'columns': None  # Will process all columns if None
    },
    'normalize': {
        'method': 'standard',
        'columns': ['amount', 'price']  # Example columns to normalize
    },
    'encode_categorical': {
        'method': 'onehot',
        'columns': ['pdv_id', 'sku_id']  # Example categorical columns
    }
}

# Example usage
if __name__ == "__main__":
    # Example usage
    cleaner = DataCleaner(DEFAULT_CLEANING_CONFIG)
    
    # Example: Clean data
    # df_cleaned = cleaner.clean_data(raw_df)
