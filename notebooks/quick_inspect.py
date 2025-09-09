#!/usr/bin/env python3
"""
Quick inspection of Parquet files to understand their structure.
"""
import os
import pandas as pd
from pathlib import Path

# Directory containing the data files
data_dir = Path("../Dados")

# List all Parquet files
parquet_files = list(data_dir.glob("*.parquet"))
print(f"Found {len(parquet_files)} Parquet files:")

for i, file_path in enumerate(parquet_files, 1):
    print(f"\n{i}. {file_path.name} ({file_path.stat().st_size / (1024*1024):.2f} MB)")
    
    try:
        # Read just the first row to get column names and data types
        df_sample = pd.read_parquet(file_path, nrows=5)
        
        print("\nColumns and data types:")
        print(df_sample.dtypes)
        
        print("\nFirst 5 rows:")
        print(df_sample)
        
        # Check for potential date columns
        date_cols = [col for col in df_sample.columns if 'data' in col.lower() or 'date' in col.lower()]
        if date_cols:
            print("\nDate columns found:", date_cols)
            
        # Check for potential sales/quantity columns
        sales_cols = [col for col in df_sample.columns 
                     if any(term in col.lower() for term in ['venda', 'quantidade', 'qtd', 'sales', 'quantity'])]
        if sales_cols:
            print("Potential sales/quantity columns:", sales_cols)
            
        # Check for PDV/SKU columns
        pdv_cols = [col for col in df_sample.columns 
                   if any(term in col.lower() for term in ['pdv', 'ponto', 'loja', 'store'])]
        if pdv_cols:
            print("Potential PDV/store columns:", pdv_cols)
            
        sku_cols = [col for col in df_sample.columns 
                   if any(term in col.lower() for term in ['sku', 'produto', 'product', 'item'])]
        if sku_cols:
            print("Potential SKU/product columns:", sku_cols)
            
    except Exception as e:
        print(f"Error reading {file_path.name}: {str(e)}")
    
    print("\n" + "="*80 + "\n")
