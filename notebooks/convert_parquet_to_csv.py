import pandas as pd
from pathlib import Path
import os

# Create output directory
output_dir = Path("../data/raw")
os.makedirs(output_dir, exist_ok=True)

# Directory containing the data files
data_dir = Path("../Dados")

# List all Parquet files
parquet_files = list(data_dir.glob("*.parquet"))
print(f"Found {len(parquet_files)} Parquet files")

for file_path in parquet_files:
    try:
        print(f"\nProcessing: {file_path.name}")
        
        # Read the first 1000 rows to understand the structure
        df = pd.read_parquet(file_path)
        
        # Generate output filename
        output_file = output_dir / f"{file_path.stem}.csv"
        
        # Save as CSV
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} rows to {output_file}")
        
        # Print basic info
        print(f"Columns: {', '.join(df.columns)}")
        print(f"Shape: {df.shape}")
        print("First 5 rows:")
        print(df.head())
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
