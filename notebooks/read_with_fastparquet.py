import pandas as pd
import fastparquet
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
        
        # Read the file with fastparquet
        pf = fastparquet.ParquetFile(file_path)
        
        # Get the first row group to understand the structure
        df = pf.head(5)
        
        # Print basic info
        print("Columns:")
        for col in df.columns:
            print(f"- {col} ({df[col].dtype})")
        
        print("\nFirst 5 rows:")
        print(df)
        
        # Save as CSV if needed
        save_csv = input(f"\nSave {file_path.name} as CSV? (y/n): ").lower()
        if save_csv == 'y':
            # Read the entire file
            full_df = pf.to_pandas()
            output_file = output_dir / f"{file_path.stem}.csv"
            full_df.to_csv(output_file, index=False)
            print(f"Saved {len(full_df)} rows to {output_file}")
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
