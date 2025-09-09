import pandas as pd
import pyarrow.parquet as pq
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
        
        # Read the first 1000 rows using pyarrow
        table = pq.read_table(file_path, columns=None, use_threads=True, memory_map=True)
        df = table.to_pandas().head(1000)
        
        # Save as CSV
        output_file = output_dir / f"sample_{file_path.stem}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved sample to {output_file}")
        
        # Print basic info
        print(f"Shape: {df.shape}")
        print("Columns:")
        for col in df.columns:
            print(f"- {col} ({df[col].dtype})")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
        continue
