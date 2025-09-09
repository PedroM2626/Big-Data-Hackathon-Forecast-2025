import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

# Directory containing the data files
data_dir = Path("../Dados")

# List all Parquet files
parquet_files = list(data_dir.glob("*.parquet"))
print(f"Found {len(parquet_files)} Parquet files")

for file_path in parquet_files:
    print(f"\n{'='*80}")
    print(f"File: {file_path.name}")
    print(f"Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # Read the parquet file metadata
        parquet_file = pq.ParquetFile(file_path)
        
        # Print metadata
        print("\nMetadata:")
        print(f"Number of row groups: {parquet_file.num_row_groups}")
        print(f"Number of rows: {parquet_file.metadata.num_rows:,}")
        print(f"Number of columns: {parquet_file.metadata.num_columns}")
        
        # Print schema
        print("\nSchema:")
        print(parquet_file.schema)
        
        # Read the first row group
        first_row_group = parquet_file.read_row_group(0)
        
        # Convert to pandas for easier inspection
        df = first_row_group.to_pandas()
        
        print("\nFirst few rows of data:")
        print(df.head())
        
        # Print column statistics
        print("\nColumn statistics:")
        for col in df.columns:
            print(f"\nColumn: {col}")
            print(f"  Type: {df[col].dtype}")
            print(f"  Non-null count: {df[col].count()}")
            print(f"  Unique values: {df[col].nunique()}")
            if df[col].nunique() < 10:  # Show values if not too many
                print(f"  Values: {df[col].unique().tolist()}")
    
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
