import pandas as pd
from pathlib import Path

# Directory containing the data files
data_dir = Path("../Dados")

# List all Parquet files
parquet_files = list(data_dir.glob("*.parquet"))
print(f"Found {len(parquet_files)} Parquet files")

for file_path in parquet_files:
    print(f"\nFile: {file_path.name}")
    print("-" * 50)
    
    try:
        # Read just the schema
        df = pd.read_parquet(file_path, nrows=5)
        print("Columns:")
        for col in df.columns:
            print(f"- {col} ({df[col].dtype})")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
    except Exception as e:
        print(f"Error reading {file_path.name}: {str(e)}")
