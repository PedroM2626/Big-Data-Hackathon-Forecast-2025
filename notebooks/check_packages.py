import sys
import pandas as pd
import numpy as np
import pyarrow
import fastparquet

print("Python version:", sys.version)
print("\nPackage versions:")
print(f"- pandas: {pd.__version__}")
print(f"- numpy: {np.__version__}")
print(f"- pyarrow: {pyarrow.__version__}")
print(f"- fastparquet: {fastparquet.__version__}")

# Check if we can import pyarrow.parquet
try:
    import pyarrow.parquet as pq
    print("\npyarrow.parquet imported successfully")
except Exception as e:
    print(f"\nError importing pyarrow.parquet: {str(e)}")

# Check if we can read a small parquet file
try:
    from pathlib import Path
    data_dir = Path("../Dados")
    parquet_files = list(data_dir.glob("*.parquet"))
    
    if parquet_files:
        print(f"\nFound {len(parquet_files)} parquet files. Trying to read the first one...")
        
        # Try reading with pyarrow
        try:
            table = pq.read_table(parquet_files[0])
            print(f"Successfully read {parquet_files[0].name} with pyarrow")
            print(f"Columns: {table.column_names}")
        except Exception as e:
            print(f"Error reading with pyarrow: {str(e)}")
        
        # Try reading with fastparquet
        try:
            df = fastparquet.ParquetFile(parquet_files[0]).to_pandas()
            print(f"\nSuccessfully read {parquet_files[0].name} with fastparquet")
            print(f"Shape: {df.shape}")
            print("First few rows:")
            print(df.head())
        except Exception as e:
            print(f"Error reading with fastparquet: {str(e)}")
    else:
        print("No parquet files found in the Dados directory")
        
except Exception as e:
    print(f"Error during parquet file check: {str(e)}")
