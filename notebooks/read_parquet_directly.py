import os
import struct
from pathlib import Path

def read_parquet_metadata(file_path):
    """Read and display metadata from a Parquet file."""
    print(f"\nReading metadata from: {file_path.name}")
    print(f"File size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        with open(file_path, 'rb') as f:
            # Read the footer length (last 8 bytes)
            f.seek(-8, 2)
            footer_length = struct.unpack('<i', f.read(4))[0]
            
            # Read the footer
            f.seek(-(footer_length + 8), 2)
            footer = f.read(footer_length)
            
            # Print basic info from the footer
            print(f"Footer length: {footer_length} bytes")
            print("\nFirst 200 bytes of footer (hex):")
            print(' '.join(f'{b:02x}' for b in footer[:200]))
            
            # Try to find column names in the footer
            try:
                # This is a very basic attempt to find strings that might be column names
                # It's not a robust method but might give us some clues
                text = footer.decode('latin-1')
                print("\nPossible column names found in footer:")
                for word in text.split('\x00'):
                    if len(word) > 3 and word.isprintable() and ' ' not in word:
                        print(f"- {word}")
            except:
                print("Could not extract text from footer")
                
    except Exception as e:
        print(f"Error reading file: {str(e)}")

def main():
    # Directory containing the data files
    data_dir = Path("../Dados")
    
    # List all Parquet files
    parquet_files = list(data_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} Parquet files")
    
    for file_path in parquet_files:
        read_parquet_metadata(file_path)
        
        # Ask if we should try to read the next file
        if file_path != parquet_files[-1]:
            input("\nPress Enter to check the next file...")

if __name__ == "__main__":
    main()
