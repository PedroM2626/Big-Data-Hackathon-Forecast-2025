import os
import re
from pathlib import Path

def search_patterns_in_file(file_path, chunk_size=8192):
    """Search for common patterns in a binary file."""
    print(f"\nAnalyzing: {file_path.name}")
    print(f"Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Common patterns to look for
    patterns = {
        'date_yyyy_mm_dd': rb'\d{4}[-/]\d{2}[-/]\d{2}',
        'date_dd_mm_yyyy': rb'\d{2}[-/]\d{2}[-/]\d{4}',
        'numeric': rb'\b\d+\.?\d*\b',
        'text': rb'[A-Za-z]{4,}',
        'alphanumeric': rb'[A-Za-z0-9_]{6,}',
        'json': rb'\{\s*"[^"]+"\s*:',
        'csv_header': rb'[A-Za-z_]+(,[A-Za-z_]+)+\r?\n',
    }
    
    try:
        with open(file_path, 'rb') as f:
            # Read first chunk
            chunk = f.read(chunk_size)
            
            # Check for text vs binary
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
            is_binary = bool(chunk.translate(None, text_chars))
            print(f"Appears to be {'binary' if is_binary else 'text'} data")
            
            # Try to detect encoding
            try:
                # Try UTF-8
                chunk.decode('utf-8')
                print("Encoding: Likely UTF-8")
            except UnicodeDecodeError:
                try:
                    # Try Latin-1
                    chunk.decode('latin-1')
                    print("Encoding: Likely Latin-1")
                except:
                    print("Encoding: Could not determine")
            
            # Search for patterns
            print("\nPatterns found:")
            for name, pattern in patterns.items():
                matches = re.findall(pattern, chunk)
                if matches:
                    print(f"- {name}: {len(matches)} matches")
                    if len(matches) < 5:  # Show samples if not too many
                        print(f"  Samples: {matches[:5]}")
            
            # Show sample of the file content
            print("\nSample of file content (first 200 bytes):")
            try:
                print(chunk[:200].decode('utf-8', errors='replace'))
            except:
                try:
                    print(chunk[:200].decode('latin-1', errors='replace'))
                except:
                    print("Could not decode as text")
                    print("Hex dump:", ' '.join(f'{b:02x}' for b in chunk[:100]))
    
    except Exception as e:
        print(f"Error reading file: {str(e)}")

def main():
    # Directory containing the data files
    data_dir = Path("../Dados")
    
    # List all files
    files = list(data_dir.glob("*.parquet"))  # Try with .parquet first
    if not files:
        files = list(data_dir.glob("*"))  # If no .parquet files, try all files
    
    print(f"Found {len(files)} files in the Dados directory")
    
    for file_path in files:
        search_patterns_in_file(file_path)
        
        # Ask if we should try to read the next file
        if file_path != files[-1]:
            input("\nPress Enter to check the next file...")

if __name__ == "__main__":
    main()
