import os
import binascii
from pathlib import Path

def check_file_format(file_path):
    """Check the file header to identify its format."""
    print(f"\nChecking file: {file_path.name}")
    print(f"Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        with open(file_path, 'rb') as f:
            # Read the first 16 bytes
            header = f.read(16)
            
            # Print hex and ASCII representation
            print("\nFirst 16 bytes (hex):", ' '.join(f'{b:02x}' for b in header))
            print("ASCII:", ''.join(chr(b) if 32 <= b <= 126 else '.' for b in header))
            
            # Check for common file signatures
            if header.startswith(b'PAR1'):
                print("\nThis appears to be a Parquet file (PAR1 magic number found)")
            elif header.startswith(b'ORC'):
                print("\nThis appears to be an ORC file (ORC magic number found)")
            elif header.startswith(b'PK\x03\x04'):
                print("\nThis appears to be a ZIP file (PK header found)")
            elif header.startswith(b'\x50\x4B\x03\x04'):
                print("\nThis appears to be a ZIP file (PK header found)")
            elif header.startswith(b'\x50\x61\x52\x30'):  # PaR0
                print("\nThis appears to be a Parquet file (PaR0 magic number found)")
            else:
                print("\nFile format not recognized from header")
                
            # Check for snappy compression
            f.seek(-4, 2)
            footer_length = int.from_bytes(f.read(4), byteorder='little')
            f.seek(-(footer_length + 4), 2)
            footer = f.read(footer_length)
            
            if b'snappy' in footer.lower():
                print("File appears to use Snappy compression")
                
    except Exception as e:
        print(f"Error reading file: {str(e)}")

def main():
    # Directory containing the data files
    data_dir = Path("../Dados")
    
    # List all files
    files = list(data_dir.glob("*"))
    print(f"Found {len(files)} files in the Dados directory")
    
    for file_path in files:
        check_file_format(file_path)
        
        # Ask if we should try to read the next file
        if file_path != files[-1]:
            input("\nPress Enter to check the next file...")

if __name__ == "__main__":
    main()
