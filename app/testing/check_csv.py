"""
Quick diagnostic script to check CSV file format.
Run this to identify CSV parsing issues before running the main app.
"""

import pandas as pd
import os

def check_csv_file():
    """Check the CSV file and report any issues."""
    print("="*60)
    print("CSV File Diagnostic Check")
    print("="*60)
    
    csv_path = "resource/AXA_QNA.csv"
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    print(f"✓ File found: {csv_path}")
    
    # Get file size
    file_size = os.path.getsize(csv_path)
    print(f"✓ File size: {file_size:,} bytes")
    
    # Read first few lines to check format
    print("\n" + "-"*60)
    print("First 3 lines of file:")
    print("-"*60)
    with open(csv_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 3:
                print(f"Line {i+1}: {line[:100]}..." if len(line) > 100 else f"Line {i+1}: {line.strip()}")
            else:
                break
    
    # Count semicolons in header
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        semicolon_count = header.count(';')
        print(f"\n✓ Semicolons in header: {semicolon_count}")
        print(f"✓ Expected columns: {semicolon_count + 1}")
    
    # Try parsing with different methods
    print("\n" + "-"*60)
    print("Attempting to parse CSV with multiple encodings...")
    print("-"*60)
    
    # Try multiple encoding and delimiter combinations
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    delimiters = [';', ',']
    
    attempt = 0
    for encoding in encodings:
        for delimiter in delimiters:
            attempt += 1
            try:
                print(f"\n{attempt}. Trying: encoding='{encoding}', sep='{delimiter}'")
                df = pd.read_csv(
                    csv_path, 
                    sep=delimiter, 
                    encoding=encoding,
                    engine='python',
                    on_bad_lines='skip'
                )
                print(f"   ✓ SUCCESS! Loaded {len(df)} rows")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Column count: {len(df.columns)}")
                
                # Check for required columns
                required = ['id_chunk', 'text_original', 'agent_type']
                missing = [col for col in required if col not in df.columns]
                if missing:
                    print(f"   ⚠️  Missing required columns: {missing}")
                else:
                    print(f"   ✓ All required columns present!")
                    
                    # Check agent_type values
                    if 'agent_type' in df.columns:
                        agent_counts = df['agent_type'].value_counts()
                        print(f"\n   Agent type distribution:")
                        for agent, count in agent_counts.items():
                            print(f"     - {agent}: {count} records")
                
                return df
                
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 80:
                    error_msg = error_msg[:77] + "..."
                print(f"   ❌ FAILED: {error_msg}")
    
    print("\n" + "="*60)
    print("All parsing methods failed!")
    print("Please check your CSV file format.")
    print("="*60)
    
    return None


if __name__ == "__main__":
    df = check_csv_file()
    
    if df is not None:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"✓ CSV file parsed successfully")
        print(f"✓ Total records: {len(df)}")
        print(f"✓ Total columns: {len(df.columns)}")
        print("\nSample data (first 2 rows):")
        print(df.head(2).to_string())
        print("\n✓ Ready to run main application!")
    else:
        print("\n❌ CSV parsing failed. Please fix the file and try again.")

