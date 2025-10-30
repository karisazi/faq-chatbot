"""
Verify CSV has all the expected columns including new metadata fields
"""
import pandas as pd
import os

csv_path = os.path.join(os.path.dirname(__file__), "resource", "AXA_QNA.csv")

print("\n" + "="*70)
print("CSV COLUMN VERIFICATION")
print("="*70 + "\n")

# Expected columns
expected_columns = [
    'id_chunk',
    'text_original',
    'question_original',
    'agent_type',
    'category_topic',
    'doc_type',
    'source',
    'product_name',
    'insurance_type',
    'topic_focus',
    'coverage_keyword',
    'action_type',
    'entity'
]

try:
    # Try different encodings and delimiters
    df = None
    encodings = ['utf-8', 'latin-1', 'cp1252']
    delimiters = [';', ',']
    
    for encoding in encodings:
        for delimiter in delimiters:
            try:
                df = pd.read_csv(
                    csv_path, 
                    sep=delimiter, 
                    encoding=encoding,
                    on_bad_lines='skip'
                )
                print(f"✓ Successfully loaded CSV with encoding='{encoding}', sep='{delimiter}'\n")
                break
            except Exception as e:
                continue
        if df is not None:
            break
    
    if df is None:
        print("✗ Failed to load CSV file")
        exit(1)
    
    print(f"Total records: {len(df)}\n")
    
    # Check columns
    actual_columns = list(df.columns)
    print("Actual columns in CSV:")
    for i, col in enumerate(actual_columns, 1):
        print(f"  {i}. {col}")
    
    print("\n" + "-"*70 + "\n")
    
    # Check which expected columns are present
    print("Column status:")
    all_present = True
    for col in expected_columns:
        if col in actual_columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} (MISSING)")
            all_present = False
    
    # Check for extra columns
    extra_columns = [col for col in actual_columns if col not in expected_columns]
    if extra_columns:
        print("\n" + "-"*70 + "\n")
        print("Extra columns (not in expected list):")
        for col in extra_columns:
            print(f"  • {col}")
    
    print("\n" + "="*70)
    
    if all_present:
        print("✅ SUCCESS: All expected columns are present!")
    else:
        print("⚠️  WARNING: Some expected columns are missing!")
    
    print("="*70 + "\n")
    
    # Show sample data from new columns
    if all_present:
        print("Sample data from new metadata columns:")
        print("-"*70)
        
        sample_size = min(3, len(df))
        for idx in range(sample_size):
            row = df.iloc[idx]
            print(f"\nRecord {idx + 1}:")
            print(f"  ID: {row.get('id_chunk', 'N/A')}")
            print(f"  Product Name: {row.get('product_name', 'N/A')}")
            print(f"  Insurance Type: {row.get('insurance_type', 'N/A')}")
            print(f"  Topic Focus: {row.get('topic_focus', 'N/A')}")
            print(f"  Coverage Keyword: {row.get('coverage_keyword', 'N/A')}")
            print(f"  Action Type: {row.get('action_type', 'N/A')}")
            print(f"  Entity: {row.get('entity', 'N/A')}")
        
        print("\n" + "="*70 + "\n")

except Exception as e:
    print(f"✗ Error: {e}\n")

