import pandas as pd
import os

# Define the absolute path to the data file
# Assuming this script (check_columns.py) is in the django-backend directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, 'analysis_app', 'data', 'Sample_data.xlsx')

def get_column_names():
    """Loads the Excel file and prints all column names."""
    print("--- Standalone Column Verification ---")
    print(f"Checking file path: {DATA_FILE_PATH}")
    
    if not os.path.exists(DATA_FILE_PATH):
        print("ERROR: File not found at the specified path.")
        return
    
    try:
        df = pd.read_excel(DATA_FILE_PATH)
        df.columns = df.columns.str.lower()
        
        print("SUCCESS: File loaded.")
        print(f"Total columns found: {len(df.columns)}")
        
        print("\n--- ALL AVAILABLE COLUMN NAMES (LOWERCASE) ---")
        print(list(df.columns))
        print("----------------------------------------------")
        
    except Exception as e:
        print(f"FATAL ERROR during file reading: {e}")

if __name__ == "__main__":
    get_column_names()