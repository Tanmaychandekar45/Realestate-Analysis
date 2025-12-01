import pandas as pd
import os
import random
import json
import time 
import requests 
from typing import Dict, Any, List

# --- Configuration for LLM API ---
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_ENDPOINT = f"{API_BASE_URL}{MODEL_NAME}:generateContent"

# Retry configuration
MAX_RETRIES = 5
INITIAL_DELAY = 1.0 # seconds
REQUEST_TIMEOUT = 15 # seconds

# --- Data Loading Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming the data file is in analysis_app/data/
DATA_FILE_PATH = os.path.join(BASE_DIR, 'data', 'Sample_data.xlsx')

# Global variable to cache the DataFrame once loaded
_data_frame = None

def load_data():
    """Loads, standardizes, and aggregates the multi-segmented real estate data."""
    global _data_frame
    
    if _data_frame is not None:
        return _data_frame

    if not os.path.exists(DATA_FILE_PATH):
        raise FileNotFoundError(f"Data file not found at: {DATA_FILE_PATH}. Please ensure the file exists.")

    try:
        df = pd.read_excel(DATA_FILE_PATH)
        df.columns = df.columns.str.lower().str.strip() # Convert to lowercase and strip whitespace

        # --- 1. Standardize Base Column Names ---
        if 'final location' in df.columns:
            df.rename(columns={'final location': 'area'}, inplace=True)
        
        # --- 2. Aggregate Segmented Data (Price using 'rate') ---
        price_cols = [col for col in df.columns if 'rate' in col]
        
        # --- 3. Calculated Demand Metric (Absorption Rate) ---
        
        TOTAL_UNITS_COL = 'total units'
        TOTAL_SOLD_COL = 'total sold - igr'

        if TOTAL_UNITS_COL not in df.columns:
            raise ValueError(f"Data standardization failed. Missing critical column: '{TOTAL_UNITS_COL}'.")
        if TOTAL_SOLD_COL not in df.columns:
            raise ValueError(f"Data standardization failed. Missing critical column: '{TOTAL_SOLD_COL}'.")
        if not price_cols:
            raise ValueError("Data standardization failed. Could not find any columns related to 'rate' (price).")
        
        # --- EXECUTION: Create aggregated columns ---
        
        # Calculate Demand as Absorption Rate (Sold Units / Total Units)
        # Add a small value (1e-6) to the denominator to prevent division by zero errors
        df['demand'] = df[TOTAL_SOLD_COL] / (df[TOTAL_UNITS_COL] + 1e-6)
        
        # Create new aggregate 'price' column by calculating the row-wise mean of all rate columns
        df['price'] = df[price_cols].mean(axis=1)
        
        # Ensure 'sqft' is present
        if 'sqft' not in df.columns:
             sqft_cols = [col for col in df.columns if 'sqft' in col or 'square' in col]
             if sqft_cols:
                 df.rename(columns={sqft_cols[0]: 'sqft'}, inplace=True)
             else:
                 # If no SQFT is found, create a placeholder column for context 
                 df['sqft'] = 1000 

        # Final check for existence of the critical columns after aggregation
        required_cols = ['area', 'year', 'price', 'demand']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"FATAL: Missing critical columns after standardization: {missing_cols}")
            raise ValueError(f"Data standardization failed. Missing expected columns: {missing_cols}.")
            
        _data_frame = df
        return df

    except Exception as e:
        print(f"Error loading or processing data: {e}")
        # Re-raise the exception to be caught by the view for the 500 error log
        raise


def get_gemini_summary(data_context: str, query: str) -> str:
    """
    Calls the Gemini API to generate a professional market summary 
    based on the provided data context.
    
    Includes exponential backoff for resilience.
    """
    api_key = os.environ.get('GEMINI_API_KEY', '')
    url = f"{API_ENDPOINT}?key={api_key}"
    
    # 1. Define the LLM's role and the task
    system_prompt = (
        "You are 'Atlas AI', a highly concise, professional, and confident real estate market analyst. "
        "Your task is to write the 'MISSION REPORT: ANALYSIS VERDICT' based ONLY on the provided JSON data. "
        "Summarize the key trends, growth percentage, and demand rating in a single paragraph (max 5 sentences). "
        "The report MUST start with a strong, definitive statement about the market trend. "
        "DO NOT use introductory phrases like 'Based on the data' or 'The data shows'."
    )
    
    # 2. Define the user query, incorporating the data
    user_query = (
        f"The user query was: '{query}'.\n\n"
        f"Here is the summarized and raw real estate data context in JSON format: \n"
        f"--- DATA CONTEXT ---\n{data_context}\n---------------------\n\n"
        "Write the MISSION REPORT: ANALYSIS VERDICT for the user, focusing on price change, demand, and overall market conclusion."
    )

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "config": {
            "systemInstruction": system_prompt,
            "temperature": 0.3 # Slightly raised temperature for less repetitive text
        }
    }

    delay = INITIAL_DELAY
    for i in range(MAX_RETRIES):
        try:
            # Use a timeout for the request
            response = requests.post(
                url, 
                headers={'Content-Type': 'application/json'}, 
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT # Added timeout
            )
            response.raise_for_status() # Raise exception for 4xx/5xx

            result = response.json()
            
            # Extract the generated text safely
            summary = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
            
            if summary:
                final_summary = summary.strip()
                # CRITICAL: Print the summary to the terminal for debugging
                print("\n--- LLM GENERATED SUMMARY (VERIFIED) ---")
                print(final_summary)
                print("------------------------------------------\n")
                return final_summary
            
            # If API call succeeds but returns no text, this fallback is used
            return "LLM Analysis failed: Received empty content from API."

        except requests.exceptions.RequestException as e:
            if i < MAX_RETRIES - 1:
                # Retry with exponential backoff
                print(f"LLM API Error (Retry {i+1}): {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"LLM API failed after {MAX_RETRIES} attempts.")
                return f"LLM Integration Failure: Max retries reached. Error: {type(e).__name__} - {e}"
        except Exception as e:
            print(f"Unexpected error processing LLM response (likely JSON/Parsing issue): {e}")
            return "LLM Integration Failure: Invalid response structure received."

    return "LLM Integration Failure: Max retries reached and failed all retries."


def process_query(query: str) -> Dict[str, Any]:
    """
    Analyzes the data based on the user's query and returns structured results,
    now including an LLM-generated summary.
    """
    df = load_data()
    
    # --- 1. Query Keyword Extraction ---
    query_lower = query.lower()
    all_areas = df['area'].unique().tolist()
    target_areas = [area for area in all_areas if area.lower() in query_lower]
    
    if not target_areas:
        # Default to a random sample if no specific area is mentioned
        target_areas = random.sample(all_areas, min(3, len(all_areas)))

    # --- 2. Data Filtering ---
    filtered_df = df[df['area'].isin(target_areas)].copy()

    # The load_data function now ensures 'price' and 'demand' exist.
    
    # --- 3. Time Series Aggregation (Chart Data) ---
    chart_data_df = filtered_df.groupby('year').agg(
        {'price': 'mean', 'demand': 'mean'}
    ).reset_index()

    chart_data = chart_data_df.rename(columns={
        'year': 'Year',
        'price': 'Avg Price',
        'demand': 'Avg Demand'
    }).to_dict('records')

    # --- 4. Prepare Context for LLM ---
    # Convert chart data and a sample of raw data to a single JSON string for the LLM context
    llm_context = {
        "analysis_areas": target_areas,
        "time_series_summary": chart_data,
        "sample_raw_data": filtered_df[['area', 'year', 'price', 'demand', 'sqft']].head(5).to_dict('records')
    }
    data_context_json = json.dumps(llm_context, indent=2)

    # --- 5. Summary Generation (LLM Call) ---
    summary_text = get_gemini_summary(data_context_json, query)
    
    # --- 6. Raw Data (Table Data) ---
    # Note: Using the newly aggregated 'price' and 'demand' columns
    table_cols = ['area', 'year', 'price', 'demand', 'sqft'] 
    table_data = filtered_df[table_cols].to_dict('records')

    # --- 7. Final Structured Output ---
    return {
        "summary": summary_text,
        "chart_data": chart_data,
        "table_data": table_data
    }