import json
from io import StringIO
import csv
import pandas as pd
from pathlib import Path
import os
import re
import logging 
import time 
import requests 
from requests.exceptions import RequestException 

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.views.decorators.http import require_http_methods 

logging.basicConfig(level=logging.INFO)

# =========================================================================
# === LLM CONFIGURATION & UTILITY (Gemini API) ===
# =========================================================================

# --- CRITICAL FIX APPLIED HERE ---
# 1. Get the key by its variable name 'GEMINI_API_KEY'.
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') 

# 2. IMPORTANT: If the key exists, strip any surrounding whitespace and quotes (") 
# that might have been accidentally read from the .env file.
if GEMINI_API_KEY:
    GEMINI_API_KEY = GEMINI_API_KEY.strip().strip('"') 
# --- END CRITICAL FIX ---

GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025" 
# Ensure the API key is safely embedded in the URL if it exists
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY or ''}"

MAX_RETRIES = 5

def generate_summary_with_llama(text_to_summarize: str) -> str:
    """
    Calls the Gemini API to generate a professional text summary 
    with exponential backoff for robust communication.
    """
    
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY environment variable is not set or is empty. Cannot call the LLM API."

    # 1. Define the conversation history (System and User messages)
    system_prompt = "You are a senior real estate market analyst. Review the raw data report provided. Write a professional, concise, and insightful one-paragraph market summary based ONLY on the facts given. Highlight key performance indicators like price movement and sales velocity. Use clear, non-technical language appropriate for a client report."
    user_query = f"Summarize this raw market report:\n\n{text_to_summarize}"
    
    messages = [
        {
            "role": "user",
            "parts": [{"text": user_query}]
        }
    ]

    # 2. Construct the API Payload
    payload = {
        "contents": messages,
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": { 
            "temperature": 0.3, # Low temperature for accurate summarization
            # FIX: Increased maxOutputTokens to 1024 to prevent MAX_TOKENS cutoff
            "maxOutputTokens": 1024 
        }
    }

    headers = {
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"Attempt {attempt + 1}: Calling Gemini API...")
            
            # 3. Make the external API Call (Key is in the URL as a query parameter)
            response = requests.post(
                GEMINI_API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            response.raise_for_status()

            # 4. Process the LLM Response
            result = response.json()
            
            # Check for potential error in API response before trying to extract text
            if 'error' in result:
                error_message = result['error'].get('message', 'Unknown API Error')
                logging.error(f"Gemini API returned an error: {error_message}")
                # Raise an HTTP error to trigger the retry/error handling
                response.raise_for_status() 

            # Extract generated text from the standard Gemini API response structure
            candidate = result.get('candidates', [{}])[0]
            generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '').strip()
            
            if generated_text:
                logging.info("Gemini API successfully returned summary.")
                return generated_text
            
            # Log specific reason if available
            finish_reason = candidate.get('finishReason', 'UNKNOWN')
            logging.error(f"Gemini returned an empty response or unexpected structure (Finish Reason: {finish_reason}): {result}")
            
            # If the finish reason is still MAX_TOKENS, even with the increase, provide a helpful error.
            if finish_reason == 'MAX_TOKENS':
                return "Error: The AI reached the maximum word count and could not complete the summary. The requested data might be too extensive for a one-paragraph summary."
            
            return "Error: LLM returned an empty response."

        except requests.exceptions.HTTPError as http_err:
            # Safely attempt to extract detailed error message from response body
            error_details = "No specific error message."
            try:
                error_details = response.json().get('error', {}).get('message', 'No specific error message.')
            except json.JSONDecodeError:
                pass # Body wasn't JSON, use default error message
                
            logging.error(f"HTTP Error: {http_err}. Details: {error_details}")

            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return f"Error: Failed to receive a valid response from the API after multiple retries. Details: {error_details}"
                
        except (RequestException, json.JSONDecodeError, IndexError, KeyError) as e:
            logging.error(f"Attempt {attempt + 1} failed due to API/JSON error: {e}")
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return "Error: Failed to receive a valid response from the LLM API after multiple retries."
        except Exception as e:
            return f"An unexpected error occurred during LLM call: {e}"
    
    return "Error: Failed to process the request."


# =========================================================================
# === APPLICATION CONFIGURATION ===
# =========================================================================
DATA_FILENAME = 'Sample_data.xlsx'
APP_DIR = Path(__file__).resolve().parent 
EXCEL_FILE_PATH = APP_DIR / 'data' / DATA_FILENAME

logging.info(f"Attempting to load data from: {EXCEL_FILE_PATH}") 

# MAPPING based on your Excel file's column headers.
REQUIRED_COLUMNS_MAPPING = {
    'final location': 'area',
    'flat - weighted average rate': 'price',
    'total sold - igr': 'total_sold',
    'total units': 'total_supply',
    'residential sold - igr': 'residential_sold', 
    'total carpet area supplied (sqft)': 'size', 
    'year': 'year',
}

ANALYSIS_COLUMNS = ['year', 'area', 'price', 'demand', 'unsold_inventory', 'size']

GLOBAL_DF = None

def load_and_preprocess_data():
    """
    Loads data from the XLSX file path and performs necessary cleaning and standardization.
    """
    global GLOBAL_DF
    if GLOBAL_DF is not None:
        return GLOBAL_DF

    if not os.path.exists(EXCEL_FILE_PATH):
        logging.error(f"Error: Data file not found at the configured path: {EXCEL_FILE_PATH}") 
        return None
    
    df = None
    
    try:
        # Use read_excel for .xlsx files
        df = pd.read_excel(EXCEL_FILE_PATH)
        logging.info(f"Successfully loaded data using pd.read_excel.")
    except Exception as e:
        logging.error(f"Error reading Excel file: {e}")
        return None

    if df is None:
        logging.error("Failed to load data.")
        return None

    try:
        # 1. Standardize column names (to allow for easy mapping)
        
        # Prepare a dictionary for renaming (Original Header -> Internal Name)
        rename_dict = {
            original_header: internal_name 
            for original_header, internal_name in REQUIRED_COLUMNS_MAPPING.items()
            if original_header in df.columns # Only include columns present in the dataframe
        }
        
        df = df.rename(columns=rename_dict, inplace=False)

        # 2. Data Cleaning and Calculations
        
        # Ensure we have the columns needed for calculations
        if 'total_supply' in df.columns and 'total_sold' in df.columns:
            df['total_supply'] = pd.to_numeric(df['total_supply'], errors='coerce').fillna(0)
            df['total_sold'] = pd.to_numeric(df['total_sold'], errors='coerce').fillna(0)
            
            # Demand = Total Sold
            df['demand'] = df['total_sold']
            # Unsold Inventory = Total Supply - Total Sold
            df['unsold_inventory'] = df['total_supply'] - df['total_sold']
        else:
            logging.warning("Missing columns for demand calculation. Creating placeholders.")
            df['demand'] = 0
            df['unsold_inventory'] = 0

        # Standard type conversions
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        if 'area' in df.columns:
            df['area'] = df['area'].astype(str).str.strip().str.title()
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')

        # Filter for essential columns
        essential_cols = ['year', 'area', 'price', 'demand']
        existing_cols = [col for col in essential_cols if col in df.columns]
        
        if len(existing_cols) < 3:
            logging.error(f"Critical columns missing after processing. Found: {df.columns.tolist()}")
            # List the headers found in the original file to help the user debug if necessary
            missing_cols_msg = ", ".join([col for col in essential_cols if col not in df.columns])
            if missing_cols_msg:
                raise ValueError(f"Required columns were not found or mapped: {missing_cols_msg}")
            return pd.DataFrame()

        df = df.dropna(subset=existing_cols)
        
        GLOBAL_DF = df
        logging.info(f"Data loaded successfully. Total records: {len(df)}")
        return df
    
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return None

# --- 2. Query Parsing Helper ---

def parse_query_for_areas(query, available_areas):
    query_lower = query.lower()
    comparison_keywords = ['compare', 'vs', 'and', 'with'] 
    is_comparison = any(kw in query_lower for kw in comparison_keywords)
    detected_areas_set = set()
    
    # Simple, non-fuzzy check for areas
    available_areas_lower_map = {area.lower(): area for area in available_areas if area is not None}
    
    for area_lower, area_original in available_areas_lower_map.items():
        # Check if the area name is present as a whole word (using regex word boundary \b)
        # This is the strict detection for explicit mentions.
        if re.search(r'\b' + re.escape(area_lower) + r'\b', query_lower):
            detected_areas_set.add(area_original)

    # CRITICAL FIX: The previous default/fallback logic is removed from this function.
    # This function now ONLY returns areas explicitly detected in the query.
    # The decision to use a default set of areas (top 3) is moved to the calling function,
    # ensuring that a successful single detection prevents the fallback.
        
    return list(detected_areas_set), is_comparison

# --- 3. Core Analysis Function ---

def analyze_real_estate(query, df):
    results = {
        "summary": "Analysis failed or data is empty.",
        "chart_data": [],
        "table_data": []
    }
    
    if df is None or df.empty:
        return results

    available_areas = df['area'].unique().tolist()
    
    # 1. Attempt to find explicit area mentions
    target_areas, is_comparison = parse_query_for_areas(query, available_areas)
    
    target_areas = [area for area in target_areas if area and area.lower() != 'nan']

    if not target_areas:
        # 2. If no area is explicitly detected (e.g., query was "Global analysis"), 
        #    we apply the default fallback to the top 3 areas.
        #    This block is now ONLY executed when the initial specific detection fails,
        #    ensuring that a successful single-area query (like "analyze baner") 
        #    does NOT fall into this default logic.
        
        valid_areas = [area for area in available_areas if area and area.lower() != 'nan']
        if valid_areas and GLOBAL_DF is not None and not GLOBAL_DF.empty:
            
            # Use value_counts to find the top 3 most represented areas
            area_counts = GLOBAL_DF['area'].value_counts()
            top_areas = area_counts.index.tolist()[:3]
            target_areas = [area for area in top_areas if area and area.lower() != 'nan']
        
        is_comparison = False # Treat as a single multi-area analysis (the default view)

    filtered_df = df[df['area'].isin(target_areas)].copy()
    
    if filtered_df.empty:
        # Improved error message when target_areas is empty after all attempts (specific or default)
        area_names = ', '.join(target_areas) if target_areas else "the queried location"
        results["summary"] = f"No data found for {area_names}. Please check the spelling or try another area. Available areas include: {', '.join(available_areas[:5])}..."
        return results

    # --- Generate Chart Data ---
    chart_data_df = pd.DataFrame()
    if all(col in filtered_df.columns for col in ['year', 'price', 'demand']):
        # Aggregate data by area and year
        chart_data_df = filtered_df.groupby(['area', 'year']).agg(
            price=('price', 'mean'),
            demand=('demand', 'sum') 
        ).reset_index()
        
        # Convert to list of dictionaries for JSON, ensuring proper type conversion for JSON serialization
        chart_json = []
        for _, row in chart_data_df.iterrows():
            chart_json.append({
                'year': int(row['year']) if pd.notna(row['year']) else None,
                'area': row['area'],
                'price': float(row['price']),
                'demand': float(row['demand'])
            })
        
        # Determine the keys needed for the frontend visualization
        chart_keys = ['year', 'price', 'demand']
        if is_comparison or len(target_areas) > 1:
            chart_keys.append('area') 
            
        results["chart_data"] = {
            'data': chart_json,
            'keys': chart_keys,
            'label': 'Price and Demand Trends Over Time'
        }
    
    # --- Format Filtered Table Data ---
    
    # Show a representative sample (e.g., first 10 rows)
    table_df = filtered_df.head(10).copy()
    
    # Create display versions of numerical columns
    if 'price' in table_df.columns:
        # Use .apply(lambda x: ...) for reliable formatting across different pandas versions and dtypes
        table_df['price_display'] = table_df['price'].apply(
            lambda x: f'Rs. {x:,.0f}' if pd.notna(x) else 'N/A'
        )
    if 'size' in table_df.columns:
        table_df['size_display'] = table_df['size'].apply(
            lambda x: f'{x:,.0f} sqft' if pd.notna(x) else 'N/A'
        )
    if 'demand' in table_df.columns:
        table_df['demand_display'] = table_df['demand'].apply(
            lambda x: f'{int(x):,} units sold' if pd.notna(x) else 'N/A'
        )
    if 'unsold_inventory' in table_df.columns:
        # NOTE: Logic unchanged per user request
        table_df['unsold_display'] = table_df['unsold_inventory'].apply(
            lambda x: f'{int(x):,} units' if pd.notna(x) else 'N/A'
        )

    final_table_columns = {
        'year': 'Year',
        'area': 'Area',
        'price_display': 'Avg. Price',
        'demand_display': 'Total Sold',
        'unsold_display': 'Unsold Inv.',
        'size_display': 'Avg. Carpet Area',
    }
    
    cols_to_select = [k for k, v in final_table_columns.items() if k in table_df.columns]
    table_data = table_df[cols_to_select].rename(columns=final_table_columns).to_dict('records')

    results["table_data"] = table_data
    
    # --- Generate RAW Report for LLM ---
    
    raw_report_parts = []
    
    # Filter out empty or 'Nan' areas for the summary text
    clean_target_areas = [area for area in target_areas if area and area.lower() != 'nan']
    area_names = ', '.join(clean_target_areas) if clean_target_areas else "the overall market"
    
    raw_report_parts.append(f"MARKET ANALYSIS REPORT - Focus Areas: {area_names}")
    raw_report_parts.append("-" * 40)
    
    # Global Stats for the filtered data
    if 'demand' in filtered_df.columns and 'price' in filtered_df.columns:
        total_sold = filtered_df['demand'].sum()
        avg_price = filtered_df['price'].mean()
        
        raw_report_parts.append(f"Total Units Sold (Demand): {total_sold:,.0f}")
        raw_report_parts.append(f"Overall Average Weighted Price: Rs. {avg_price:,.0f}")
        
        if 'unsold_inventory' in filtered_df.columns:
            total_unsold = filtered_df['unsold_inventory'].sum()
            raw_report_parts.append(f"Total Unsold Inventory: {total_unsold:,.0f} units")

        # Analyze Price Trend (Requires chart_data_df from above)
        if not chart_data_df.empty and 'year' in chart_data_df.columns:
            min_year = chart_data_df['year'].min()
            max_year = chart_data_df['year'].max()
            
            # Calculate Y-o-Y change for the overall average price
            if max_year > min_year:
                # Get data for the two extremes of the time period
                price_start = chart_data_df[chart_data_df['year'] == min_year]['price'].mean()
                price_end = chart_data_df[chart_data_df['year'] == max_year]['price'].mean()
                
                if pd.notna(price_start) and pd.notna(price_end) and price_start > 0:
                    price_change = ((price_end - price_start) / price_start) * 100
                    raw_report_parts.append(f"Price Change from {min_year} to {max_year}: {price_change:+.2f}%")

        raw_report = "\n".join(raw_report_parts)
        
        # --- LLM Integration: Call Gemini to summarize the RAW Report ---
        llm_summary = generate_summary_with_llama(raw_report)
        results["summary"] = llm_summary
        
    else:
        results["summary"] = f"Basic analysis for {area_names}: Data loaded, but key metrics (price/demand) were insufficient to generate a comprehensive report."

    return results

# --- Django View (Entry Point for Analysis) ---

@csrf_exempt
@require_http_methods(["POST"]) # Use require_http_methods for clarity
def analyze_query(request):
    df = load_and_preprocess_data()

    if df is None or df.empty:
        error_msg = f'Failed to load data. Ensure {DATA_FILENAME} is in analysis_app/data/ and that the required columns are present: Final Location, Flat - Weighted Average Rate, Total Units, and Total Sold - IGR.'
        return JsonResponse({'error': error_msg}, status=500)

    try:
        data = json.loads(request.body)
        user_query = data.get('query', '').strip()

        if not user_query:
            # If no query, default to a global analysis (handled within analyze_real_estate)
            user_query = "Global analysis"
            
        analysis_results = analyze_real_estate(user_query, df)

        # Ensure the response structure is correct for the frontend
        return JsonResponse({
            'summary': analysis_results.get('summary', 'No summary generated.'),
            'chartData': analysis_results.get('chart_data', {'data': [], 'keys': []}),
            'tableData': analysis_results.get('table_data', []),
        }, status=200)

    except Exception as e:
        logging.exception("Unexpected backend error")
        return JsonResponse({'error': f'Backend crashed: {str(e)}'}, status=500)


# --- Django View (Entry Point for CSV Download) ---

@csrf_exempt
@require_POST
def download_csv(request):
    """
    Takes the JSON data (processed table data) from the request body, 
    converts it to a CSV file, and returns it as a downloadable attachment.
    """
    try:
        # 1. Parse the JSON data sent from the frontend
        data = json.loads(request.body)
        
        # We expect the data to be a list of objects (rows)
        if not isinstance(data, list) or not data:
            return JsonResponse({'error': 'Invalid or empty data payload.'}, status=400)

        # 2. Prepare the CSV content in memory
        output = StringIO()
        
        # Use csv.DictWriter since the data is a list of dictionaries
        fieldnames = list(data[0].keys())
        csv_writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        # Write the header row
        csv_writer.writeheader()
        
        csv_writer.writerows(data)
            
        # 3. Create the HTTP response object
        response = HttpResponse(output.getvalue(), content_type='text/csv')
        
        # Set the filename and attachment header
        response['Content-Disposition'] = 'attachment; filename="analysis_results.csv"'
        
        return response

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON format in request body for CSV download.'}, status=400)
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"CSV download error: {e}")
        return JsonResponse({'error': f'An unexpected error occurred during CSV creation: {str(e)}'}, status=500)


if GLOBAL_DF is None:
    # Attempt to load data on startup (will only log success or failure)
    # The view functions will handle the case where it fails.
    load_and_preprocess_data()