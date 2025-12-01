# analysis_app/views.py
import json
from io import StringIO
import csv # <--- NEW IMPORT
import pandas as pd
from pathlib import Path
import os
import re
import logging 

from django.http import JsonResponse, HttpResponse # <--- Updated imports
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST # <--- NEW IMPORT

logging.basicConfig(level=logging.INFO)

# =========================================================================
# === CONFIGURATION ===
# =========================================================================
DATA_FILENAME = 'Sample_data.xlsx'  # <<< CHANGED TO .xlsx
APP_DIR = Path(__file__).resolve().parent 
EXCEL_FILE_PATH = APP_DIR / 'data' / DATA_FILENAME

logging.info(f"Attempting to load data from: {EXCEL_FILE_PATH}") 

# --- 1. Load and Pre-process Data ---
GLOBAL_DF = None

# MAPPING based on your Excel file's column headers.
# Keys are the standardized (lowercase, stripped) names from the Excel file.
# Values are the internal names used by the analysis logic.
REQUIRED_COLUMNS_MAPPING = {
    'final location': 'area', # Note: Excel column headers usually retain spaces before standardization
    'flat - weighted average rate': 'price',  # Exact match for your price column
    'total sold - igr': 'total_sold',  # Exact match for your sales column (used for demand)
    'total units': 'total_supply',  # Exact match for supply
    'residential sold - igr': 'residential_sold', 
    'total carpet area supplied (sqft)': 'size', 
    'year': 'year',
}

ANALYSIS_COLUMNS = ['year', 'area', 'price', 'demand', 'unsold_inventory', 'size']

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
        existing_cols = [col for col in ['year', 'area', 'price', 'demand'] if col in df.columns]
        
        if len(existing_cols) < 3:
             logging.error(f"Critical columns missing after processing. Found: {df.columns.tolist()}")
             # List the headers found in the original file to help the user debug if necessary
             missing_cols_msg = ", ".join([col for col in ['area', 'price', 'total_sold', 'total_supply'] if col not in df.columns])
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
    detected_areas = []
    
    # Simple, non-fuzzy check for areas
    for area in available_areas:
        # Check if the area name is present as a whole word (using regex word boundary \b)
        if re.search(r'\b' + re.escape(area.lower()) + r'\b', query_lower):
             detected_areas.append(area)

    if not detected_areas and len(available_areas) > 0:
        valid_areas = [area for area in available_areas if area and area.lower() != 'nan']
        unique_areas = list(set(valid_areas))
        # Default: Return top 3 areas if none detected
        detected_areas = unique_areas[:3] if len(unique_areas) >= 3 else unique_areas
        
    return detected_areas, is_comparison

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
    target_areas, is_comparison = parse_query_for_areas(query, available_areas)
    
    target_areas = [area for area in target_areas if area and area.lower() != 'nan']

    if not target_areas:
        results["summary"] = "Could not identify specific real estate areas in your query. Please specify an area like 'Analyze Wakad' or 'Compare Hinjawadi and Aundh'."
        return results

    filtered_df = df[df['area'].isin(target_areas)].copy()
    
    if filtered_df.empty:
        results["summary"] = f"No data found for the area(s): {', '.join(target_areas)}. Please check the spelling or try another area."
        return results

    # --- Generate Chart Data ---
    if all(col in filtered_df.columns for col in ['year', 'price', 'demand']):
        # Aggregate data by area and year
        chart_data_df = filtered_df.groupby(['area', 'year']).agg(
            price=('price', 'mean'),
            demand=('demand', 'sum') 
        ).reset_index()
        
        # Convert to list of dictionaries for JSON, keeping area separate for comparison charts
        chart_json = []
        for _, row in chart_data_df.iterrows():
            chart_json.append({
                'year': row['year'],
                'area': row['area'],
                'price': row['price'],
                'demand': row['demand']
            })
        
        # If comparison, keys should include 'area' for dynamic visualization
        if is_comparison or len(target_areas) > 1:
            chart_keys = ['year', 'area', 'price', 'demand']
        else:
            # If single area, simplify keys
            chart_keys = ['year', 'price', 'demand']
            
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
        table_df['price_display'] = 'Rs. ' + table_df['price'].map('{:,.0f}'.format)
    if 'size' in table_df.columns:
        table_df['size_display'] = table_df['size'].map('{:,.0f} sqft'.format)
    if 'demand' in table_df.columns:
        table_df['demand_display'] = table_df['demand'].astype('Int64').astype(str) + ' units sold'
    if 'unsold_inventory' in table_df.columns:
        table_df['unsold_display'] = table_df['unsold_inventory'].astype('Int64').astype(str) + ' units'

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
    
    # --- Generate Summary ---
    summary_parts = []
    area_names = ', '.join(target_areas)
    
    # Robust summary generation using available columns
    if 'demand' in filtered_df.columns and 'price' in filtered_df.columns:
        total_sold = filtered_df['demand'].sum()
        avg_price = filtered_df['price'].mean()
        
        summary_parts.append(f"**Analysis for {area_names}**:")
        summary_parts.append(f"The market in {area_names} has recorded a total of **{total_sold:,.0f} units sold** in the filtered period.")
        summary_parts.append(f"The average weighted price is **Rs. {avg_price:,.0f}**.")
        
        if 'unsold_inventory' in filtered_df.columns and filtered_df['unsold_inventory'].sum() > 0:
             total_unsold = filtered_df['unsold_inventory'].sum()
             summary_parts.append(f"The total unsold inventory across the recorded period is **{total_unsold:,.0f} units**.")
        elif 'unsold_inventory' in filtered_df.columns:
             summary_parts.append("Inventory data suggests low or zero unsold units based on available records.")


    else:
        summary_parts.append(f"Basic analysis for {area_names}: Data loaded, but key metrics (price/demand) were not found in the expected columns.")

    results["summary"] = "\n\n".join(summary_parts)
    
    return results

# --- Django View (Entry Point for Analysis) ---

@csrf_exempt
def analyze_query(request):
    df = load_and_preprocess_data()

    if df is None or df.empty:
        error_msg = f'Failed to load data. Ensure {DATA_FILENAME} is in analysis_app/data/ and that the required columns are present: Final Location, Flat - Weighted Average Rate, Total Units, and Total Sold - IGR.'
        return JsonResponse({'error': error_msg}, status=500)

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_query = data.get('query', '').strip()

            if not user_query:
                return JsonResponse({'error': 'No query provided.'}, status=400)

            analysis_results = analyze_real_estate(user_query, df)

            return JsonResponse({
                'summary': analysis_results['summary'],
                'chartData': analysis_results['chart_data'],
                'tableData': analysis_results['table_data'],
            }, status=200)

        except Exception as e:
            logging.error(f"Analysis error: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'POST required'}, status=405)

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
        
        # Write the data rows
        csv_writer.writerows(data)
            
        # 3. Create the HTTP response object
        response = HttpResponse(output.getvalue(), content_type='text/csv')
        
        # Set the filename and attachment header
        response['Content-Disposition'] = 'attachment; filename="analysis_results.csv"'
        
        return response

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON format in request body.'}, status=400)
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"CSV download error: {e}")
        return JsonResponse({'error': f'An unexpected error occurred during CSV creation: {str(e)}'}, status=500)


if GLOBAL_DF is None:
    # Attempt to load data on startup (will only log success or failure)
    load_and_preprocess_data()