import json
from io import StringIO
import csv
import pandas as pd
from pathlib import Path
import os
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

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    GEMINI_API_KEY = GEMINI_API_KEY.strip().strip('"')

GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY or ''}"

MAX_RETRIES = 5

# --- JSON Schema Definition for Structured Output ---
INTENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "query_type": {
            "type": "STRING",
            "description": "The type of analysis requested. Must be 'time_series' for trend analysis, 'comparison' for comparing multiple subjects, or 'other' if the intent is unclear."
        },
        "metric": {
            "type": "STRING",
            "description": "The primary metric the user is interested in (e.g., 'price', 'demand', 'inventory'). Default to 'price' if unclear."
        },
        "time_range_years": {
            "type": "INTEGER",
            "description": "The number of recent years the user specified (e.g., 3 for 'last 3 years'). Default to 0 if not specified (meaning 'all available years')."
        },
        "comparison_targets": {
            "type": "ARRAY",
            "description": "A list of areas, products, or subjects the user wants to compare (e.g., ['Mumbai', 'Delhi']).",
            "items": {"type": "STRING"}
        }
    },
    "required": ["query_type", "metric"]
}

def extract_query_intent(user_query: str) -> dict | None:
    """
    Calls the Gemini API to extract structured intent from the user's query.
    """
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY missing for intent extraction.")
        return None

    system_prompt = (
        "You are a parser for a real estate analytics application. Analyze the user's query to identify "
        "the analysis type, the metric of interest (price, demand, or unsold_inventory), the requested time frame "
        "in years, and any specific areas to compare. If the time range is not specified, use 0. "
        "**Crucially, only include areas that are EXPLICITLY mentioned in the user's query in comparison_targets. Do NOT assume comparison areas.** "
        "If exactly one area is mentioned, list only that area and set query_type to 'time_series'. "
        "If two or more areas are mentioned, list them and set query_type to 'comparison'."
    )

    payload = {
        "contents": [{ "parts": [{ "text": user_query }] }],
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": INTENT_SCHEMA
        }
    }

    headers = {"Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                GEMINI_API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            response.raise_for_status()

            result = response.json()

            if 'error' in result:
                logging.error(f"Gemini API Error: {result['error'].get('message', 'Unknown API Error')}")
                raise Exception("API Error")

            candidate_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()

            if candidate_text:
                intent = json.loads(candidate_text)
                logging.info(f"Intent extracted: {intent}")
                return intent

        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            logging.error(f"Intent extraction attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return None
    return None

def generate_summary_with_llama(text_to_summarize: str, allowed_areas: list) -> str:
    """
    Calls the Gemini API to generate a professional text summary with exponential backoff.
    Uses the strictest possible system prompt to enforce data grounding.
    Only allowed_areas can be mentioned in the summary.
    """

    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY environment variable is not set or is empty."

    # Build a string of allowed areas
    allowed_areas_str = ', '.join([area.title() for area in allowed_areas])

    system_prompt = (
        "You are a senior real estate market analyst. Review the raw data report provided. "
        "Write a professional, concise, and insightful one-paragraph market summary based ONLY on the facts given. "
        f"**CRITICAL: DO NOT mention any area other than these: {allowed_areas_str}.** "
        "Stick strictly to the area(s) and metrics provided in the report. Highlight key performance indicators like price movement and sales velocity. "
        "Use clear, non-technical language appropriate for a client report."
    )

    user_query = f"Summarize this raw market report:\n\n{text_to_summarize}"

    messages = [
        {"role": "user", "parts": [{"text": user_query}]}
    ]

    payload = {
        "contents": messages,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024}
    }

    headers = {"Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"Attempt {attempt + 1}: Calling Gemini API for summary...")
            response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()

            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '').strip()

            if generated_text:
                logging.info("Gemini API successfully returned summary.")
                return generated_text

            finish_reason = candidate.get('finishReason', 'UNKNOWN')
            if finish_reason == 'MAX_TOKENS':
                return "Error: The AI reached the maximum word count and could not complete the summary."
            return "Error: LLM returned an empty response."

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return f"Error: Failed to get response from LLM after multiple retries. {e}"

    return "Error: Failed to process the request."





# =========================================================================
# === APPLICATION CONFIGURATION & DATA LOADING ===
# =========================================================================
DATA_FILENAME = 'Sample_data.xlsx'
APP_DIR = Path(__file__).resolve().parent
EXCEL_FILE_PATH = APP_DIR / 'data' / DATA_FILENAME

logging.info(f"Attempting to load data from: {EXCEL_FILE_PATH}")

REQUIRED_COLUMNS_MAPPING = {
    'final location': 'area',
    'flat - weighted average rate': 'price',
    'total sold - igr': 'total_sold',
    'total units': 'total_supply',
    'residential sold - igr': 'residential_sold',
    'total carpet area supplied (sqft)': 'size',
    'year': 'year',
}

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
        df = pd.read_excel(EXCEL_FILE_PATH)
        logging.info(f"Successfully loaded data using pd.read_excel.")
    except Exception as e:
        logging.error(f"Error reading Excel file: {e}")
        return None

    if df is None:
        logging.error("Failed to load data.")
        return None

    try:
        # 1. Standardize column names
        rename_dict = {
            original_header: internal_name
            for original_header, internal_name in REQUIRED_COLUMNS_MAPPING.items()
            if original_header in df.columns
        }

        df = df.rename(columns=rename_dict, inplace=False)

        # 2. Data Cleaning and Calculations
        if 'total_supply' in df.columns and 'total_sold' in df.columns:
            df['total_supply'] = pd.to_numeric(df['total_supply'], errors='coerce').fillna(0)
            df['total_sold'] = pd.to_numeric(df['total_sold'], errors='coerce').fillna(0)

            df['demand'] = df['total_sold']
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


def get_default_areas(df, num_areas=3):
    """
    Fallbacks to get the top N areas based on data volume if LLM doesn't specify.
    """
    if df is None or df.empty or 'area' not in df.columns:
        return []

    # Filter out invalid areas before counting
    valid_df = df[df['area'].notna() & (df['area'].astype(str).str.lower() != 'nan')]

    if not valid_df.empty:
        # Sort by the number of data points for a simple measure of relevance
        area_counts = valid_df['area'].value_counts()
        top_areas = area_counts.index.tolist()
        return top_areas[:num_areas]
    return []


def analyze_real_estate(query, df):
    results = {
        "summary": "Analysis failed or data is empty.",
        "chart_data": {'data': [], 'keys': []},
        "table_data": []
    }

    if df is None or df.empty:
        return results

    # Normalize area names in the DataFrame
    df['area_clean'] = df['area'].astype(str).str.strip().str.title()

    # 1. Extract Intent using Gemini
    intent = extract_query_intent(query)

    if not intent:
        logging.warning("Intent extraction failed, running default analysis.")
        intent = {
            'query_type': 'time_series',
            'metric': 'price',
            'time_range_years': 0,
            'comparison_targets': []
        }

    # 2. Determine Targets and Filters based on Intent
    target_areas = intent.get('comparison_targets', [])
    target_areas_clean = [area.strip().title() for area in target_areas]

    available_areas = df['area_clean'].unique().tolist()
    valid_areas_in_data = [area for area in available_areas if area and str(area).lower() != 'nan']

    # Keep only areas that exist in the data
    target_areas_clean = [area for area in target_areas_clean if area in valid_areas_in_data]

    # Fallback to top 3 areas if no valid areas are provided
    if not target_areas_clean:
        target_areas_clean = get_default_areas(df, num_areas=3)
        intent['query_type'] = 'comparison'

    if not target_areas_clean:
        results["summary"] = "Could not identify any valid areas in the data to analyze."
        return results

    # Warn if some requested areas are missing
    missing_areas = set([area.strip().title() for area in target_areas]) - set(target_areas_clean)
    if missing_areas:
        logging.warning(f"Requested areas not found in data: {missing_areas}")

    # Filter the DataFrame
    filtered_df = df[df['area_clean'].isin(target_areas_clean)].copy()

    # Time filter
    time_range_years = intent.get('time_range_years', 0)
    if time_range_years > 0 and 'year' in filtered_df.columns:
        filtered_df = filtered_df[pd.to_numeric(filtered_df['year'], errors='coerce').notna()]
        if not filtered_df.empty:
            filtered_df['year'] = filtered_df['year'].astype('Int64')
            max_year = filtered_df['year'].max()
            start_year = max_year - time_range_years + 1
            filtered_df = filtered_df[filtered_df['year'] >= start_year]
            logging.info(f"Filtered data to last {time_range_years} years (from {start_year} to {max_year}).")

    if filtered_df.empty:
        results["summary"] = f"No data found for the area(s): {', '.join(target_areas_clean)} within the specified time frame."
        return results

    is_comparison = len(target_areas_clean) > 1

    # --- Generate chart data ---
    chart_data_df = pd.DataFrame()
    if all(col in filtered_df.columns for col in ['year', 'price', 'demand', 'unsold_inventory']):
        chart_data_df = filtered_df.groupby(['area_clean', 'year']).agg(
            price=('price', 'mean'),
            demand=('demand', 'sum'),
            unsold_inventory=('unsold_inventory', 'sum')
        ).reset_index()

        chart_json = []
        for _, row in chart_data_df.iterrows():
            chart_json.append({
                'year': int(row['year']) if pd.notna(row['year']) else None,
                'area': row['area_clean'],
                'price': float(row['price']),
                'demand': float(row['demand']),
                'unsold_inventory': float(row['unsold_inventory'])
            })

        requested_metric = intent.get('metric', 'price').lower()
        if requested_metric not in chart_data_df.columns:
            requested_metric = 'price'

        chart_keys = ['year', requested_metric]
        if is_comparison:
            chart_keys.append('area')

        time_text = f" (Last {time_range_years} Years)" if time_range_years > 0 else " (All Available Years)"
        area_text = "Comparison" if is_comparison else target_areas_clean[0]
        label = f"{area_text}: {requested_metric.replace('_', ' ').title()} Trend{time_text}"

        results["chart_data"] = {
            'data': chart_json,
            'keys': chart_keys,
            'label': label,
            'type': 'comparison' if is_comparison else 'timeseries'
        }

    # --- Format filtered table data ---
    table_df = filtered_df.head(10).copy()
    if 'price' in table_df.columns:
        table_df['price_display'] = table_df['price'].apply(lambda x: f'Rs. {x:,.0f}' if pd.notna(x) else 'N/A')
    if 'size' in table_df.columns:
        table_df['size_display'] = table_df['size'].apply(lambda x: f'{x:,.0f} sqft' if pd.notna(x) else 'N/A')
    if 'demand' in table_df.columns:
        table_df['demand_display'] = table_df['demand'].apply(lambda x: f'{int(x):,} units sold' if pd.notna(x) else 'N/A')
    if 'unsold_inventory' in table_df.columns:
        table_df['unsold_display'] = table_df['unsold_inventory'].apply(lambda x: f'{int(x):,} units' if pd.notna(x) else 'N/A')

    final_table_columns = {
        'year': 'Year',
        'area_clean': 'Area',
        'price_display': 'Avg. Price',
        'demand_display': 'Total Sold',
        'unsold_display': 'Unsold Inv.',
        'size_display': 'Avg. Carpet Area',
    }

    cols_to_select = [k for k, v in final_table_columns.items() if k in table_df.columns]
    table_data = table_df[cols_to_select].rename(columns=final_table_columns).to_dict('records')
    results["table_data"] = table_data

    # --- Generate RAW report for LLM ---
    raw_report_parts = []
    area_names = ', '.join(target_areas_clean) if target_areas_clean else "the overall market"
    time_frame = f" from {filtered_df['year'].min()} to {filtered_df['year'].max()}" if 'year' in filtered_df.columns and not filtered_df.empty else ""
    focus_area = target_areas_clean[0] if not is_comparison else None

    if 'demand' in filtered_df.columns and 'price' in filtered_df.columns:
        if not is_comparison:
            # Single area focus
            raw_report_parts.append(f"*** STRICTLY CONFIDENTIAL ANALYSIS FOR: {focus_area.upper()} ***")
            raw_report_parts.append(f"This report is based ONLY on data from {focus_area} {time_frame}.")
            raw_report_parts.append("-" * 40)

            area_data = chart_data_df[chart_data_df['area_clean'] == focus_area].sort_values(by='year')
            if not area_data.empty:
                table_lines = ["Year | Avg Price (Rs.) | Demand (Units Sold) | Unsold Inventory (Units)"]
                table_lines.append("-----|-------------------|---------------------|-------------------------")
                for _, row in area_data.iterrows():
                    table_lines.append(f"{row['year']} | {row['price']:,.0f} | {row['demand']:,.0f} | {row['unsold_inventory']:,.0f}")
                raw_report_parts.extend(table_lines)

                min_year = area_data['year'].min()
                max_year = area_data['year'].max()
                if max_year > min_year:
                    price_start = area_data[area_data['year'] == min_year]['price'].iloc[0]
                    price_end = area_data[area_data['year'] == max_year]['price'].iloc[0]
                    if pd.notna(price_start) and pd.notna(price_end) and price_start > 0:
                        price_change = ((price_end - price_start) / price_start) * 100
                        raw_report_parts.append(f"\nTotal Price Change ({min_year} to {max_year}): {price_change:+.2f}%")

                total_sold = area_data['demand'].sum()
                raw_report_parts.append(f"Total Units Sold in {focus_area}: {total_sold:,.0f}")

            # Hard-block other areas
            other_areas = [a for a in valid_areas_in_data if a not in target_areas_clean]
            if other_areas:
                raw_report_parts.append(f"\n*** DATA ENDED. DO NOT MENTION: {', '.join(other_areas)} ***")

        else:
            # Comparison
            raw_report_parts.append(f"MARKET ANALYSIS REPORT - Focus: {area_names} {time_frame}")
            raw_report_parts.append("-" * 40)
            total_sold = filtered_df['demand'].sum()
            avg_price = filtered_df['price'].mean()
            raw_report_parts.append(f"Total Units Sold (Demand): {total_sold:,.0f}")
            raw_report_parts.append(f"Overall Average Weighted Price: Rs. {avg_price:,.0f}")
            if 'unsold_inventory' in filtered_df.columns:
                total_unsold = filtered_df['unsold_inventory'].sum()
                raw_report_parts.append(f"Total Unsold Inventory: {total_unsold:,.0f} units")
            raw_report_parts.append("\n--- Area Specific Highlights ---")
            for area in target_areas_clean:
                area_df = filtered_df[filtered_df['area_clean'] == area]
                if not area_df.empty:
                    area_sold = area_df['demand'].sum()
                    area_price = area_df['price'].mean()
                    raw_report_parts.append(f"{area}: Units Sold: {area_sold:,.0f}, Avg Price: Rs. {area_price:,.0f}")

    raw_report = "\n".join(raw_report_parts)

    # --- LLM integration with hard-blocking ---
    llm_summary = generate_summary_with_llama(raw_report, allowed_areas=target_areas_clean)

    results["summary"] = llm_summary

    return results



# --- Django View (Entry Point for Analysis) ---

@csrf_exempt
@require_http_methods(["POST"])
def analyze_query(request):
    df = load_and_preprocess_data()

    if df is None or df.empty:
        error_msg = f'Failed to load data. Ensure {DATA_FILENAME} is in analysis_app/data/ and that the required columns are present: Final Location, Flat - Weighted Average Rate, Total Units, and Total Sold - IGR.'
        return JsonResponse({'error': error_msg}, status=500)

    try:
        data = json.loads(request.body)
        user_query = data.get('query', '').strip()

        if not user_query:
            user_query = "Global analysis of price and demand trends for all years"

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
        logging.error(f"CSV download error: {e}")
        return JsonResponse({'error': f'An unexpected error occurred during CSV creation: {str(e)}'}, status=500)

import re

def enforce_area_restrictions(summary: str, allowed_areas: list, all_areas: list) -> str:
    """
    Hard-blocks any area name that is not in `allowed_areas`.
    Removes mentions of other areas entirely.
    """

    allowed_set = set(a.lower() for a in allowed_areas)

    # Areas to block = all areas except allowed ones
    blocked_areas = [a for a in all_areas if a.lower() not in allowed_set]

    # Sort longer names first to avoid substring collisions
    blocked_areas = sorted(blocked_areas, key=len, reverse=True)

    clean_summary = summary

    for area in blocked_areas:
        # Case-insensitive removal of any mention of this blocked area
        pattern = r"\b" + re.escape(area) + r"\b"
        clean_summary = re.sub(pattern, "", clean_summary, flags=re.IGNORECASE)

    # Remove accidental double spaces from removals
    clean_summary = re.sub(r"\s{2,}", " ", clean_summary).strip()

    return clean_summary
def process_user_query(user_query, df, intent_extractor):

    # 1. Extract Intent
    intent = intent_extractor.extract(user_query)

    target_areas = intent.get("areas", [])   # ← MUST EXIST HERE
    query_type   = intent.get("query_type")
    num_years    = intent.get("num_years")

    # ---- Area Validation ----
    valid_areas = df["location"].unique().tolist()

    # Keep only valid
    target_areas = [a for a in target_areas if a in valid_areas]

    if not target_areas:
        return { "summary": "Area not found.", "data": None }

    # Force mode based on count
    if len(target_areas) == 1:
        query_type = "time_series"
    else:
        query_type = "comparison"

    # ---- FILTER DATAFRAME ----
    filtered_df = df[df["location"].isin(target_areas)]


if GLOBAL_DF is None:
    # Attempt to load data on startup (will only log success or failure)
    load_and_preprocess_data()