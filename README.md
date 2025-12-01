
# Atlas : Real EstateMarket Analysis Console


## Overview

Project Atlas is a full-stack data analysis application designed to provide quick, insightful reports on real estate market data. The application uses a Python/Django backend for data processing (using Pandas) and a React frontend (single App.jsx file) for a responsive user interface, including charts and a raw data matrix.

## Key Features:

Query-based Analysis: Analyze data based on location (e.g., Wakad, Baner).

Time-Series Visualization: Dynamic charts showing price and demand trends over time.

Summary Reports: Conversational summaries of key market findings.

CSV Download: Export filtered data directly from the analysis view.

## 💾 Data Requirement

This application relies on a single Excel file for its analysis:

File: Sample_data.xlsx

Location (Backend): This file must be placed inside the Django application's data directory:

analysis_app/data/Sample_data.xlsx


The application is specifically configured to load and process the column names found in the provided sample data.

## 🛠️ Backend Setup (Django/Python)

The backend handles data loading, filtering, aggregation, and serving the analysis results.

1. Prerequisites

Python 3.8+

Django (installed via pip)

Pandas (for data manipulation)

Openpyxl (required by Pandas to read .xlsx files)

2. File Structure

The project assumes the following structure:

/project_root
├──
requirements.txt
├── manage.py 
└── analysis_app/
    ├── __init__.py
    ├── urls.py
    ├── views.py  <-- The main analysis logic
    └── data/
        └── Sample_data.xlsx <-- The required data file


3. Installation

Install the necessary Python packages:

    pip install django pandas openpyxl numpy


4. Running the Server

Start the Django development server:

    python manage.py runserver


The API endpoints will be accessible at http://127.0.0.1:8000/api/.

## 🌐 API Endpoints

The frontend communicates with the following endpoints defined in analysis_app/urls.py:

Endpoint

Method

Description

/api/analyze/

POST

Runs the market analysis based on the query provided in the JSON body. Returns summary, chart, and table data.

/api/download/

POST

Filters the data based on the query and returns the complete filtered dataset as a CSV file attachment.

## 💻 Frontend Setup (React/JSX)

The frontend is contained entirely within a single component file.

1. File Location

realestate-frontend/src/App.jsx

2. Dependencies

The React component uses:

recharts for charting.

lucide-react for icons.

Tailwind CSS (assumed available in the hosting environment for styling).

3. Running the Frontend (Local Development)

Since the frontend is a single JSX file, it can be easily integrated into any standard React project initialized via npm create vite@latest or similar tools.

Crucial Note on API:
The App.jsx file contains the line:

const API_BASE_URL = '[http://127.0.0.1:8000](http://127.0.0.1:8000)'; 


If your Django backend is running on a different host or port, you must update this constant in App.jsx to match.

