# analysis_app/urls.py

from django.urls import path
from . import views

# Defines the specific URL patterns for the analysis_app.
urlpatterns = [
    # This path maps the 'analyze/' segment to the 'analyze_query' function in views.py.
    # When combined with the project's base path ('api/'), the full endpoint is:
    # /api/analyze/
    path('analyze/', views.analyze_query, name='analyze_query'),
]