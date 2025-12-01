from django.urls import path
from .views import analyze_query, download_csv

urlpatterns = [
    path('analyze/', analyze_query, name='analyze_query'),
    path('download/', download_csv, name='download_csv'),
]
