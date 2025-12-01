from django.urls import path
from . import views

urlpatterns = [
    path('analyze/', views.analyze_query, name='analyze_query'),
    path('download/', views.download_csv, name='download_csv'),
]
