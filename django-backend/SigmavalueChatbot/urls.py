"""SigmavalueChatbot URL Configuration

The `urlpatterns` list routes URLs to views.
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # Optional: Django Admin interface
    path('admin/', admin.site.urls),
    
    # Crucial: Route all paths starting with 'api/' to the analysis_app's URL patterns.
    path('api/', include('analysis_app.urls')),
]