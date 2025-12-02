#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -o errexit

# Change directory to the Django project root before running manage.py
cd django-backend

# 1. Run database migrations
echo "Running database migrations..."
python manage.py makemigrations 
python manage.py migrate --noinput

# 2. Collect static files for production serving
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear

echo "Deployment script finished successfully."