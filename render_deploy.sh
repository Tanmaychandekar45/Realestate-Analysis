#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Apply database migrations
echo "Running collectstatic..."
python manage.py collectstatic --noinput

# 2. Start the Gunicorn server
# Note: We explicitly set the PYTHONPATH to the current directory (.), 
# ensuring Python can find the 'django_backend' module.
echo "Starting Gunicorn server..."
exec gunicorn django_backend.wsgi:application --bind 0.0.0.0:$PORT  