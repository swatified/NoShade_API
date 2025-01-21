#!/bin/bash

# Set environment variables
export PATH="/opt/python/3.10.15/bin:$PATH"
export PYTHONPATH="/opt/python/3.10.15/lib/python3.10/site-packages:$PYTHONPATH"
export TRANSFORMERS_OFFLINE=1  # Prevent model downloads during startup
export TRANSFORMERS_CACHE="/home/site/wwwroot/.cache/transformers"

# Install dependencies
python -m pip install --upgrade pip
pip install numpy==1.24.3
pip install -r requirements.txt --no-cache-dir

# Run Django commands
python manage.py collectstatic --noinput
python manage.py migrate

# Start gunicorn with increased timeout
gunicorn API.wsgi:application --bind=0.0.0.0:8000 --timeout 600 --workers 2 --threads 2 --preload