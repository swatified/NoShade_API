#!/bin/bash
export PATH="/opt/python/3.10.15/bin:$PATH"
export PYTHONPATH="/opt/python/3.10.15/lib/python3.10/site-packages:$PYTHONPATH"

python -m pip install --upgrade pip
pip install numpy==1.24.3
pip install -r requirements.txt
python manage.py collectstatic --noinput
python manage.py migrate
gunicorn API.wsgi:application --bind=0.0.0.0:8000 --timeout 600 --workers 4