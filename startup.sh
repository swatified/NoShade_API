#!/bin/bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py collectstatic --noinput
python manage.py migrate
gunicorn API.wsgi:application --bind=0.0.0.0:8000