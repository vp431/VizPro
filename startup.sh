#!/bin/bash

# Log startup information
echo "Starting VizPro application..."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# Set the number of worker processes to 1 to avoid memory issues
# Timeout is increased to 300 seconds
# Preload the app for faster response
gunicorn --bind=0.0.0.0:8000 --workers=1 --timeout=300 --preload app:server 