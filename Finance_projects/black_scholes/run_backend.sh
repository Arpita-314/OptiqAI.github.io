#!/bin/bash
# Start Black-Scholes Backend Server

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the FastAPI server
python backend_api.py

