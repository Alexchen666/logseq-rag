#!/bin/bash

# Exit on any error
set -e

echo "Starting Logseq RAG System..."

# Activate the virtual environment created by uv
source .venv/bin/activate

# Set default Streamlit configuration
export STREAMLIT_SERVER_PORT=${PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=${HOST:-0.0.0.0}
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create necessary directories if they don't exist
mkdir -p /app/data
mkdir -p /app/logs

# Print environment info
echo "Python version: $(python --version)"
echo "Streamlit version: $(streamlit --version)"
echo "Starting Streamlit on ${STREAMLIT_SERVER_ADDRESS}:${STREAMLIT_SERVER_PORT}"

# Start the Streamlit application
exec streamlit run app.py \
    --server.port=${STREAMLIT_SERVER_PORT} \
    --server.address=${STREAMLIT_SERVER_ADDRESS} \
    --server.headless=${STREAMLIT_SERVER_HEADLESS} \
    --browser.gatherUsageStats=${STREAMLIT_BROWSER_GATHER_USAGE_STATS} \