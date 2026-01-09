#!/bin/bash

# Start Ollama explicitly
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 5
fi

# 1. Ingest (check if needed or run every time)
if [ ! -d "./chroma_db" ]; then
    echo "Running Ingestion..."
    python ingest.py
fi

# 2. Start API in background
echo "Starting FastAPI backend..."
uvicorn rag_service:app --reload &
API_PID=$!

# 3. Start Dashboard
echo "Starting Streamlit Dashboard..."
streamlit run monitoring.py &
DASH_PID=$!

echo "Services running. API: http://localhost:8000, Dashboard: http://localhost:8501"
echo "Press CTRL+C to stop all."

trap "kill $API_PID $DASH_PID; exit" INT
wait
