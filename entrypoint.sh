#!/bin/sh

# Start Ollama server in background
ollama serve &

# Wait for Ollama to fully start
sleep 5

# Start your Streamlit app
streamlit run /app/emne.py --server.address=0.0.0.0 --server.port=8501
