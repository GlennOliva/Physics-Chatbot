#!/bin/bash

# Start Ollama server in background
ollama serve &

# Wait for Ollama to fully start
sleep 5

# Start your Streamlit app
streamlit run /app/app.py --server.address=0.0.0.0 --server.port=8501
