#!/bin/bash
# Script to run the chatbot app with the new directory structure

# Add the current directory to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the streamlit app
streamlit run app/frontend/chatbot_app.py
