#!/bin/bash

# Set environment variables
export PROJECT_ID="learningemini"  # Your GCP project ID
export LOCATION="us-central1"      # Your region
export MODEL_ID="2279310694123831296"  # gemini-assistant-custom model
export ENDPOINT_ID="4660193172909981696"  # The endpoint ID for the model

# Verify authentication and API access first
echo "Checking authentication and API access..."
python3 check_auth.py

if [ $? -ne 0 ]; then
  echo "Authentication check failed. Please fix the issues before continuing."
  exit 1
fi

# Run the API script
echo "Running Gemini REST API script..."
python3 gemini_rest_api.py 