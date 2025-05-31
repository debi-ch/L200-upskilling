#!/bin/bash

# Exit on error
set -e

# Configuration
PROJECT_ID="learningemini"  # Your GCP project ID
REGION="us-central1"        # Your preferred region
SERVICE_NAME="rag-chatbot"  # Your service name

# Build and deploy using Cloud Build
echo "Building and deploying using Cloud Build..."
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --project ${PROJECT_ID} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars="CHATBOT_ENV=production,USE_TUNED_MODEL=true"

echo "Deployment complete! Your service will be available at the URL above." 