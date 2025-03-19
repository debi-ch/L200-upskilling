# Chatbot Project Status

Current Status:
- Chatbot deployed on Cloud Run
- Using Gemma model
- Using Cloud Build for deployments
- Using Artifact Registry for container images

Next Steps:
- Add Streamlit dropdown to switch between Gemma and Gemini
- Compare model performances
- Add more features

To get started again:
1. Activate virtual environment: 
   ```bash
   source venv/bin/activate
   ```

2. Run app locally: 
   ```bash
   streamlit run chatbot_app.py
   ```

3. Deploy changes using Cloud Build: 
   ```bash
   # Submit build to Cloud Build
   gcloud builds submit --tag us-west1-docker.pkg.dev/learningemini/chatbot/chatbot-app

   # Deploy to Cloud Run
   gcloud run deploy chatbot-app \
     --image us-west1-docker.pkg.dev/learningemini/chatbot/chatbot-app \
     --platform managed \
     --region us-west1 \
     --allow-unauthenticated
   ```

Local Development:
- All code is in L200-upskilling directory
- Using virtual environment for Python dependencies
- Streamlit for web interface
- Firestore for chat history
- Cloud Logging for monitoring

Features:
- Web-based chat interface
- Persistent chat history
- Cloud logging and monitoring
- Session management
- Model switching capability (coming soon)

Dependencies:
- google-cloud-aiplatform
- google-cloud-logging>=3.8.0
- google-cloud-firestore
- streamlit
