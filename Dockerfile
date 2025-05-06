FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV CHATBOT_ENV=production
ENV GCP_PROJECT_ID=learningemini
ENV USE_TUNED_MODEL=true

# Create logs directory
RUN mkdir -p logs

# Create a non-root user to run the app
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose the Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app/frontend/chatbot_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 