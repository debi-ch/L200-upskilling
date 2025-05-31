FROM python:3.9-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

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
ENV PORT=8080

# Create necessary directories
RUN mkdir -p logs
RUN mkdir -p data/embeddings
RUN mkdir -p documents/pdfs
RUN mkdir -p documents/images

# Create a non-root user to run the app
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose the port
EXPOSE ${PORT}

# Run the app - use the PORT env variable
CMD streamlit run app/frontend/chatbot_app.py --server.port=${PORT} --server.address=0.0.0.0 