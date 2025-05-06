import os
import json
import requests
from google.auth.transport.requests import Request
import google.auth

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "learningemini")
LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL_ID = os.environ.get("MODEL_ID", "2279310694123831296")  # gemini-assistant-custom model
ENDPOINT_ID = os.environ.get("ENDPOINT_ID", "4660193172909981696")  # endpoint ID

def get_access_token():
    """Get access token for API requests."""
    try:
        # Make sure to get fresh credentials
        credentials, _ = google.auth.default(quota_project_id=PROJECT_ID)
        auth_req = Request()
        credentials.refresh(auth_req)
        return credentials.token
    except Exception as e:
        print(f"Error getting access token: {e}")
        return None

def gemini_generate_content(prompt, model="gemini-1.5-pro"):
    """Generate content using the Gemini API directly."""
    access_token = get_access_token()
    if not access_token:
        print("Failed to get access token. Check your authentication.")
        return None
    
    # Gemini API URL format
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
    
    # Add API key as a query parameter if it exists
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        url += f"?key={api_key}"
    
    # Request headers with OAuth token if no API key is provided
    headers = {}
    if not api_key:
        headers["Authorization"] = f"Bearer {access_token}"
    
    headers["Content-Type"] = "application/json"
    
    # Request body for Gemini API
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
            "topK": 40,
            "topP": 0.95
        }
    }
    
    print(f"Sending request to: {url}")
    print(f"Request payload: {json.dumps(data, indent=2)}")
    
    # Make API request
    response = requests.post(url, headers=headers, json=data)
    
    # Handle response
    if response.status_code == 200:
        print("✅ API call successful!")
        return response.json()
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

def get_gemini_models():
    """List available Gemini models."""
    access_token = get_access_token()
    if not access_token:
        print("Failed to get access token. Check your authentication.")
        return None
    
    # Gemini API URL for listing models
    url = "https://generativelanguage.googleapis.com/v1/models"
    
    # Add API key as a query parameter if it exists
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        url += f"?key={api_key}"
    
    # Request headers with OAuth token if no API key is provided
    headers = {}
    if not api_key:
        headers["Authorization"] = f"Bearer {access_token}"
    
    print(f"Fetching Gemini models from: {url}")
    
    # Make API request
    response = requests.get(url, headers=headers)
    
    # Handle response
    if response.status_code == 200:
        models = response.json().get("models", [])
        print(f"Found {len(models)} Gemini models")
        for model in models:
            print(f"- {model.get('name')} (Display Name: {model.get('displayName')})")
        return models
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

def extract_text_from_response(response):
    """Extract the generated text from the Gemini API response."""
    if not response:
        return None
        
    try:
        candidates = response.get("candidates", [])
        if candidates and len(candidates) > 0:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and len(parts) > 0:
                return parts[0].get("text", "")
    except (KeyError, IndexError) as e:
        print(f"Error extracting text: {e}")
    
    print("Could not parse response:")
    print(json.dumps(response, indent=2))
    return None

if __name__ == "__main__":
    print("=== Gemini API Direct Access ===")
    print(f"Using Project ID: {PROJECT_ID}")
    print(f"Using Location: {LOCATION}")
    
    # First, verify authentication works
    token = get_access_token()
    if not token:
        print("Authentication failed. Please check your Google Cloud credentials.")
        exit(1)
    else:
        print("Authentication successful!")
    
    # List available Gemini models
    print("\nListing available Gemini models...")
    models = get_gemini_models()
    
    # Try different models
    print("\nTrying multiple Gemini models...")
    gemini_models = [
        "gemini-1.0-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro",  # Alias for 1.0-pro in some regions
    ]
    
    prompt = "What is machine learning?"
    
    for model in gemini_models:
        print(f"\n=== Trying model: {model} ===")
        response = gemini_generate_content(prompt, model)
        
        if response:
            text = extract_text_from_response(response)
            if text:
                print("\nGenerated Text:")
                print(text)
                print("\nSuccess! Found a working model.")
                break
    else:
        print("\nFailed to generate a response from any Gemini model.")
        
        # Last resort - get an API key
        if not os.environ.get("GEMINI_API_KEY"):
            print("\nTip: You may need to get a Gemini API key.")
            print("Visit: https://ai.google.dev/ to get a key.")
            print("Then set it as an environment variable: export GEMINI_API_KEY='your-key-here'") 