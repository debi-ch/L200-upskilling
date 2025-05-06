import os
import json
import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.auth
import google.auth.transport.requests

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
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials.token
    except Exception as e:
        print(f"Error getting access token: {e}")
        return None

# Function to generate content from fine-tuned model
def generate_content(prompt, temperature=0.2, max_output_tokens=1024):
    """Generate content using Vertex AI REST API."""
    access_token = get_access_token()
    if not access_token:
        print("Failed to get access token. Check your authentication.")
        return None
    
    # Try different URL formats
    urls = [
        # Format for endpoint
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict",
        # Format for custom fine-tuned models
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/models/{MODEL_ID}:predict",
        # Format for Gemini models
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/gemini-1.0-pro:generateContent"
    ]
    
    # Request headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Request body - try different payload formats
    payloads = [
        # Format 1: Endpoint format - this is standard for Vertex AI endpoints
        {
            "instances": [
                {
                    "content": prompt
                }
            ]
        },
        # Format 2: Standard for Vertex AI custom models
        {
            "instances": [
                {
                    "prompt": prompt
                }
            ],
            "parameters": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topK": 40,
                "topP": 0.95
            }
        },
        # Format 3: Gemini format
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topK": 40,
                "topP": 0.95
            }
        }
    ]
    
    # Try each URL with each payload format
    success = False
    for url in urls:
        for payload in payloads:
            print(f"\nTrying URL: {url}")
            print(f"Request payload: {json.dumps(payload, indent=2)}")
            
            # Make API request
            response = requests.post(url, headers=headers, json=payload)
            
            # Handle response
            if response.status_code == 200:
                print("✅ API call successful!")
                success = True
                return response.json()
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
    
    if not success:
        print("All API call attempts failed.")
        # Try with specific model and endpoint combinations
        print("\nTrying specific model access patterns...")
        
        # Try endpoint with specific payload
        url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"
        payload = {
            "instances": [
                {
                    "content": prompt
                }
            ]
        }
        print(f"Request payload: {json.dumps(payload, indent=2)}")
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print("✅ Specific endpoint call successful!")
            return response.json()
        else:
            print(f"❌ Specific endpoint error: {response.status_code}")
            print(response.text)
            return None

# List available fine-tuned models
def list_models():
    """List available fine-tuned models."""
    access_token = get_access_token()
    if not access_token:
        print("Failed to get access token. Check your authentication.")
        return None
    
    # Try both endpoints for listing models
    urls = [
        # Custom models in your project
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/models",
        # Alternatively try to list tuned models specifically
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/tuningJobs",
        # Also try endpoints
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints"
    ]
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    models = []
    endpoints = []
    tuning_jobs = []
    
    for url in urls:
        print(f"\nTrying to list models with URL: {url}")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            if "models" in response.json():
                current_models = response.json().get("models", [])
                print(f"Found {len(current_models)} models with this endpoint")
                models.extend(current_models)
            elif "tuningJobs" in response.json():
                jobs = response.json().get("tuningJobs", [])
                print(f"Found {len(jobs)} tuning jobs")
                tuning_jobs.extend(jobs)
                for job in jobs:
                    print(f"- {job.get('displayName')} (Status: {job.get('state')})")
            elif "endpoints" in response.json():
                current_endpoints = response.json().get("endpoints", [])
                endpoints.extend(current_endpoints)
                print(f"Found {len(current_endpoints)} endpoints")
                for endpoint in current_endpoints:
                    endpoint_id = endpoint.get("name", "").split("/")[-1]
                    print(f"- {endpoint.get('displayName')} (ID: {endpoint_id})")
                    
                    # Check if the endpoint has a deployedModels field
                    if "deployedModels" in endpoint:
                        deployed_models = endpoint.get("deployedModels", [])
                        for dm in deployed_models:
                            model_id = dm.get("model", "").split("/")[-1]
                            print(f"  - Deployed Model: {model_id}")
            else:
                print(f"Endpoint returned data but no models or tuning jobs found")
                print(f"Response keys: {response.json().keys()}")
        else:
            print(f"Error accessing {url}: {response.status_code}")
            print(response.text)
    
    if models:
        print(f"\nFound a total of {len(models)} models:")
        for model in models:
            model_id = model.get("name", "").split("/")[-1]
            print(f"- {model.get('displayName')} (ID: {model_id})")
        
    if endpoints:
        print(f"\nFound a total of {len(endpoints)} endpoints:")
        for endpoint in endpoints:
            endpoint_id = endpoint.get("name", "").split("/")[-1]
            print(f"- {endpoint.get('displayName')} (ID: {endpoint_id})")
    
    return models

# Example usage
if __name__ == "__main__":
    print(f"Using Project ID: {PROJECT_ID}")
    print(f"Using Location: {LOCATION}")
    print(f"Using Model ID: {MODEL_ID}")
    print(f"Using Endpoint ID: {ENDPOINT_ID}")
    
    # First, verify authentication works
    token = get_access_token()
    if not token:
        print("Authentication failed. Please check your Google Cloud credentials.")
        exit(1)
    else:
        print("Authentication successful!")
    
    # List available models to find your fine-tuned model
    print("\nListing available models...")
    models = list_models()
    
    # Example prompt
    prompt = "What is machine learning?"
    
    print(f"\nGenerating response for prompt: '{prompt}'")
    response = generate_content(prompt)
    
    if response:
        print("\nAPI Response:")
        print(json.dumps(response, indent=2))
        
        # Extract the generated text (format depends on model response structure)
        try:
            # Try to extract text - format may vary depending on model response
            if "predictions" in response:
                predictions = response.get("predictions", [])
                if predictions and len(predictions) > 0:
                    if "content" in predictions[0]:
                        generated_text = predictions[0].get("content", "")
                    else:
                        generated_text = str(predictions[0])
                    print("\nGenerated Text:")
                    print(generated_text)
            elif "content" in response:
                generated_text = response.get("content", "")
                print("\nGenerated Text:")
                print(generated_text)
            elif "candidates" in response:
                candidates = response.get("candidates", [])
                if candidates and len(candidates) > 0:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts and len(parts) > 0:
                        generated_text = parts[0].get("text", "")
                        print("\nGenerated Text:")
                        print(generated_text)
            else:
                print("\nResponse format different than expected. Full response:")
                print(json.dumps(response, indent=2))
        except (KeyError, IndexError) as e:
            print(f"Could not extract generated text: {e}")
            print("Check the full response structure above.") 