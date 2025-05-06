import os
import json
import requests

# Configuration
API_KEY = os.environ.get("GEMINI_API_KEY", "")  # Gemini API key from Google AI Studio
if not API_KEY:
    print("⚠️ No API key found. Please set the GEMINI_API_KEY environment variable.")
    print("Visit https://ai.google.dev/ to get a key.")
    print("Then run: export GEMINI_API_KEY='your-key-here'")

def gemini_generate_content(prompt, model="gemini-1.5-pro"):
    """Generate content using the Gemini API with an API key."""
    if not API_KEY:
        print("Error: API key is required. Please set GEMINI_API_KEY environment variable.")
        return None
    
    # Gemini API URL format
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={API_KEY}"
    
    # Request headers
    headers = {
        "Content-Type": "application/json"
    }
    
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
    
    print(f"Sending request to: {url.split('?')[0]}")  # Don't show the key in logs
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
    if not API_KEY:
        print("Error: API key is required. Please set GEMINI_API_KEY environment variable.")
        return None
    
    # Gemini API URL for listing models
    url = f"https://generativelanguage.googleapis.com/v1/models?key={API_KEY}"
    
    print(f"Fetching Gemini models from API...")
    
    # Make API request
    response = requests.get(url)
    
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

def try_with_fine_tuned_model(prompt, model_id):
    """Try to generate content with the fine-tuned model through Gemini API."""
    if not API_KEY:
        print("Error: API key is required. Please set GEMINI_API_KEY environment variable.")
        return None
    
    # First, try using the model ID directly
    response = gemini_generate_content(prompt, model=model_id)
    if response:
        return response
    
    # If that doesn't work, try other approaches
    # Note: This is a placeholder for future work with fine-tuned models
    # via the Gemini API once they support it
    return None

if __name__ == "__main__":
    print("=== Gemini API Access with API Key ===")
    
    if not API_KEY:
        print("⚠️ No API key found. Please set the GEMINI_API_KEY environment variable.")
        print("Visit https://ai.google.dev/ to get a key.")
        exit(1)
    
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
        print("Please check your API key or try again later.") 