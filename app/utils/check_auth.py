import os
import json
import google.auth
import google.auth.transport.requests
from google.cloud import storage
import importlib

def check_authentication():
    """Check if Google Cloud authentication is working correctly."""
    try:
        # Get credentials and project ID
        credentials, project_id = google.auth.default()
        
        # Check if we got a project ID
        if not project_id:
            project_id = os.environ.get("PROJECT_ID", "learningemini")
            print(f"No default project ID found, using environment variable: {project_id}")
        else:
            print(f"Successfully authenticated with project ID: {project_id}")
        
        # Refresh the credentials
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        
        # Check if we got a valid token
        if credentials.token:
            print("✅ Successfully obtained access token")
            return True
        else:
            print("❌ Failed to obtain access token")
            return False
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return False

def check_storage_access():
    """Check if we can access Google Cloud Storage."""
    try:
        # Create a storage client
        storage_client = storage.Client()
        
        # List buckets (up to 10)
        buckets = list(storage_client.list_buckets(max_results=10))
        
        print(f"✅ Successfully accessed Google Cloud Storage. Found {len(buckets)} buckets.")
        if buckets:
            print("Buckets:")
            for bucket in buckets:
                print(f"  - {bucket.name}")
        return True
    except Exception as e:
        print(f"❌ Google Cloud Storage access error: {e}")
        return False

def check_vertex_ai_access():
    """Check if we can access Vertex AI API."""
    try:
        # Import dynamically to handle module not found errors gracefully
        try:
            from google.cloud import aiplatform
        except ImportError:
            print("❌ Could not import google.cloud.aiplatform. Running pip install...")
            import subprocess
            subprocess.run(["pip", "install", "google-cloud-aiplatform"], check=True)
            from google.cloud import aiplatform
        
        # Get current version
        try:
            import google.cloud.aiplatform
            version = importlib.metadata.version("google-cloud-aiplatform")
            print(f"Using google-cloud-aiplatform version: {version}")
        except:
            print("Could not determine google-cloud-aiplatform version")
        
        # Get project ID
        _, project_id = google.auth.default()
        if not project_id:
            project_id = os.environ.get("PROJECT_ID", "learningemini")
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location="us-central1")
        
        # Try to list models - handle different API versions
        print("Attempting to list Vertex AI models...")
        try:
            # Try new API (with max_results)
            models = aiplatform.Model.list(max_results=10)
        except TypeError as e:
            if "got an unexpected keyword argument 'max_results'" in str(e):
                # Try older API version (without max_results)
                print("Using older API format (without max_results)...")
                models = aiplatform.Model.list()
            else:
                raise
        
        # Print model info
        if models:
            print(f"✅ Successfully accessed Vertex AI API. Found {len(models)} models.")
            print("Models:")
            for model in models:
                try:
                    print(f"  - {model.display_name} (ID: {model.name})")
                except AttributeError:
                    # Handle different model object formats
                    print(f"  - {model}")
        else:
            print("✅ Successfully accessed Vertex AI API. No models found.")
        
        return True
    except Exception as e:
        print(f"❌ Vertex AI API access error: {e}")
        print("This may indicate that the Vertex AI API is not enabled for your project")
        print("or there's an issue with your installation of the google-cloud-aiplatform package.")
        print("Enable API at: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com")
        print("To reinstall the package, run: pip install -U google-cloud-aiplatform")
        return False

def check_specific_model_access():
    """Check if we can access the specific fine-tuned model."""
    try:
        # Import necessary libraries
        import requests
        
        # Get project ID and credentials
        credentials, project_id = google.auth.default()
        if not project_id:
            project_id = os.environ.get("PROJECT_ID", "learningemini")
        
        model_id = os.environ.get("MODEL_ID", "2279310694123831296")  # gemini-assistant-custom model
        location = os.environ.get("LOCATION", "us-central1")
        
        print(f"Attempting to access model ID: {model_id}")
        
        # Get token for API requests
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        token = credentials.token
        
        # Try different URL formats
        urls = [
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/models/{model_id}",
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/tuningJobs/{model_id}",
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}"
        ]
        
        for url in urls:
            print(f"Trying URL: {url}")
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                print(f"✅ Successfully accessed model at: {url}")
                model_info = response.json()
                # Print only key details to avoid cluttering the output
                print(f"Model details: Name: {model_info.get('name', 'N/A')}, Display Name: {model_info.get('displayName', 'N/A')}")
                if 'createTime' in model_info:
                    print(f"Create Time: {model_info.get('createTime', 'N/A')}")
                return True
            else:
                print(f"❌ Could not access model at {url}: {response.status_code}")
                print(f"Error: {response.text}")
        
        print("❌ Could not access the model with any of the URL formats")
        return False
    except Exception as e:
        print(f"❌ Error accessing the specific model: {e}")
        return False

if __name__ == "__main__":
    import importlib.metadata
    
    # Print versions for debugging
    try:
        version = importlib.metadata.version("google-cloud-aiplatform")
        print(f"google-cloud-aiplatform version: {version}")
    except:
        print("Could not determine google-cloud-aiplatform version")
    
    print("==== Google Cloud Authentication Check ====")
    auth_ok = check_authentication()
    
    if auth_ok:
        print("\n==== Google Cloud Storage Access Check ====")
        storage_ok = check_storage_access()
        
        print("\n==== Vertex AI API Access Check ====")
        vertex_ai_ok = check_vertex_ai_access()
        
        print("\n==== Fine-tuned Model Access Check ====")
        model_ok = check_specific_model_access()
        
        if auth_ok and storage_ok and vertex_ai_ok and model_ok:
            print("\n✅ All checks passed. Your environment is correctly set up for Vertex AI.")
        else:
            print("\n❌ Some checks failed. Please address the issues above.")
    else:
        print("\n❌ Authentication failed. Please fix authentication before continuing.")