import os
import argparse
import time
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import aiplatform

# Set environment variables
PROJECT_ID = "learningemini"
LOCATION = "us-central1"

def init_vertex_ai():
    """Initialize Vertex AI"""
    print(f"Initializing Vertex AI with project: {PROJECT_ID}, location: {LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

def deploy_model_to_endpoint(model_name, endpoint_name=None):
    """Deploy the model to an endpoint"""
    if endpoint_name is None:
        endpoint_name = f"gemini-tuned-endpoint-{int(time.time())}"
    
    print(f"Deploying model {model_name} to endpoint {endpoint_name}...")
    
    # Get model ID from the full model name
    model_id = model_name.split('/')[-1] if '/' in model_name else model_name
    
    # Create an endpoint
    try:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
        print(f"Created endpoint: {endpoint.name}")
        
        # Deploy the model to the endpoint
        model = aiplatform.Model(model_name=model_name)
        model.deploy(
            endpoint=endpoint,
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=1,
            deploy_request_timeout=1800,
            sync=True
        )
        
        print(f"Model successfully deployed to endpoint {endpoint.name}")
        print(f"Endpoint URI: {endpoint.resource_name}")
        
        return endpoint.resource_name
    
    except Exception as e:
        print(f"Error deploying model: {e}")
        return None

def test_endpoint(endpoint_resource_name, prompt="What's the most interesting thing about artificial intelligence?"):
    """Test the deployed endpoint"""
    print(f"Testing endpoint {endpoint_resource_name} with prompt: '{prompt}'")
    
    try:
        # Create a prediction client
        client = aiplatform.prediction.PredictionServiceClient(client_options={
            "api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"
        })
        
        # Format the request
        instance = {
            "prompt": prompt
        }
        
        # Get a prediction
        response = client.predict(
            endpoint=endpoint_resource_name,
            instances=[instance],
            parameters={"temperature": 0.2, "maxOutputTokens": 1024}
        )
        
        print("\nEndpoint response:")
        print("="*80)
        print(response)
        print("="*80)
        
        return response
    
    except Exception as e:
        print(f"Error testing endpoint: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Deploy a fine-tuned Gemini model to an endpoint")
    parser.add_argument("--model_name", required=True, type=str, 
                        help="The name of the fine-tuned model to deploy")
    parser.add_argument("--endpoint_name", type=str, default=None,
                        help="Custom name for the endpoint (optional)")
    args = parser.parse_args()
    
    # Initialize Vertex AI
    init_vertex_ai()
    
    # Deploy the model
    endpoint_name = deploy_model_to_endpoint(args.model_name, args.endpoint_name)
    
    if endpoint_name:
        # Test the endpoint
        if input("\nWould you like to test the endpoint? (y/n): ").lower() == 'y':
            test_prompt = input("Enter a test prompt (or press Enter for default): ")
            if not test_prompt:
                test_prompt = "What's the most interesting thing about artificial intelligence?"
            
            test_endpoint(endpoint_name, test_prompt)
        
        print(f"\nEndpoint deployment complete. You can now use this endpoint in your application.")
        print(f"To use this endpoint in code, use the following endpoint name:")
        print(f"{endpoint_name}")
    else:
        print("\nEndpoint deployment failed. Check the logs for details.")

def generate_integration_code(endpoint_name):
    """Generate code to use the endpoint in an application"""
    print("\nTo use the deployed model endpoint in your application:")
    print("-"*80)
    print(f"""from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project="{PROJECT_ID}", location="{LOCATION}")

# Create a prediction client
endpoint = aiplatform.Endpoint("{endpoint_name}")

# Function to generate response
def generate_response(prompt):
    response = endpoint.predict(
        instances=[{{"prompt": prompt}}],
        parameters={{"temperature": 0.2, "maxOutputTokens": 1024}}
    )
    return response.predictions[0]
    
# Example usage
response = generate_response("Your prompt here")
print(response)
""")
    print("-"*80)

if __name__ == "__main__":
    main()
    
    print("\nAfter deployment, you can integrate this endpoint with your application.")
    endpoint_name = input("Enter the endpoint name to generate integration code (or press Enter to skip): ")
    if endpoint_name:
        generate_integration_code(endpoint_name) 