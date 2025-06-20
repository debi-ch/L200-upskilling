"""
Gemini Model Integration

This module provides integration with Google's Gemini models,
including both base and fine-tuned versions.
"""

from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content

# Import configuration and logging
from app.config import GCP_PROJECT_ID, GCP_LOCATION, ModelConfig
from app.utils.logging_utils import ChatbotLogger
from app.backend.models.weather_chat import get_location_weather

# Initialize logger
logger = ChatbotLogger("gemini_model")

# Flag to control which model to use
USE_TUNED_MODEL = ModelConfig.USE_TUNED_MODEL

def chat_with_gemini(prompt_text, tools=None):
    """
    Sends a prompt to the selected Gemini model and gets a response.
    
    Args:
        prompt_text (str): The prompt text to send to the model
        tools (list, optional): A list of tools the model can use. Defaults to None.
        
    Returns:
        str: The model's response text
    """
    if USE_TUNED_MODEL:
        # Try the fine-tuned model (gemini-assistant-custom)
        return chat_with_model_internal(
            ModelConfig.GEMINI_TUNED_MODEL_NAME, 
            ModelConfig.GEMINI_TUNED_MODEL_PROJECT_NUM, 
            ModelConfig.GEMINI_TUNED_MODEL_REGION, 
            prompt_text, 
            "fine-tuned",
            tools=tools
        )
    else:
        # Use the base model
        return chat_with_model_internal(
            ModelConfig.GEMINI_BASE_MODEL_ID, 
            GCP_PROJECT_ID, 
            GCP_LOCATION, 
            prompt_text, 
            "base",
            tools=tools
        )

def chat_with_model_internal(model_id, project_id, location, prompt_text, model_type_str, tools=None):
    """
    Internal function to call a specific Gemini model, with optional function calling.
    
    Args:
        model_id (str): The model ID or resource name
        project_id (str): The Google Cloud project ID
        location (str): The Google Cloud region
        prompt_text (str): The prompt text to send to the model
        model_type_str (str): A string describing the model type (for logging)
        tools (list, optional): A list of tools the model can use. Defaults to None.
        
    Returns:
        str: The model's final response text
    """
    try:
        logger.info(
            f"Using {model_type_str} model",
            model_id=model_id,
            project_id=project_id,
            location=location
        )
        
        # Initialize Vertex AI (idempotent)
        vertexai.init(project=project_id, location=location)
        
        # Load the model and start a chat session
        model = GenerativeModel(model_name=model_id)
        chat = model.start_chat()

        # Send the prompt and tools to the model
        logger.debug("Sending prompt to model with tools", prompt=prompt_text, tools_available=str(tools) if tools else "None")
        response = chat.send_message(prompt_text, tools=tools)
        
        # Check if the model wants to call a function
        function_call = response.candidates[0].content.parts[0].function_call
        if not function_call:
            # If no function call, return the text response
            logger.info("Model returned a direct text response.")
            return response.text

        # --- Handle Function Call ---
        logger.info("Model requested a function call", function_name=function_call.name, args=str(function_call.args))
        
        # Call the requested function
        if function_call.name == "get_location_weather":
            location_arg = function_call.args.get("location")
            api_response = get_location_weather(location=location_arg)
            
            # Convert dict to a string or a Part object for the model
            # This ensures the model gets a clean, usable response.
            part = Part.from_function_response(
                name=function_call.name,
                response={
                    "content": str(api_response),
                }
            )

            # Send the function response back to the model
            logger.info("Sending function response back to the model", function_response=str(api_response))
            final_response = chat.send_message(part)
            
            return final_response.text
        else:
            logger.warning(f"Model requested an unknown function: {function_call.name}")
            return f"Error: The model requested an unknown function: {function_call.name}"

            
    except Exception as e:
        logger.error(
            f"Error during prediction with {model_type_str} model",
            error_type=type(e).__name__,
            error_message=str(e)
        )
        
        # If the fine-tuned model failed, try falling back to the base model
        if model_type_str == "fine-tuned":
            logger.warning("Falling back to base model due to error")
            set_model_preference(False)  # Change the preference
            # Call the base model directly, avoiding recursion into chat_with_gemini
            return chat_with_model_internal(
                ModelConfig.GEMINI_BASE_MODEL_ID, 
                GCP_PROJECT_ID, 
                GCP_LOCATION, 
                prompt_text, 
                "base (fallback)",
                tools=tools
            )
        else:
            # If the base model (or fallback) also failed
            return f"An error occurred with the {model_type_str} model: {str(e)}"

def set_model_preference(use_tuned_model=True):
    """
    Set whether to use the tuned model or base model.
    
    Args:
        use_tuned_model (bool): True to use the tuned model, False to use the base model
        
    Returns:
        bool: The new model preference
    """
    global USE_TUNED_MODEL
    USE_TUNED_MODEL = use_tuned_model
    model_name = "Tuned Model (gemini-assistant-custom)" if USE_TUNED_MODEL else "Base Model"
    logger.info(f"Model preference set to: {model_name}")
    return USE_TUNED_MODEL

def get_current_model_name():
    """
    Returns the name of the currently active model.
    
    Returns:
        str: A descriptive name of the current model
    """
    if USE_TUNED_MODEL:
        return "Gemini (Fine-tuned: gemini-assistant-custom)"
    else:
        return "Gemini (Base)"

# Test function to run when this module is executed directly
if __name__ == "__main__":
    print("Welcome to the Gemini Chat!")
    print("Type 'exit' to quit.")
    print("Type 'switch' to toggle between base and fine-tuned models.")
    
    print(f"Currently using: {get_current_model_name()}")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        elif user_input.lower() == "switch":
            # Use the function to set the preference
            set_model_preference(not USE_TUNED_MODEL)
            print(f"Switched to: {get_current_model_name()}")
            continue

        print(f"{get_current_model_name()} is thinking...")
        response = chat_with_gemini(user_input)
        print(f"Response: {response}")
        print("-" * 40) 