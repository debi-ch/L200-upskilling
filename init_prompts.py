from prompt_manager import PromptManager

def initialize_prompts():
    manager = PromptManager()
    
    # Only add prompts if they don't exist for the model
    if not manager.get_prompt_versions("gemini"):
        # Add Gemini prompt
        manager.add_prompt_version(
            "gemini",
            "You are an advanced AI assistant powered by Google's Gemini model. "
            "You excel at providing detailed, accurate, and helpful responses. "
            "Please analyze queries thoroughly and provide comprehensive answers.",
            "Initial Gemini prompt"
        )
    
    if not manager.get_prompt_versions("gemma"):
        # Add Gemma prompt
        manager.add_prompt_version(
            "gemma",
            "You are an AI assistant powered by Google's Gemma model. "
            "You are focused on providing clear, concise, and accurate responses. "
            "Please maintain a helpful and informative tone while being direct and efficient.",
            "Initial Gemma prompt"
        )
    
    print("Prompts initialized successfully!")

if __name__ == "__main__":
    initialize_prompts() 