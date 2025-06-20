"""
Configuration settings for the application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', 'AIzaSyBPYcSejsvuaM4dEZpexIeq3eOVUiPKry4')

# Weather API Settings
# WEATHER_API_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Default location settings
DEFAULT_LOCATION = {
    'latitude': 40.7128,  # New York City
    'longitude': -74.0060
}

def validate_config():
    """Validate that required configuration is present."""
    missing_keys = []
    
    if not GOOGLE_MAPS_API_KEY:
        missing_keys.append('GOOGLE_MAPS_API_KEY')
        print("⚠️ Missing required configuration:")
        for key in missing_keys:
            print(f"- {key}")
        print("\nPlease set these environment variables.")
        return False
    
    return True 

# Set the API key in environment if not already set
if 'GOOGLE_MAPS_API_KEY' not in os.environ:
    os.environ['GOOGLE_MAPS_API_KEY'] = GOOGLE_MAPS_API_KEY

# For any other services that require API keys
# Example:
# OTHER_SERVICE_API_KEY = os.getenv("OTHER_SERVICE_API_KEY")

# Weather API settings (if needed)
# Example:
# WEATHER_API_BASE_URL = "https://api.weatherapi.com/v1"
# WEATHER_API_KEY = os.getenv("WEATHER_API_KEY") 