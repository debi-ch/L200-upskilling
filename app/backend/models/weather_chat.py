"""
Weather Integration Module

This module provides weather-related functionality without ADK dependency,
following the same pattern as other chat modules.
"""

import os
import requests
from datetime import datetime
from typing import Dict, Any
from ..tools.geocoding import get_coordinates
from app.utils.logging_utils import ChatbotLogger
from vertexai.generative_models import Tool, FunctionDeclaration

# Initialize logger
logger = ChatbotLogger("weather_model")

# Define the get_location_weather function as a tool for the Gemini model
get_weather_tool = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="get_location_weather",
            description="Get the current weather and a 7-day forecast for a specific location.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, or city and country, e.g., San Francisco, CA or Florence, Italy"
                    }
                },
                "required": ["location"]
            },
        )
    ],
)

def get_weather_forecast(latitude: float, longitude: float, timezone: str = "auto") -> Dict[str, Any]:
    """
    Retrieves weather forecast from Google Weather API.
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        timezone (str): Timezone for weather data (defaults to "auto")
        
    Returns:
        dict: Weather forecast data including current conditions and daily forecast
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "error_message": "Google Maps API key not configured. Please set GOOGLE_MAPS_API_KEY environment variable."
        }
    
    try:
        # Get current conditions
        current_url = "https://weather.googleapis.com/v1/currentConditions:lookup"
        current_params = {
            "key": api_key,
            "location.latitude": latitude,
            "location.longitude": longitude
        }
        
        logger.info(f"Requesting CURRENT conditions from URL: {current_url}")
        logger.info(f"With PARAMS: {current_params}")

        current_response = requests.get(current_url, params=current_params)
        logger.info(f"CURRENT conditions response status: {current_response.status_code}")
        logger.info(f"CURRENT conditions response text: {current_response.text}")
        current_response.raise_for_status()
        current_data = current_response.json()

        # Get daily forecast
        forecast_url = "https://weather.googleapis.com/v1/forecast/days:lookup"
        forecast_params = {
            "key": api_key,
            "location.latitude": latitude,
            "location.longitude": longitude,
            "days": 7  # Get 7 days forecast
        }
        
        logger.info(f"Requesting FORECAST from URL: {forecast_url}")
        forecast_response = requests.get(forecast_url, params=forecast_params)
        logger.info(f"FORECAST response status: {forecast_response.status_code}")
        logger.info(f"FORECAST response text: {forecast_response.text}")
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
        
        # Process current weather
        temp_c = current_data.get("temperature", {}).get("degrees")
        if temp_c is None:
            return {
                "status": "error",
                "error_message": "Temperature data is missing from the weather response"
            }
            
        current_weather = {
            "temperature": temp_c,
            "humidity": current_data.get("relativeHumidity"),
            "wind_speed": current_data.get("wind", {}).get("speed", {}).get("value"),
            "conditions": current_data.get("weatherCondition", {}).get("description", {}).get("text", "Unknown")
        }
        
        # Process daily forecast
        daily_forecast = []
        for day in forecast_data.get("forecastDays", []):
            date_dict = day.get("displayDate", {})
            date_str = f"{date_dict.get('year')}-{date_dict.get('month'):02d}-{date_dict.get('day'):02d}"
            
            daytime_forecast = day.get("daytimeForecast", {})

            day_data = {
                "date": date_str,
                "max_temp": day.get("maxTemperature", {}).get("degrees"),
                "min_temp": day.get("minTemperature", {}).get("degrees"),
                "precipitation_prob": daytime_forecast.get("precipitation", {}).get("probability", {}).get("percent"),
                "wind_speed": daytime_forecast.get("wind", {}).get("speed", {}).get("value")
            }
            daily_forecast.append(day_data)
        
        return {
            "status": "success",
            "current": current_weather,
            "forecast": daily_forecast
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(
            "Failed to fetch weather data",
            error_type=type(e).__name__,
            error_message=str(e)
        )
        return {
            "status": "error",
            "error_message": f"Failed to fetch weather data: {str(e)}"
        }

def get_location_weather(location: str) -> Dict[str, Any]:
    """
    Get weather forecast for a location by name.
    
    Args:
        location (str): Name of the location (e.g., "Florence, Italy")
        
    Returns:
        dict: Combined location and weather data
    """
    try:
        # First get coordinates
        logger.info(f"Getting coordinates for location: {location}")
        location_data = get_coordinates(location)
        if location_data["status"] != "success":
            logger.error(
                "Failed to get coordinates",
                error_message=location_data.get("error_message", "Unknown error"),
                location=location
            )
            return location_data
        
        # Then get weather data
        logger.info(f"Getting weather data for coordinates: {location_data['latitude']}, {location_data['longitude']}")
        weather_data = get_weather_forecast(
            latitude=location_data["latitude"],
            longitude=location_data["longitude"]
        )
        
        if weather_data["status"] != "success":
            logger.error(
                "Failed to get weather data",
                error_message=weather_data.get("error_message", "Unknown error"),
                location=location,
                coordinates={"lat": location_data["latitude"], "lon": location_data["longitude"]}
            )
            return weather_data
        
        # Format temperature for display
        current = weather_data["current"]
        temp_c = current.get("temperature")
        
        if temp_c is None:
            logger.error(
                "Temperature data is missing",
                location=location,
                coordinates={"lat": location_data["latitude"], "lon": location_data["longitude"]}
            )
            return {
                "status": "error",
                "error_message": "Temperature data is missing from the weather response"
            }
            
        temp_f = (temp_c * 9/5) + 32
        
        # Create a more user-friendly response
        return {
            "status": "success",
            "location": location_data["formatted_address"],
            "current": {
                "temperature": f"{temp_c:.1f}°C ({temp_f:.1f}°F)",
                "conditions": current["conditions"],
                "humidity": f"{current['humidity']}%",
                "wind_speed": f"{current['wind_speed']} km/h"
            },
            "forecast": weather_data["forecast"]
        }
    except Exception as e:
        import traceback
        logger.error(
            "Error processing weather request",
            error_type=type(e).__name__,
            error_message=str(e),
            location=location,
            traceback=traceback.format_exc()
        )
        return {
            "status": "error",
            "error_message": f"Error processing weather request: {str(e)}"
        }

def chat_with_weather(prompt: str) -> str:
    """
    Process a weather-related chat message and return a formatted response.
    
    Args:
        prompt (str): User's input message
        
    Returns:
        str: Formatted response with weather information
    """
    try:
        # Extract location from prompt
        words = prompt.lower().split()
        location = None
        for i, word in enumerate(words):
            if word in ["in", "at", "for", "about"]:
                if i + 1 < len(words):
                    location = " ".join(words[i+1:])
                    break
        
        if not location:
            return "I couldn't determine which location you're asking about. Could you please specify a city or place?"
        
        # Get weather information
        weather_info = get_location_weather(location)
        if weather_info["status"] != "success":
            return f"Sorry, I couldn't get weather information: {weather_info.get('error_message', 'Unknown error')}"
        
        # Format response
        current = weather_info["current"]
        response = [
            f"🌡️ Weather in {weather_info['location']}:",
            f"• Current conditions: {current['conditions']}",
            f"• Temperature: {current['temperature']}",
            f"• Humidity: {current['humidity']}",
            f"• Wind Speed: {current['wind_speed']}"
        ]
        
        # Add forecast if available
        if weather_info.get("forecast"):
            # Get tomorrow's forecast if available, otherwise use today's
            forecast = weather_info["forecast"][0]  # Default to today
            if len(weather_info["forecast"]) > 1:
                forecast = weather_info["forecast"][1]  # Use tomorrow if available
            
            response.extend([
                "",
                "🔮 Forecast:",
                f"• High: {forecast['max_temp']}°C",
                f"• Low: {forecast['min_temp']}°C",
                f"• Precipitation chance: {forecast['precipitation_prob']}%",
                f"• Wind speed: {forecast['wind_speed']} km/h"
            ])
        
        return "\n".join(response)
            
    except Exception as e:
        logger.error(
            "Error in weather chat",
            error_type=type(e).__name__,
            error_message=str(e)
        )
        return f"I apologize, but I encountered an error while processing your request: {str(e)}"

# Test function to run when this module is executed directly
if __name__ == "__main__":
    print("Welcome to the Weather Chat!")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        print("Getting weather information...")
        response = chat_with_weather(user_input)
        print(f"Response: {response}")
        print("-" * 40) 