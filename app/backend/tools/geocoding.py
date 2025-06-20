"""
Geocoding Tool Module

Provides location to coordinate conversion using Google Maps Geocoding API.
"""

import os
from typing import Dict, Any
import requests
from app.utils.logging_utils import ChatbotLogger

# Initialize logger
logger = ChatbotLogger("geocoding_tool")

def get_coordinates(location: str) -> Dict[str, Any]:
    """
    Convert location name to coordinates using Google Maps Geocoding API.
    
    Args:
        location (str): Name of the location (e.g., "Florence, Italy")
        
    Returns:
        dict: Location data including:
            - latitude
            - longitude
            - formatted_address
            - status
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    logger.info(f"Attempting to get coordinates for location: {location}")
    logger.info(f"API Key present: {'Yes' if api_key else 'No'}")
    
    if not api_key:
        logger.error("Google Maps API key not found in environment")
        return {
            "status": "error",
            "error_message": "Google Maps API key not configured. Please set GOOGLE_MAPS_API_KEY environment variable."
        }
    
    try:
        # Prepare the request
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": location,
            "key": api_key
        }
        
        logger.info(f"Making geocoding request for: {location}")
        # Make the request
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "OK" and data["results"]:
            result = data["results"][0]
            location_data = {
                "status": "success",
                "latitude": result["geometry"]["location"]["lat"],
                "longitude": result["geometry"]["location"]["lng"],
                "formatted_address": result["formatted_address"]
            }
            logger.info(f"Successfully got coordinates for {location}: {location_data}")
            return location_data
        else:
            error_msg = data.get("error_message", data["status"])
            logger.error(f"Geocoding failed for {location}: {error_msg}")
            return {
                "status": "error",
                "error_message": f"Geocoding failed: {error_msg}"
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {location}: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Failed to fetch coordinates: {str(e)}"
        } 