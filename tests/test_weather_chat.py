"""
Weather Chat Test Suite

Tests the weather chat functionality including:
- Location extraction
- Weather API integration
- Response formatting
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import requests

from app.backend.models.weather_chat import (
    get_weather_forecast,
    get_location_weather,
    chat_with_weather
)

# Test data
MOCK_COORDINATES = {
    "status": "success",
    "latitude": 43.7696,
    "longitude": 11.2558,
    "formatted_address": "Florence, Metropolitan City of Florence, Italy"
}

MOCK_WEATHER = {
    "status": "success",
    "current": {
        "temperature": 25.0,
        "humidity": 65,
        "wind_speed": 10.5,
        "conditions": "Clear sky"
    },
    "forecast": [
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "max_temp": 28.0,
            "min_temp": 18.0,
            "precipitation_prob": 10,
            "wind_speed": 12.5
        },
        {  # Add tomorrow's forecast
            "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "max_temp": 27.0,
            "min_temp": 17.0,
            "precipitation_prob": 20,
            "wind_speed": 11.5
        }
    ]
}

@patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "test_key"}, clear=True)
class TestWeatherForecast:
    """Test cases for weather forecast functionality"""
    
    @patch('app.backend.models.weather_chat.requests.get')
    def test_valid_coordinates(self, mock_get):
        """Test weather retrieval with valid coordinates"""
        # Mock the API responses
        mock_current_response = MagicMock()
        mock_current_response.raise_for_status = MagicMock()
        mock_current_response.json.return_value = {
            "temperature": {"degrees": 25.0},
            "relativeHumidity": 65,
            "wind": {"speed": {"value": 10.5}},
            "weatherCondition": {"description": {"text": "Clear sky"}}
        }
        
        mock_forecast_response = MagicMock()
        mock_forecast_response.raise_for_status = MagicMock()
        mock_forecast_response.json.return_value = {
            "dailyForecast": [{
                "date": datetime.now().strftime("%Y-%m-%d"),
                "temperature": {"max": {"degrees": 28.0}, "min": {"degrees": 18.0}},
                "precipitation": {"probability": {"percent": 10}},
                "wind": {"maxSpeed": {"value": 12.5}}
            }]
        }
        
        def mock_get_side_effect(*args, **kwargs):
            if "currentConditions" in args[0]:
                return mock_current_response
            return mock_forecast_response
            
        mock_get.side_effect = mock_get_side_effect
        
        result = get_weather_forecast(latitude=43.7696, longitude=11.2558)

        assert result["status"] == "success"
        assert result["current"]["temperature"] == 25.0
        assert result["current"]["humidity"] == 65
        assert result["current"]["wind_speed"] == 10.5
        assert result["current"]["conditions"] == "Clear sky"
        assert len(result["forecast"]) == 1

    @patch('app.backend.models.weather_chat.requests.get')
    def test_invalid_coordinates(self, mock_get):
        """Test weather retrieval with invalid coordinates"""
        mock_get.side_effect = requests.exceptions.RequestException("Invalid coordinates")
        
        result = get_weather_forecast(latitude=1000, longitude=1000)

        assert result["status"] == "error"
        assert "Failed to fetch weather data" in result["error_message"]

@patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "test_key"}, clear=True)
class TestLocationWeather:
    """Test cases for location weather functionality"""
    
    @patch('requests.get')
    def test_valid_location(self, mock_get):
        """Test getting weather for a valid location"""
        # Mock geocoding response
        mock_geo_response = MagicMock()
        mock_geo_response.raise_for_status = MagicMock()
        mock_geo_response.json.return_value = {
            "status": "OK",
            "results": [{"geometry": {"location": {"lat": 43.7696, "lng": 11.2558}}, "formatted_address": "Florence, Metropolitan City of Florence, Italy"}]
        }

        # Mock weather responses
        mock_current_response = MagicMock()
        mock_current_response.raise_for_status = MagicMock()
        mock_current_response.json.return_value = {
            "temperature": {"degrees": 25.0},
            "relativeHumidity": 65,
            "wind": {"speed": {"value": 10.5}},
            "weatherCondition": {"description": {"text": "Clear sky"}}
        }
        
        mock_forecast_response = MagicMock()
        mock_forecast_response.raise_for_status = MagicMock()
        mock_forecast_response.json.return_value = {
            "dailyForecast": [{"date": datetime.now().strftime("%Y-%m-%d"), "temperature": {"max": {"degrees": 28.0}, "min": {"degrees": 18.0}}, "precipitation": {"probability": {"percent": 10}}, "wind": {"maxSpeed": {"value": 12.5}}}]
        }
        
        def mock_get_side_effect(*args, **kwargs):
            url = args[0]
            if "maps.googleapis.com" in url:
                return mock_geo_response
            elif "currentConditions" in url:
                return mock_current_response
            elif "forecast" in url:
                return mock_forecast_response
            return MagicMock()

        mock_get.side_effect = mock_get_side_effect

        result = get_location_weather("Florence, Italy")
    
        assert result["status"] == "success"
        assert "Florence" in result["location"]
        assert "°C" in result["current"]["temperature"]
        assert "%" in result["current"]["humidity"]
        assert "km/h" in result["current"]["wind_speed"]

    @patch('app.backend.tools.geocoding.requests.get')
    def test_invalid_location(self, mock_geo_get):
        """Test getting weather for an invalid location"""
        mock_geo_response = MagicMock()
        mock_geo_response.raise_for_status = MagicMock()
        mock_geo_response.json.return_value = {"status": "ZERO_RESULTS"}
        mock_geo_get.return_value = mock_geo_response
        
        result = get_location_weather("NonexistentCity123, Nowhere")
        assert result["status"] == "error"
        assert "Geocoding failed" in result["error_message"]

class TestWeatherChat:
    """Test suite for weather chat functionality"""
    
    @patch('app.backend.models.weather_chat.get_location_weather')
    def test_location_extraction(self, mock_get_weather):
        """Test that locations are correctly extracted from prompts"""
        # Mock successful weather response
        mock_get_weather.return_value = {
            "status": "success",
            "location": "Paris, France",
            "current": {
                "temperature": "20.0°C (68.0°F)",
                "conditions": "Clear sky",
                "humidity": "60%",
                "wind_speed": "8.5 km/h"
            },
            "forecast": []
        }
        
        test_cases = [
            ("What's the weather in Paris?", True),
            ("Tell me about the weather for Tokyo", True),
            ("How's the weather looking in New York City today?", True),
            ("Weather forecast", False)  # No location specified
        ]
        
        for prompt, should_have_location in test_cases:
            response = chat_with_weather(prompt)
            if not should_have_location:
                assert "couldn't determine which location" in response.lower()
            else:
                assert response is not None
                assert len(response) > 0
                assert "Weather in" in response
    
    @patch('app.backend.models.weather_chat.get_location_weather')
    def test_response_formatting(self, mock_get_weather):
        """Test that responses are properly formatted"""
        mock_get_weather.return_value = {
            "status": "success",
            "location": "Florence, Italy",
            "current": {
                "temperature": "25.0°C (77.0°F)",
                "conditions": "Clear sky",
                "humidity": "65%",
                "wind_speed": "10.5 km/h"
            },
            "forecast": [{
                "date": datetime.now().strftime("%Y-%m-%d"),
                "max_temp": 28.0,
                "min_temp": 18.0,
                "precipitation_prob": 10,
                "wind_speed": 12.5
            }]
        }
        
        response = chat_with_weather("What's the weather in Florence?")
        assert "🌡️ Weather in Florence, Italy" in response
        assert "Current conditions: Clear sky" in response
        assert "Temperature: 25.0°C (77.0°F)" in response
    
    def test_error_handling(self):
        """Test error handling in chat responses"""
        with patch('app.backend.models.weather_chat.get_location_weather') as mock_get_weather:
            mock_get_weather.return_value = {
                "status": "error",
                "error_message": "API request failed"
            }
            
            response = chat_with_weather("What's the weather in Paris?")
            assert "Sorry, I couldn't get weather information" in response
            assert "API request failed" in response

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 