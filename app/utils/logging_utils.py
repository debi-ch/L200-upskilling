"""
Logging Utilities for Chatbot Application

This module provides centralized logging functionality using Google Cloud Logging
as well as local logging for development.
"""

import os
import time
import logging as python_logging
from google.cloud import logging as google_logging

# Configure standard Python logging
python_logging.basicConfig(
    level=python_logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        python_logging.StreamHandler(),
        python_logging.FileHandler('chatbot_app.log')
    ]
)

class ChatbotLogger:
    """
    A centralized logger for the chatbot application that supports
    both Google Cloud Logging and local logging.
    """
    
    def __init__(self, name="chatbot_app", use_cloud=True):
        """
        Initialize the logger.
        
        Args:
            name (str): The name of the logger
            use_cloud (bool): Whether to use Google Cloud Logging
        """
        self.name = name
        self.use_cloud = use_cloud
        self.python_logger = python_logging.getLogger(name)
        
        # Initialize Google Cloud Logging if enabled
        if use_cloud:
            try:
                self.client = google_logging.Client()
                self.cloud_logger = self.client.logger(name)
                self.cloud_enabled = True
            except Exception as e:
                self.python_logger.error(f"Failed to initialize Google Cloud Logging: {e}")
                self.cloud_enabled = False
        else:
            self.cloud_enabled = False
    
    def info(self, message, **kwargs):
        """Log an info message"""
        self.python_logger.info(message)
        if self.cloud_enabled:
            self._log_to_cloud('INFO', message, **kwargs)
    
    def warning(self, message, **kwargs):
        """Log a warning message"""
        self.python_logger.warning(message)
        if self.cloud_enabled:
            self._log_to_cloud('WARNING', message, **kwargs)
    
    def error(self, message, **kwargs):
        """Log an error message"""
        self.python_logger.error(message)
        if self.cloud_enabled:
            self._log_to_cloud('ERROR', message, **kwargs)
    
    def debug(self, message, **kwargs):
        """Log a debug message"""
        self.python_logger.debug(message)
        if self.cloud_enabled:
            self._log_to_cloud('DEBUG', message, **kwargs)
    
    def log_user_message(self, session_id, user_message, model_name):
        """Log a user message with standardized format"""
        log_data = {
            'event_type': 'user_message',
            'session_id': session_id,
            'model': model_name,
            'message': user_message,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.info("User message", **log_data)
        return log_data
    
    def log_model_response(self, session_id, model_name, prompt_length, response, response_time):
        """Log a model response with standardized format"""
        log_data = {
            'event_type': 'model_response',
            'session_id': session_id,
            'model': model_name,
            'prompt_length': prompt_length,
            'response_length': len(response) if response else 0,
            'response_time': response_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.info("Model response", **log_data)
        return log_data
    
    def log_error(self, error_type, error_message, **context):
        """Log an error with additional context"""
        log_data = {
            'event_type': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'context': context,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.error(f"{error_type}: {error_message}", **log_data)
    
    def _log_to_cloud(self, severity, message, **kwargs):
        """Send a log entry to Google Cloud Logging"""
        if self.cloud_enabled:
            try:
                # Prepare the log entry data
                log_data = {
                    'message': message,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    **kwargs
                }
                
                # Log as a structured entry
                self.cloud_logger.log_struct(
                    log_data,
                    severity=severity
                )
            except Exception as e:
                # Fall back to local logging if cloud logging fails
                self.python_logger.error(f"Failed to log to Google Cloud: {e}")

# Create a default instance for import
default_logger = ChatbotLogger()

# Convenience functions using the default logger
def info(message, **kwargs):
    default_logger.info(message, **kwargs)

def warning(message, **kwargs):
    default_logger.warning(message, **kwargs)

def error(message, **kwargs):
    default_logger.error(message, **kwargs)

def debug(message, **kwargs):
    default_logger.debug(message, **kwargs)

def log_user_message(session_id, user_message, model_name):
    return default_logger.log_user_message(session_id, user_message, model_name)

def log_model_response(session_id, model_name, prompt_length, response, response_time):
    return default_logger.log_model_response(session_id, model_name, prompt_length, response, response_time)

def log_error(error_type, error_message, **context):
    default_logger.log_error(error_type, error_message, **context) 