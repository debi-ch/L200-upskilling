"""
Model Integration Module

This module provides integration with various LLM models.
"""
from app.backend.models.gemini_chat import chat_with_gemini, set_model_preference, get_current_model_name
from app.backend.models.gemma_chat import chat_with_gemma
