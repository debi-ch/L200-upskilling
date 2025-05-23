"""
Model Integration

This module provides integration with various LLM models.
"""

from app.backend.models.gemini_chat import chat_with_gemini, set_model_preference, get_current_model_name
from app.backend.models.gemma_chat import chat_with_gemma
from app.backend.models.gemini_rag import chat_with_gemini_rag

__all__ = [
    'chat_with_gemini',
    'chat_with_gemma',
    'set_model_preference',
    'get_current_model_name',
    'chat_with_gemini_rag',
]
