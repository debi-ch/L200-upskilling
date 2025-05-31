"""
Memory-Enhanced RAG System for Travel Assistant

This module combines travel information RAG with user memory/profile RAG
to provide personalized travel recommendations based on user history and preferences.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple

# Add the project root to sys.path for imports
APPLICATION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if APPLICATION_ROOT not in sys.path:
    sys.path.insert(0, APPLICATION_ROOT)

from app.backend.rag.rag_engine import RAGEngine
from app.backend.rag.user_profile_rag import UserProfileRAG
from app.backend.rag.config import RAGConfig
from app.backend.models.gemini_chat_refactored import chat_with_gemini
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("memory_enhanced_rag")

class MemoryEnhancedRAG:
    """RAG system that combines travel information with user memory"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.travel_rag = None
        self.user_rag = None
        self.is_initialized = False

    def load_travel_data(self, travel_data_file: str):
        """Load travel data into the RAG engine"""
        try:
            logger.info("Loading travel data...")
            self.travel_rag = RAGEngine()
            self.travel_rag.process_and_upsert_dataset(travel_data_file)
            logger.info("Travel data loaded successfully")
            self.is_initialized = True  # Set initialized flag when data is loaded
        except Exception as e:
            logger.error(f"Error loading travel data: {e}")
            self.travel_rag = None
            self.is_initialized = False

    def load_user_data(self, user_data_file: str):
        """Load user data into the user profile RAG"""
        try:
            logger.info("Loading user data...")
            self.user_rag = UserProfileRAG(self.config)
            
            # The user data file should contain both profiles and chat histories
            # For now, we'll assume it's in the format our test generates
            self.user_rag.initialize_user_data(user_data_file, user_data_file)
            self.user_rag.is_initialized = True  # Set user RAG initialization flag
            logger.info("User data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading user data: {e}")
            self.user_rag = None

    def query_with_memory(self, query: str, user_id: str, top_k: int = 3) -> str:
        """Query both travel and user data to generate a personalized response."""
        try:
            # Get relevant travel information
            travel_result = self.travel_rag.query(query, top_k=top_k)
            travel_context = travel_result.get('context', '')
            
            # Get user context if available
            user_context = ''
            if self.user_rag and self.user_rag.is_initialized:
                user_context, _ = self.user_rag.get_user_context(user_id, query, max_results=2)
            
            # Get user profile summary
            profile_summary = ''
            if self.user_rag:
                profile_summary = self.user_rag.get_user_profile_summary(user_id)
            
            # Combine contexts and generate response
            combined_prompt = f"""You are an expert and friendly AI travel assistant specializing in Buenos Aires.
Your task is to provide personalized travel recommendations based on the user's profile and preferences.

User Profile and Preferences:
{profile_summary}

Previous Travel Experience and Interests:
{user_context}

Relevant Travel Information:
{travel_context}

Important Instructions:
1. Use the user's known preferences (budget, interests, travel style) to tailor your recommendations
2. Reference their past travel experiences when relevant
3. If you don't have specific information about something, acknowledge what you do know about their preferences
4. Keep your tone friendly and conversational, as if continuing an ongoing discussion
5. If this is your first interaction, briefly acknowledge their preferences but still ask for any specific requirements for this trip

Based on this information, please provide a personalized response to: "{query}"
"""
            
            # Use Gemini to generate the final response
            response = chat_with_gemini(combined_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error in query_with_memory: {e}", exc_info=True)
            return "I encountered an error while trying to generate a personalized response. Please try again."

    def _generate_response(self, query: str, travel_context: str, user_context: str) -> str:
        """Generate response using appropriate prompt template"""
        
        try:
            # Determine which template to use based on available context
            if travel_context and user_context:
                # Both contexts available - use memory-enhanced template
                prompt = self.config.memory_enhanced_template.format(
                    user_context=user_context,
                    travel_context=travel_context,
                    query=query
                )
            elif user_context:
                # Only user context available
                prompt = self.config.user_context_only_template.format(
                    user_context=user_context,
                    query=query
                )
            elif travel_context:
                # Only travel context available - use standard RAG template
                prompt = self.config.context_prompt_template.format(
                    context=travel_context,
                    query=query
                )
            else:
                # No context available - direct query
                prompt = f"You are a helpful travel assistant. Please answer this query: {query}"
            
            # Generate response using Gemini
            response = chat_with_gemini(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    def get_personalization_insights(self, user_id: str, query: str) -> Dict[str, Any]:
        """Get insights about how the response was personalized"""
        
        if not self.user_rag or not user_id:
            return {"personalized": False, "reason": "No user context available"}
        
        try:
            insights = self.user_rag.get_personalization_insights(user_id, query)
            insights["personalized"] = insights.get("has_user_context", False)
            return insights
        except Exception as e:
            logger.error(f"Error getting personalization insights: {e}")
            return {"personalized": False, "reason": f"Error: {e}"}

    def search_similar_users(self, query: str, exclude_user_id: str = None) -> List[Dict]:
        """Find users with similar preferences for the query"""
        
        if not self.user_rag:
            return []
        
        try:
            return self.user_rag.search_similar_users(query, exclude_user_id)
        except Exception as e:
            logger.error(f"Error searching similar users: {e}")
            return []

    def get_user_profile_summary(self, user_id: str) -> str:
        """Get a summary of user profile"""
        
        if not self.user_rag or not user_id:
            return ""
        
        try:
            return self.user_rag.get_user_profile_summary(user_id)
        except Exception as e:
            logger.error(f"Error getting user profile summary: {e}")
            return "" 