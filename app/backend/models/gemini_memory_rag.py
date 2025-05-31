"""
Gemini Memory RAG Model Integration

This module provides Memory-enhanced RAG functionality using the MemoryEnhancedRAG system.
It includes user identification and personalized responses based on user memory.
"""
import os
from typing import Dict, Any, Optional

from app.backend.models.memory_enhanced_rag import MemoryEnhancedRAG
from app.backend.rag.user_profile_rag import UserProfileRAG
from app.backend.rag.config import RAGConfig
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("gemini_memory_rag_model")

# Global instance of MemoryEnhancedRAG
memory_rag_instance = None
_memory_rag_initialized = False

def _ensure_memory_rag_initialized() -> bool:
    """Ensures the Memory RAG system is initialized."""
    global memory_rag_instance, _memory_rag_initialized
    
    if _memory_rag_initialized and memory_rag_instance:
        logger.info("Memory RAG already initialized and instance exists")
        return True
    
    try:
        logger.info("Initializing Memory RAG system...")
        config = RAGConfig()
        memory_rag_instance = MemoryEnhancedRAG(config)
        logger.info("Created new MemoryEnhancedRAG instance")
        
        # Get the application root directory
        app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        
        # Load travel data using simplified path
        travel_data_path = os.path.join(app_root, "data", "travel", "sample_travel.ndjson")
        
        if os.path.exists(travel_data_path):
            logger.info(f"Loading travel data from: {travel_data_path}")
            memory_rag_instance.load_travel_data(travel_data_path)
            logger.info(f"Travel data loaded, is_initialized={memory_rag_instance.is_initialized}")
            if not memory_rag_instance.travel_rag:
                logger.error("Travel data loading failed - RAG engine not created")
                return False
        else:
            logger.warning(f"Travel data file not found: {travel_data_path}")
            return False
        
        # Load user data with simplified paths
        user_data_path = os.path.join(app_root, "data", "user_profiles", "test_users.json")
        chat_history_path = os.path.join(app_root, "data", "user_profiles", "test_chat_histories.json")
        
        if os.path.exists(user_data_path) and os.path.exists(chat_history_path):
            logger.info(f"Loading user data from: {user_data_path}")
            memory_rag_instance.user_rag = UserProfileRAG(config)
            memory_rag_instance.user_rag.initialize_user_data(user_data_path, chat_history_path)
            
            if not memory_rag_instance.user_rag.is_initialized:
                logger.error("User data loading failed - UserProfileRAG not initialized")
                return False
            logger.info("User data loaded and initialized successfully")
        else:
            logger.error("User data or chat history files not found")
            return False
        
        # Set initialized only if both travel and user data are ready
        if (memory_rag_instance.travel_rag and 
            memory_rag_instance.user_rag and 
            memory_rag_instance.user_rag.is_initialized):
            _memory_rag_initialized = True
            status = get_memory_rag_status()
            logger.info(f"Memory RAG system initialization complete. Status: {status}")
            return True
        else:
            logger.error("Memory RAG system failed to initialize - components not ready")
            return False
        
    except Exception as e:
        logger.error(f"Failed to initialize Memory RAG system: {e}", exc_info=True)
        return False

def chat_with_memory_rag(
    user_prompt: str, 
    user_id: str = "default_user",
    top_k: int = 3
) -> str:
    """
    Chat with Gemini using Memory-enhanced RAG.
    
    Args:
        user_prompt: The user's question/prompt
        user_id: Unique identifier for the user (for personalization)
        top_k: Number of relevant documents to retrieve
    
    Returns:
        Personalized response based on user memory and travel data
    """
    if not _ensure_memory_rag_initialized():
        logger.error("Memory RAG system not initialized. Falling back to error message.")
        return "I'm having trouble accessing my memory and knowledge base right now. Please try again later."
    
    try:
        logger.info(f"Memory RAG Chat: User '{user_id}' asked: '{user_prompt[:100]}...'")
        
        # Use the Memory RAG system to generate a personalized response
        response = memory_rag_instance.query_with_memory(
            query=user_prompt,
            user_id=user_id,
            top_k=top_k
        )
        
        logger.info(f"Generated personalized response for user '{user_id}' (length: {len(response)})")
        return response
        
    except Exception as e:
        logger.error(f"Error in Memory RAG chat: {e}", exc_info=True)
        return "I encountered an issue while generating your personalized response. Please try again."

def add_user_interaction(user_id: str, user_message: str, assistant_response: str) -> bool:
    """
    Add a user interaction to the memory system for future personalization.
    
    Args:
        user_id: Unique identifier for the user
        user_message: The user's message
        assistant_response: The assistant's response
    
    Returns:
        True if successfully added, False otherwise
    """
    if not _ensure_memory_rag_initialized():
        return False
    
    try:
        # Create a simple interaction record
        interaction = {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "timestamp": "now"  # You might want to use actual timestamp
        }
        
        # Add to user's chat history (this would need to be implemented in MemoryEnhancedRAG)
        # For now, we'll log it
        logger.info(f"Recording interaction for user '{user_id}': {user_message[:50]}...")
        return True
        
    except Exception as e:
        logger.error(f"Error recording user interaction: {e}", exc_info=True)
        return False

def get_user_profile_summary(user_id: str) -> Optional[str]:
    """
    Get a summary of the user's profile and preferences.
    
    Args:
        user_id: Unique identifier for the user
    
    Returns:
        Summary string or None if not available
    """
    if not _ensure_memory_rag_initialized():
        return None
    
    try:
        # This would need to be implemented in the MemoryEnhancedRAG system
        # For now, return a placeholder
        return f"User profile for '{user_id}' - personalization active"
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}", exc_info=True)
        return None

def get_memory_rag_status() -> Dict[str, bool]:
    """Get the current status of the Memory RAG system.
    
    Returns:
        Dict with status flags:
        - initialized: Whether the system has been initialized
        - instance_available: Whether the MemoryEnhancedRAG instance exists
        - ready: Whether the system is ready to process queries with personalization
    """
    global memory_rag_instance, _memory_rag_initialized
    
    status = {
        "initialized": _memory_rag_initialized,
        "instance_available": memory_rag_instance is not None,
        "ready": False
    }
    
    # Check if the instance is ready (requires both travel and user data)
    if status["instance_available"]:
        status["ready"] = (
            memory_rag_instance.is_initialized and
            memory_rag_instance.travel_rag is not None and
            memory_rag_instance.user_rag is not None and
            memory_rag_instance.user_rag.is_initialized
        )
    
    return status

def debug_memory_rag_content():
    """Debug function to check content retrieval in Memory RAG"""
    global memory_rag_instance
    
    if memory_rag_instance is None:
        return {"error": "Memory RAG not initialized"}
    
    # Test a simple query
    test_query = "luxury hotels Buenos Aires"
    
    # Check RAG engine content
    rag_result = memory_rag_instance.travel_rag.query(test_query, top_k=3)
    
    return {
        "rag_context_length": len(rag_result.get('context', '')),
        "rag_context_preview": rag_result.get('context', '')[:200],
        "chunk_data_map_size": len(memory_rag_instance.travel_rag.chunk_data_map),
        "vector_search_results": len(rag_result.get('search_results', []))
    }

def force_memory_rag_reload():
    """Force reload the Memory RAG system to fix content issues"""
    global memory_rag_instance
    
    logger.info("Force reloading Memory RAG system...")
    
    # Clear the existing instance
    memory_rag_instance = None
    
    # Reinitialize
    _ensure_memory_rag_initialized()
    
    # Test the reload
    debug_info = debug_memory_rag_content()
    logger.info(f"Memory RAG reload complete. Debug info: {debug_info}")
    
    return debug_info

if __name__ == '__main__':
    # Simple test for this module
    print("Testing gemini_memory_rag.py...")
    
    # Test initialization
    if _ensure_memory_rag_initialized():
        print("✅ Memory RAG system initialized successfully!")
        
        # Test queries
        test_queries = [
            "I'm looking for budget-friendly hotels in Buenos Aires",
            "What are some romantic restaurants for a special dinner?",
            "Can you recommend family-friendly activities?"
        ]
        
        test_user = "test_user_123"
        
        for query in test_queries:
            print(f"\nUser Query: {query}")
            response = chat_with_memory_rag(query, test_user)
            print(f"Memory RAG Response: {response[:200]}...")
            print("---")
    else:
        print("❌ Failed to initialize Memory RAG system") 