#!/usr/bin/env python3
"""
Quick test for Memory RAG system with pre-made sample data.
"""

import json
from app.backend.models.gemini_memory_rag import MemoryEnhancedRAG

def test_memory_rag_quick():
    """Quick test with pre-made sample data."""
    print("🚀 Quick Memory RAG Test")
    print("=" * 50)
    
    # Initialize the Memory RAG engine
    print("Initializing Memory RAG engine...")
    memory_rag = MemoryEnhancedRAG()
    
    # Sample user profile
    sample_user = {
        "user_id": "test_user_001",
        "name": "Sarah Johnson",
        "age": 32,
        "location": "San Francisco, CA",
        "travel_preferences": {
            "budget": "mid-range",
            "accommodation_type": "boutique hotels",
            "interests": ["cultural experiences", "local cuisine", "photography"],
            "travel_style": "solo traveler"
        },
        "past_trips": [
            {"destination": "Paris", "year": 2023, "rating": 5},
            {"destination": "Tokyo", "year": 2022, "rating": 4}
        ]
    }
    
    # Sample chat history
    sample_chat_history = [
        {"role": "user", "content": "I'm planning a trip to Buenos Aires next month"},
        {"role": "assistant", "content": "That's exciting! Buenos Aires is a wonderful destination. What type of experiences are you most interested in?"},
        {"role": "user", "content": "I love cultural experiences and trying local food"}
    ]
    
    print(f"Testing with user: {sample_user['name']}")
    print(f"Chat history length: {len(sample_chat_history)} messages")
    
    # Test query
    test_query = "Can you recommend some boutique hotels in Buenos Aires that would be good for a solo traveler interested in culture and food?"
    
    print(f"\nTest Query: {test_query}")
    print("\nGenerating response...")
    
    try:
        response = memory_rag.chat_with_memory(
            query=test_query,
            user_id=sample_user["user_id"],
            chat_history=sample_chat_history
        )
        
        print("\n" + "="*50)
        print("MEMORY RAG RESPONSE:")
        print("="*50)
        print(response)
        print("="*50)
        
        # Test without chat history
        print("\n🔄 Testing without chat history...")
        response_no_history = memory_rag.chat_with_memory(
            query="What are some must-visit cultural sites in Buenos Aires?",
            user_id=sample_user["user_id"],
            chat_history=None
        )
        
        print("\nResponse without chat history:")
        print("-" * 30)
        print(response_no_history)
        
        # Test without user ID
        print("\n🔄 Testing without user ID...")
        response_no_user = memory_rag.chat_with_memory(
            query="Tell me about tango shows in Buenos Aires",
            user_id=None,
            chat_history=sample_chat_history
        )
        
        print("\nResponse without user ID:")
        print("-" * 30)
        print(response_no_user)
        
        print("\n✅ Quick Memory RAG test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_rag_quick() 