#!/usr/bin/env python3
"""
Test script for Memory RAG system (Phase 1 of Week 6)
Tests user profile generation, memory processing, and personalized responses.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to sys.path
APPLICATION_ROOT = os.path.abspath(os.path.dirname(__file__))
if APPLICATION_ROOT not in sys.path:
    sys.path.insert(0, APPLICATION_ROOT)

def test_sample_data_generation():
    """Test 1: Generate sample user data"""
    
    print("🧪 Test 1: Generating sample user data...")
    
    try:
        from generate_sample_users import generate_user_profiles, save_user_profiles, generate_chat_history_for_user
        
        # Generate a few test users
        users = generate_user_profiles(3)
        
        if not users:
            print("❌ Failed to generate users")
            return False
        
        # Save users
        save_user_profiles(users, "data/user_profiles/test_users.json")
        
        # Generate chat histories
        all_chat_data = {}
        for user in users:
            print(f"Generating chat history for {user['name']}...")
            conversations = generate_chat_history_for_user(user, 2)
            
            if conversations:
                all_chat_data[user['user_id']] = {
                    'user_profile': user,
                    'conversations': conversations
                }
        
        # Save chat histories
        chat_filename = "data/user_profiles/test_chat_histories.json"
        Path(chat_filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(chat_filename, 'w') as f:
            json.dump(all_chat_data, f, indent=2)
        
        print(f"✅ Generated {len(users)} test users with chat histories")
        return True
        
    except Exception as e:
        print(f"❌ Error in sample data generation: {e}")
        return False

def test_memory_processor():
    """Test 2: Memory processor functionality"""
    
    print("\n🧪 Test 2: Testing memory processor...")
    
    try:
        from app.backend.rag.memory_processor import MemoryProcessor
        
        # Initialize processor
        processor = MemoryProcessor()
        
        # Load test data
        profiles_file = "data/user_profiles/test_users.json"
        chat_file = "data/user_profiles/test_chat_histories.json"
        
        if not os.path.exists(profiles_file):
            print("❌ Test user data not found. Run test 1 first.")
            return False
        
        processor.load_user_profiles(profiles_file)
        processor.load_chat_histories(chat_file)
        
        # Test processing
        all_chunks = processor.process_all_user_data()
        
        if not all_chunks:
            print("❌ No chunks generated")
            return False
        
        print(f"✅ Generated {len(all_chunks)} memory chunks")
        
        # Test user context extraction
        if processor.user_profiles:
            first_user_id = list(processor.user_profiles.keys())[0]
            context = processor.get_user_context_for_query(first_user_id, "budget travel")
            print(f"✅ User context extraction: {len(context)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in memory processor test: {e}")
        return False

def test_user_profile_rag():
    """Test 3: User Profile RAG functionality"""
    
    print("\n🧪 Test 3: Testing User Profile RAG...")
    
    try:
        from app.backend.rag.user_profile_rag import UserProfileRAG
        
        # Initialize RAG
        user_rag = UserProfileRAG()
        
        # Test data files
        profiles_file = "data/user_profiles/test_users.json"
        chat_file = "data/user_profiles/test_chat_histories.json"
        
        if not os.path.exists(profiles_file):
            print("❌ Test user data not found. Run test 1 first.")
            return False
        
        # Initialize with test data
        user_rag.initialize_user_data(profiles_file, chat_file)
        
        if not user_rag.is_initialized:
            print("❌ User RAG initialization failed")
            return False
        
        # Test user context retrieval
        with open(profiles_file, 'r') as f:
            users = json.load(f)
        
        if users:
            test_user_id = users[0]['user_id']
            test_queries = [
                "budget travel options",
                "cultural experiences",
                "beach destinations",
                "adventure activities"
            ]
            
            for query in test_queries:
                context, chunks = user_rag.get_user_context(test_user_id, query)
                print(f"✅ Query '{query}': {len(chunks)} relevant chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in User Profile RAG test: {e}")
        return False

def test_memory_enhanced_rag():
    """Test 4: Memory-Enhanced RAG integration"""
    
    print("\n🧪 Test 4: Testing Memory-Enhanced RAG...")
    
    try:
        from app.backend.models.gemini_memory_rag import MemoryEnhancedRAG
        
        # Initialize memory RAG
        memory_rag = MemoryEnhancedRAG()
        
        # Use test data
        profiles_file = "data/user_profiles/test_users.json"
        chat_file = "data/user_profiles/test_chat_histories.json"
        
        if not os.path.exists(profiles_file):
            print("❌ Test user data not found. Run test 1 first.")
            return False
        
        # Initialize with test data
        memory_rag.initialize(
            user_profiles_file=profiles_file,
            chat_histories_file=chat_file
        )
        
        # Test queries
        test_queries = [
            "I want to visit Buenos Aires. What hotels do you recommend?",
            "What are some budget-friendly options?",
            "I'm interested in cultural experiences"
        ]
        
        # Test without user context
        print("\n🤖 Testing without user context:")
        for query in test_queries[:1]:
            response = memory_rag.chat_with_memory(query)
            print(f"✅ Response generated: {len(response)} characters")
        
        # Test with user context
        with open(profiles_file, 'r') as f:
            users = json.load(f)
        
        if users:
            test_user_id = users[0]['user_id']
            test_user_name = users[0]['name']
            
            print(f"\n🧑‍💻 Testing with user context: {test_user_name}")
            for query in test_queries[1:]:
                response = memory_rag.chat_with_memory(query, test_user_id)
                print(f"✅ Personalized response: {len(response)} characters")
                
                # Test personalization insights
                insights = memory_rag.get_personalization_insights(test_user_id, query)
                print(f"✅ Personalization insights: {insights.get('personalized', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Memory-Enhanced RAG test: {e}")
        return False

def test_integration_scenarios():
    """Test 5: Integration scenarios"""
    
    print("\n🧪 Test 5: Testing integration scenarios...")
    
    try:
        from app.backend.models.gemini_memory_rag import get_memory_rag_instance, get_current_memory_mode
        
        # Test global instance
        memory_rag = get_memory_rag_instance()
        
        # Test mode detection
        mode = get_current_memory_mode()
        print(f"✅ Current mode: {mode}")
        
        # Test with different user scenarios
        profiles_file = "data/user_profiles/test_users.json"
        
        if os.path.exists(profiles_file):
            with open(profiles_file, 'r') as f:
                users = json.load(f)
            
            if len(users) >= 2:
                # Test similar user search
                similar_users = memory_rag.search_similar_users(
                    "budget travel", 
                    exclude_user_id=users[0]['user_id']
                )
                print(f"✅ Similar users found: {len(similar_users)}")
                
                # Test profile summaries
                for user in users[:2]:
                    summary = memory_rag.get_user_profile_summary(user['user_id'])
                    print(f"✅ Profile summary for {user['name']}: {len(summary)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in integration scenarios test: {e}")
        return False

def cleanup_test_data():
    """Clean up test data files"""
    
    test_files = [
        "data/user_profiles/test_users.json",
        "data/user_profiles/test_chat_histories.json"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🧹 Cleaned up {file_path}")

def main():
    """Run all memory RAG tests"""
    
    print("🚀 Starting Memory RAG Test Suite (Phase 1 - Week 6)")
    print("=" * 60)
    
    tests = [
        ("Sample Data Generation", test_sample_data_generation),
        ("Memory Processor", test_memory_processor),
        ("User Profile RAG", test_user_profile_rag),
        ("Memory-Enhanced RAG", test_memory_enhanced_rag),
        ("Integration Scenarios", test_integration_scenarios)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Memory RAG Phase 1 is ready.")
        print("\nNext steps:")
        print("1. Generate production user data: python generate_sample_users.py")
        print("2. Integrate with Streamlit app")
        print("3. Test with real user interactions")
    else:
        print("⚠️ Some tests failed. Please review the errors above.")
    
    # Ask about cleanup
    # cleanup = input("\n🧹 Clean up test data files? (y/n): ").lower().strip()
    # if cleanup == 'y':
    #     cleanup_test_data()

if __name__ == "__main__":
    main() 