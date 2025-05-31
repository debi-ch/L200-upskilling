"""
User Profile RAG Engine
Specialized RAG system for user memory, preferences, and chat history.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import json

# Add the project root to sys.path for imports
APPLICATION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if APPLICATION_ROOT not in sys.path:
    sys.path.insert(0, APPLICATION_ROOT)

from app.backend.rag.memory_processor import MemoryProcessor
from app.backend.rag.embedding_service import EmbeddingService
from app.backend.rag.vector_db import VectorSearchDB
from app.backend.rag.config import RAGConfig

class UserProfileRAG:
    """RAG engine specifically for user profiles and memory"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.memory_processor = MemoryProcessor()
        self.embedding_service = EmbeddingService()
        
        # Use a separate vector database for user data
        self.user_vector_db = VectorSearchDB()
        
        # In-memory storage for quick user context retrieval
        self.user_context_map = {}
        self.is_initialized = False
    
    def initialize_user_data(self, profiles_file: str, chat_histories_file: str) -> None:
        """Initialize user data from files"""
        
        print("🧠 Initializing User Profile RAG...")
        
        try:
            # Load user data
            self.memory_processor.load_user_profiles(profiles_file)
            self.memory_processor.load_chat_histories(chat_histories_file)
            
            # Process all user data into chunks
            user_chunks = self.memory_processor.process_all_user_data()
            
            if not user_chunks:
                print("⚠️ No user chunks to process")
                self.is_initialized = False
                return
            
            # Generate embeddings and store in vector database
            print(f"📊 Processing {len(user_chunks)} user memory chunks...")
            
            successful_chunks = 0
            for chunk in user_chunks:
                try:
                    # Generate embedding
                    embedding = self.embedding_service.generate_embedding(chunk['content'])
                    
                    # Store in vector database using correct method name
                    self.user_vector_db.upsert_documents([{
                        'id': chunk['chunk_id'],
                        'embedding': embedding,
                        'metadata': chunk['metadata']
                    }])
                    
                    # Store in memory map for quick retrieval
                    self.user_context_map[chunk['chunk_id']] = {
                        'content': chunk['content'],
                        'metadata': chunk['metadata']
                    }
                    successful_chunks += 1
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk['chunk_id']}: {e}")
                    continue
            
            # Only set initialized if we successfully processed some chunks
            if successful_chunks > 0:
                self.is_initialized = True
                print(f"✅ User Profile RAG initialized with {len(self.user_context_map)} memory chunks")
            else:
                self.is_initialized = False
                print("❌ Failed to process any chunks successfully")
                
        except Exception as e:
            print(f"❌ Error initializing User Profile RAG: {e}")
            self.is_initialized = False
    
    def get_user_context(self, user_id: str, query: str, max_results: int = 5) -> Tuple[str, List[Dict]]:
        """Get relevant user context for a query"""
        
        if not self.is_initialized:
            return "", []
        
        try:
            # Search for relevant user context using correct method name
            search_results = self.user_vector_db.search(
                query_text=query,
                top_k=max_results * 2  # Get more results to filter by user
            )
            
            # Filter results for the specific user and get context
            user_contexts = []
            retrieved_chunks = []
            
            for result in search_results:
                chunk_id = result['id']
                
                if chunk_id in self.user_context_map:
                    chunk_data = self.user_context_map[chunk_id]
                    
                    # Only include chunks for the specific user
                    if chunk_data['metadata'].get('user_id') == user_id:
                        user_contexts.append(chunk_data['content'])
                        retrieved_chunks.append({
                            'chunk_id': chunk_id,
                            'content': chunk_data['content'],
                            'metadata': chunk_data['metadata'],
                            'score': result.get('distance', 0)
                        })
                        
                        if len(user_contexts) >= max_results:
                            break
            
            # Combine contexts
            combined_context = "\n\n".join(user_contexts) if user_contexts else ""
            
            return combined_context, retrieved_chunks
            
        except Exception as e:
            print(f"Error retrieving user context: {e}")
            return "", []
    
    def get_user_profile_summary(self, user_id: str) -> str:
        """Get a structured summary of user profile and preferences"""
        
        try:
            # Get the raw profile data
            profile_data = self.memory_processor.get_user_profile(user_id)
            if not profile_data:
                return ""
            
            # Extract key preferences
            preferences = profile_data.get('travel_preferences', {})
            
            # Format the summary in a structured way
            summary = f"""Name: {profile_data.get('name', 'Unknown')}
Location: {profile_data.get('location', 'Unknown')}

Travel Preferences:
• Budget Range: {preferences.get('budget_range', 'Not specified')}
• Travel Style: {preferences.get('travel_style', 'Not specified')}
• Accommodation: {preferences.get('accommodation_type', 'Not specified')}
• Group Size: {preferences.get('group_size', 'Not specified')}
• Key Interests: {', '.join(preferences.get('interests', ['Not specified']))}

Languages: {', '.join(profile_data.get('languages', ['Not specified']))}
Dietary Restrictions: {', '.join(profile_data.get('dietary_restrictions', ['None']))}
Travel Goals: {', '.join(profile_data.get('travel_goals', ['Not specified']))}"""

            return summary
            
        except Exception as e:
            print(f"Error getting user profile summary: {e}")
            return ""
    
    def search_similar_users(self, query: str, exclude_user_id: str = None, max_results: int = 3) -> List[Dict]:
        """Find users with similar preferences or experiences"""
        
        if not self.is_initialized:
            return []
        
        try:
            # Search for relevant contexts using correct method name
            search_results = self.user_vector_db.search(
                query_text=query,
                top_k=max_results * 3  # Get more to filter
            )
            
            # Group by user and find similar users
            user_matches = {}
            
            for result in search_results:
                chunk_id = result['id']
                
                if chunk_id in self.user_context_map:
                    chunk_data = self.user_context_map[chunk_id]
                    user_id = chunk_data['metadata'].get('user_id')
                    
                    # Skip the excluded user
                    if user_id == exclude_user_id:
                        continue
                    
                    if user_id not in user_matches:
                        user_matches[user_id] = {
                            'user_id': user_id,
                            'name': chunk_data['metadata'].get('name', 'Unknown'),
                            'matches': [],
                            'best_score': result.get('distance', 1.0)
                        }
                    
                    user_matches[user_id]['matches'].append({
                        'type': chunk_data['metadata'].get('type'),
                        'content': chunk_data['content'][:200] + "...",
                        'score': result.get('distance', 1.0)
                    })
                    
                    # Update best score
                    current_score = result.get('distance', 1.0)
                    if current_score < user_matches[user_id]['best_score']:
                        user_matches[user_id]['best_score'] = current_score
            
            # Sort by best score and return top matches
            similar_users = sorted(user_matches.values(), key=lambda x: x['best_score'])
            return similar_users[:max_results]
            
        except Exception as e:
            print(f"Error finding similar users: {e}")
            return []
    
    def add_real_time_context(self, user_id: str, conversation_messages: List[Dict]) -> None:
        """Add real-time conversation context (for future enhancement)"""
        
        # This could be used to add current conversation context
        # to the user's memory in real-time
        pass
    
    def get_personalization_insights(self, user_id: str, query: str) -> Dict[str, Any]:
        """Get insights for personalizing responses"""
        
        if not self.is_initialized:
            return {}
        
        # Get user context
        user_context, retrieved_chunks = self.get_user_context(user_id, query)
        
        # Get profile summary
        profile_summary = self.get_user_profile_summary(user_id)
        
        # Analyze retrieved chunks for insights
        insights = {
            'has_user_context': bool(user_context),
            'context_types': [],
            'relevant_experiences': [],
            'preferences': {},
            'profile_summary': profile_summary
        }
        
        for chunk in retrieved_chunks:
            chunk_type = chunk['metadata'].get('type', 'unknown')
            insights['context_types'].append(chunk_type)
            
            if chunk_type == 'past_trip':
                insights['relevant_experiences'].append({
                    'destination': chunk['metadata'].get('destination'),
                    'year': chunk['metadata'].get('year'),
                    'purpose': chunk['metadata'].get('purpose')
                })
            elif chunk_type == 'travel_preferences':
                insights['preferences'].update({
                    'budget_range': chunk['metadata'].get('budget_range'),
                    'travel_style': chunk['metadata'].get('travel_style')
                })
        
        return insights

def test_user_profile_rag():
    """Test function for User Profile RAG"""
    
    print("🧪 Testing User Profile RAG...")
    
    # Initialize RAG
    user_rag = UserProfileRAG()
    
    # Test with sample data (you'll need to generate this first)
    profiles_file = "data/user_profiles/sample_users.json"
    chat_file = "data/user_profiles/sample_chat_histories.json"
    
    if not os.path.exists(profiles_file):
        print(f"❌ Sample data not found at {profiles_file}")
        print("Please run: python generate_sample_users.py")
        return
    
    # Initialize with sample data
    user_rag.initialize_user_data(profiles_file, chat_file)
    
    # Test queries
    test_queries = [
        "I want to visit a beach destination",
        "What are some budget-friendly options?",
        "I'm interested in cultural experiences",
        "Where should I go for adventure travel?"
    ]
    
    # Test with first user
    with open(profiles_file, 'r') as f:
        users = json.load(f)
    
    if users:
        test_user_id = users[0]['user_id']
        test_user_name = users[0]['name']
        
        print(f"\n🧑‍💻 Testing with user: {test_user_name} ({test_user_id})")
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            
            context, chunks = user_rag.get_user_context(test_user_id, query)
            
            if context:
                print(f"📊 Retrieved {len(chunks)} relevant memory chunks")
                print(f"🧠 User Context:\n{context[:300]}...")
            else:
                print("❌ No relevant context found")
        
        # Test personalization insights
        print(f"\n🎯 Personalization insights for: {query}")
        insights = user_rag.get_personalization_insights(test_user_id, query)
        print(f"Insights: {insights}")

if __name__ == "__main__":
    test_user_profile_rag() 