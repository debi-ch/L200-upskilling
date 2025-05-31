"""
Memory Processor for User Profile and Chat History RAG
Extracts and processes user context for embedding generation and retrieval.
"""

import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re

class MemoryProcessor:
    """Processes user profiles and chat history for memory-based RAG"""
    
    def __init__(self):
        self.user_profiles = {}
        self.chat_histories = {}
    
    def load_user_profiles(self, profiles_file: str) -> None:
        """Load user profiles from JSON file"""
        try:
            with open(profiles_file, 'r') as f:
                profiles = json.load(f)
            
            for profile in profiles:
                self.user_profiles[profile['user_id']] = profile
            
            print(f"Loaded {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            print(f"Error loading user profiles: {e}")
    
    def load_chat_histories(self, chat_file: str) -> None:
        """Load chat histories from JSON file"""
        try:
            with open(chat_file, 'r') as f:
                self.chat_histories = json.load(f)
            
            print(f"Loaded chat histories for {len(self.chat_histories)} users")
            
        except Exception as e:
            print(f"Error loading chat histories: {e}")
    
    def extract_user_preferences(self, user_id: str) -> List[Dict[str, Any]]:
        """Extract user preferences as embeddable chunks"""
        
        if user_id not in self.user_profiles:
            return []
        
        profile = self.user_profiles[user_id]
        chunks = []
        
        # Basic profile chunk
        basic_info = f"""
User: {profile['name']}
Age: {profile['age']}
Location: {profile['location']}
Languages: {', '.join(profile.get('languages', []))}
"""
        
        chunks.append({
            'chunk_id': f"{user_id}_basic_profile",
            'content': basic_info.strip(),
            'metadata': {
                'user_id': user_id,
                'type': 'basic_profile',
                'name': profile['name']
            }
        })
        
        # Travel preferences chunk
        prefs = profile.get('travel_preferences', {})
        pref_text = f"""
Travel Preferences for {profile['name']}:
- Budget Range: {prefs.get('budget_range', 'Not specified')}
- Travel Style: {prefs.get('travel_style', 'Not specified')}
- Accommodation Type: {prefs.get('accommodation_type', 'Not specified')}
- Group Size: {prefs.get('group_size', 'Not specified')}
- Interests: {', '.join(prefs.get('interests', []))}
"""
        
        chunks.append({
            'chunk_id': f"{user_id}_travel_preferences",
            'content': pref_text.strip(),
            'metadata': {
                'user_id': user_id,
                'type': 'travel_preferences',
                'name': profile['name'],
                'budget_range': prefs.get('budget_range'),
                'travel_style': prefs.get('travel_style')
            }
        })
        
        # Past trips chunks
        for i, trip in enumerate(profile.get('past_trips', [])):
            trip_text = f"""
Past Trip by {profile['name']}:
Destination: {trip.get('destination', 'Unknown')}
Year: {trip.get('year', 'Unknown')}
Purpose: {trip.get('purpose', 'Unknown')}
Duration: {trip.get('duration', 'Unknown')}
Accommodation: {trip.get('accommodation', 'Unknown')}
Highlights: {', '.join(trip.get('highlights', []))}
"""
            
            chunks.append({
                'chunk_id': f"{user_id}_trip_{i+1}",
                'content': trip_text.strip(),
                'metadata': {
                    'user_id': user_id,
                    'type': 'past_trip',
                    'name': profile['name'],
                    'destination': trip.get('destination'),
                    'year': trip.get('year'),
                    'purpose': trip.get('purpose')
                }
            })
        
        # Special requirements chunk
        special_reqs = []
        if profile.get('dietary_restrictions'):
            special_reqs.append(f"Dietary restrictions: {', '.join(profile['dietary_restrictions'])}")
        if profile.get('accessibility_needs'):
            special_reqs.append(f"Accessibility needs: {profile['accessibility_needs']}")
        
        if special_reqs:
            special_text = f"""
Special Requirements for {profile['name']}:
{chr(10).join(special_reqs)}
"""
            
            chunks.append({
                'chunk_id': f"{user_id}_special_requirements",
                'content': special_text.strip(),
                'metadata': {
                    'user_id': user_id,
                    'type': 'special_requirements',
                    'name': profile['name']
                }
            })
        
        # Travel goals chunk
        if profile.get('travel_goals'):
            goals_text = f"""
Travel Goals for {profile['name']}:
{', '.join(profile['travel_goals'])}
"""
            
            chunks.append({
                'chunk_id': f"{user_id}_travel_goals",
                'content': goals_text.strip(),
                'metadata': {
                    'user_id': user_id,
                    'type': 'travel_goals',
                    'name': profile['name']
                }
            })
        
        return chunks
    
    def extract_conversation_insights(self, user_id: str, max_conversations: int = 5) -> List[Dict[str, Any]]:
        """Extract insights from recent conversations"""
        
        if user_id not in self.chat_histories:
            return []
        
        user_data = self.chat_histories[user_id]
        conversations = user_data.get('conversations', [])
        profile = user_data.get('user_profile', {})
        
        chunks = []
        
        # Process recent conversations
        recent_conversations = conversations[-max_conversations:] if conversations else []
        
        for i, conv in enumerate(recent_conversations):
            # Extract user questions and preferences from conversation
            user_messages = [msg for msg in conv.get('messages', []) if msg['role'] == 'user']
            assistant_messages = [msg for msg in conv.get('messages', []) if msg['role'] == 'assistant']
            
            if not user_messages:
                continue
            
            # Create conversation summary
            conv_summary = f"""
Recent Conversation with {profile.get('name', 'User')} (Date: {conv.get('date', 'Unknown')}):

User Questions/Interests:
{chr(10).join([f"- {msg['content'][:200]}..." if len(msg['content']) > 200 else f"- {msg['content']}" for msg in user_messages])}

Key Topics Discussed:
{self._extract_topics_from_conversation(conv)}
"""
            
            chunks.append({
                'chunk_id': f"{user_id}_conversation_{i+1}",
                'content': conv_summary.strip(),
                'metadata': {
                    'user_id': user_id,
                    'type': 'conversation_history',
                    'name': profile.get('name', 'User'),
                    'conversation_date': conv.get('date'),
                    'conversation_id': conv.get('conversation_id')
                }
            })
        
        return chunks
    
    def _extract_topics_from_conversation(self, conversation: Dict[str, Any]) -> str:
        """Extract key topics from a conversation using simple keyword analysis"""
        
        all_text = " ".join([msg['content'] for msg in conversation.get('messages', [])])
        
        # Common travel-related keywords
        travel_keywords = {
            'destinations': ['destination', 'city', 'country', 'place', 'location', 'visit'],
            'accommodation': ['hotel', 'hostel', 'airbnb', 'resort', 'stay', 'accommodation'],
            'activities': ['activity', 'tour', 'museum', 'restaurant', 'food', 'culture', 'adventure'],
            'planning': ['plan', 'budget', 'cost', 'time', 'duration', 'when', 'how'],
            'transportation': ['flight', 'train', 'bus', 'car', 'transport', 'travel']
        }
        
        topics_found = []
        text_lower = all_text.lower()
        
        for category, keywords in travel_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics_found.append(category)
        
        return ', '.join(topics_found) if topics_found else 'general travel discussion'
    
    def get_user_context_for_query(self, user_id: str, query: str) -> str:
        """Get relevant user context for a specific query"""
        
        if user_id not in self.user_profiles:
            return ""
        
        profile = self.user_profiles[user_id]
        context_parts = []
        
        # Always include basic info
        context_parts.append(f"User: {profile['name']} ({profile['age']} years old, from {profile['location']})")
        
        # Include relevant preferences based on query
        query_lower = query.lower()
        prefs = profile.get('travel_preferences', {})
        
        if any(word in query_lower for word in ['budget', 'cost', 'price', 'expensive', 'cheap']):
            context_parts.append(f"Budget preference: {prefs.get('budget_range', 'Not specified')}")
        
        if any(word in query_lower for word in ['hotel', 'stay', 'accommodation', 'where']):
            context_parts.append(f"Preferred accommodation: {prefs.get('accommodation_type', 'Not specified')}")
        
        if any(word in query_lower for word in ['activity', 'do', 'see', 'visit', 'experience']):
            interests = prefs.get('interests', [])
            if interests:
                context_parts.append(f"Interests: {', '.join(interests)}")
        
        # Include relevant past trips
        past_trips = profile.get('past_trips', [])
        if past_trips and any(word in query_lower for word in ['been', 'visited', 'before', 'similar']):
            recent_destinations = [trip['destination'] for trip in past_trips[-3:]]
            context_parts.append(f"Recent destinations: {', '.join(recent_destinations)}")
        
        return "\n".join(context_parts)
    
    def process_all_user_data(self) -> List[Dict[str, Any]]:
        """Process all user data into embeddable chunks"""
        
        all_chunks = []
        
        for user_id in self.user_profiles.keys():
            # Get preference chunks
            pref_chunks = self.extract_user_preferences(user_id)
            all_chunks.extend(pref_chunks)
            
            # Get conversation chunks
            conv_chunks = self.extract_conversation_insights(user_id)
            all_chunks.extend(conv_chunks)
        
        print(f"Processed {len(all_chunks)} memory chunks for {len(self.user_profiles)} users")
        return all_chunks
    
    def get_user_by_name(self, name: str) -> Optional[str]:
        """Get user_id by name (for testing purposes)"""
        
        for user_id, profile in self.user_profiles.items():
            if profile.get('name', '').lower() == name.lower():
                return user_id
        return None 