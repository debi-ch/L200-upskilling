#!/usr/bin/env python3
"""
Generate sample user profile data for testing Memory RAG functionality.
Uses Gemini to create realistic user profiles with travel preferences and history.
"""

import json
import sys
import os
from pathlib import Path

# Add the project root to sys.path
APPLICATION_ROOT = os.path.abspath(os.path.dirname(__file__))
if APPLICATION_ROOT not in sys.path:
    sys.path.insert(0, APPLICATION_ROOT)

from app.backend.models.gemini_chat import chat_with_gemini

def generate_user_profiles(num_users=10):
    """Generate sample user profiles using Gemini"""
    
    prompt = """Generate a realistic user profile for a travel chatbot user. Include the following information in JSON format:

{
  "user_id": "unique_identifier",
  "name": "Full Name",
  "age": 25-65,
  "location": "City, Country",
  "travel_preferences": {
    "budget_range": "budget/mid-range/luxury",
    "travel_style": "adventure/cultural/relaxation/business",
    "accommodation_type": "hotel/hostel/airbnb/resort",
    "group_size": "solo/couple/family/group",
    "interests": ["list", "of", "interests"]
  },
  "past_trips": [
    {
      "destination": "City, Country",
      "year": 2020-2024,
      "purpose": "vacation/business/family",
      "highlights": ["memorable experiences"],
      "accommodation": "where they stayed",
      "duration": "X days"
    }
  ],
  "dietary_restrictions": ["if any"],
  "languages": ["languages spoken"],
  "accessibility_needs": "if any",
  "travel_goals": ["future travel aspirations"],
  "personality_traits": ["helpful for personalization"]
}

Make this person realistic and diverse. Include 2-4 past trips. Vary the demographics, preferences, and experiences."""

    users = []
    
    for i in range(num_users):
        print(f"Generating user profile {i+1}/{num_users}...")
        
        try:
            response = chat_with_gemini(prompt)
            
            # Extract JSON from response
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            user_data = json.loads(response_text.strip())
            user_data['user_id'] = f"user_{i+1:03d}"
            users.append(user_data)
            
        except Exception as e:
            print(f"Error generating user {i+1}: {e}")
            continue
    
    return users

def save_user_profiles(users, filename="data/user_profiles/sample_users.json"):
    """Save user profiles to file"""
    
    # Create directory if it doesn't exist
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(users, f, indent=2)
    
    print(f"Saved {len(users)} user profiles to {filename}")

def generate_chat_history_for_user(user_profile, num_conversations=3):
    """Generate sample chat history for a user based on their profile"""
    
    user_context = f"""
User Profile:
- Name: {user_profile['name']}
- Age: {user_profile['age']}
- Location: {user_profile['location']}
- Travel Style: {user_profile['travel_preferences']['travel_style']}
- Budget: {user_profile['travel_preferences']['budget_range']}
- Interests: {', '.join(user_profile['travel_preferences']['interests'])}
- Past Trips: {', '.join([trip['destination'] for trip in user_profile['past_trips']])}
"""

    prompt = f"""Based on this user profile, generate {num_conversations} realistic chat conversations they might have with a travel assistant chatbot. Each conversation should be 3-5 exchanges (user question, bot response, user follow-up, etc.).

{user_context}

Format as JSON:
{{
  "conversations": [
    {{
      "conversation_id": "conv_001",
      "date": "2024-XX-XX",
      "messages": [
        {{"role": "user", "content": "user message", "timestamp": "2024-XX-XX HH:MM:SS"}},
        {{"role": "assistant", "content": "bot response", "timestamp": "2024-XX-XX HH:MM:SS"}}
      ]
    }}
  ]
}}

Make the conversations reflect their interests, past experiences, and travel style. Include questions about destinations, planning, recommendations, etc."""

    try:
        response = chat_with_gemini(prompt)
        
        # Extract JSON from response
        response_text = response.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        chat_data = json.loads(response_text.strip())
        return chat_data['conversations']
        
    except Exception as e:
        print(f"Error generating chat history for {user_profile['name']}: {e}")
        return []

def main():
    """Main function to generate sample data"""
    
    print("🤖 Generating sample user profiles with Gemini...")
    
    # Generate user profiles
    users = generate_user_profiles(10)
    
    if not users:
        print("❌ Failed to generate user profiles")
        return
    
    # Save user profiles
    save_user_profiles(users)
    
    # Generate chat histories for each user
    print("\n💬 Generating sample chat histories...")
    
    all_chat_data = {}
    
    for user in users:
        print(f"Generating chat history for {user['name']}...")
        conversations = generate_chat_history_for_user(user)
        
        if conversations:
            all_chat_data[user['user_id']] = {
                'user_profile': user,
                'conversations': conversations
            }
    
    # Save chat histories
    chat_filename = "data/user_profiles/sample_chat_histories.json"
    Path(chat_filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(chat_filename, 'w') as f:
        json.dump(all_chat_data, f, indent=2)
    
    print(f"\n✅ Generated data for {len(users)} users")
    print(f"📁 User profiles saved to: data/user_profiles/sample_users.json")
    print(f"💬 Chat histories saved to: {chat_filename}")
    print("\nNext steps:")
    print("1. Review the generated data")
    print("2. Run the memory RAG processor to create embeddings")
    print("3. Test the memory-enhanced chatbot")

if __name__ == "__main__":
    main() 