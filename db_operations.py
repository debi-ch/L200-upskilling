from google.cloud import firestore
import time
import uuid

class ChatDatabase:
    def __init__(self):
        self.db = firestore.Client()
        self.chats_collection = self.db.collection('chat_histories')

    def create_session(self):
        """Create a new chat session and return its ID"""
        session_id = str(uuid.uuid4())
        session_ref = self.chats_collection.document(session_id)
        session_ref.set({
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'messages': []
        })
        return session_id

    def save_message(self, session_id, role, content):
        """Save a message to the chat history"""
        session_ref = self.chats_collection.document(session_id)
        
        # Get current messages
        session = session_ref.get()
        if session.exists:
            current_messages = session.to_dict().get('messages', [])
        else:
            current_messages = []

        # Add new message
        current_messages.append({
            'role': role,
            'content': content,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })

        # Update document
        session_ref.update({
            'messages': current_messages,
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        })

    def get_chat_history(self, session_id):
        """Retrieve chat history for a session"""
        session_ref = self.chats_collection.document(session_id)
        session = session_ref.get()
        if session.exists:
            return session.to_dict().get('messages', [])
        return []

    def list_sessions(self, limit=10):
        """List recent chat sessions"""
        sessions = self.chats_collection.order_by(
            'updated_at', direction=firestore.Query.DESCENDING
        ).limit(limit).stream()
        return [(doc.id, doc.to_dict()) for doc in sessions] 