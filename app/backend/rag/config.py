"""
RAG Configuration
"""
import os
from app.config import GCP_PROJECT_ID, GCP_LOCATION # Assuming these are correctly set in app.config

class VectorDBConfig:
    # Vertex AI Vector Search settings
    # Display name of your index (used by the code to try and get an aiplatform.MatchingEngineIndex object)
    VECTOR_SEARCH_INDEX_DISPLAY_NAME = "travel_documents_index"
    # Actual Resource ID of your Index (from the successful ingestion run)
    VECTOR_SEARCH_INDEX_RESOURCE_ID = "514151428357357568"

    # Actual Resource ID of your Index Endpoint (from the successful ingestion run)
    VECTOR_SEARCH_ENDPOINT_RESOURCE_ID = "5284202305556578304"
    # Display name for the endpoint, if your code uses it to get the endpoint object. 
    # If the code uses the resource ID directly, this might be just for reference.
    VECTOR_SEARCH_ENDPOINT_DISPLAY_NAME = "travel_index_endpoint"

    # Actual ID of the Deployed Index on the Endpoint (from console)
    VECTOR_SEARCH_DEPLOYED_INDEX_ID = "travel_index_deploy_v2"
    
    # Vector dimensions for the embedding model
    EMBEDDING_DIMENSIONS = 768  # for text-embedding-gecko
    
    # Retrieve top-k results from vector search
    DEFAULT_TOP_K = 5
    
    # Similarity metric and ANN algorithm parameters
    # Ensure this matches how your existing index 'travel_documents_index' was configured
    SIMILARITY_MEASURE = "COSINE_DISTANCE" 
    APPROXIMATE_NEIGHBORS_COUNT = 10 # Should match existing index config
    LEAF_NODE_EMBEDDING_COUNT = 500  # Should match existing index config
    LEAF_NODES_TO_SEARCH_PERCENT = 7 # Should match existing index config

class UserProfileVectorDBConfig:
    """Configuration for user profile and memory vector database"""
    
    # For now, we'll use the same index as travel documents but with different metadata
    # In production, you might want a separate index for user profiles
    USER_PROFILE_INDEX_DISPLAY_NAME = "user_profiles_index"
    USER_PROFILE_INDEX_RESOURCE_ID = "514151428357357568"  # Same as travel for now
    
    USER_PROFILE_ENDPOINT_RESOURCE_ID = "5284202305556578304"  # Same as travel for now
    USER_PROFILE_ENDPOINT_DISPLAY_NAME = "user_profiles_endpoint"
    
    USER_PROFILE_DEPLOYED_INDEX_ID = "travel_index_deploy_v2"  # Same as travel for now
    
    # Memory-specific settings
    MAX_USER_CONTEXT_CHUNKS = 5
    MAX_CONVERSATION_HISTORY = 10
    MEMORY_CHUNK_SIZE = 500  # Smaller chunks for user data
    
    # User data processing
    ENABLE_REAL_TIME_MEMORY = True
    MEMORY_RETENTION_DAYS = 90

class DocumentProcessingConfig:
    CHUNK_SIZE = 1000  # Max characters per chunk
    CHUNK_OVERLAP = 200 # Characters of overlap between chunks
    # Key fields to extract from the source NDJSON for metadata
    METADATA_FIELDS = [
        "hotel_name", 
        "hotel_address", 
        "nearest_attractions" 
        # Add other relevant fields from your NDJSON like 'property_type', etc.
    ]
    # Input/Output settings (can be useful if you save processed chunks)
    # NDJSON_INPUT_DIR = os.environ.get(
    #     "NDJSON_INPUT_DIR", 
    #     os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "travel") # Adjust path if needed
    # )
    # PROCESSED_CHUNKS_DIR = os.environ.get(
    #     "PROCESSED_CHUNKS_DIR", 
    #     os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed") # Adjust path if needed
    # )

class RAGPromptTemplates:
    # Basic template for augmenting a prompt with retrieved context
    CONTEXT_PROMPT_TEMPLATE = """
You are an expert and friendly AI travel assistant specializing in Buenos Aires.
Your knowledge about specific hotels, their features, and locations comes ONLY from the following retrieved context from local travel documents.

Please answer the user's query based on this context.

- If the context provides specific names (e.g., hotel names from the 'From hotel ...' prefix in the context), please try to use them in your answer.
- If the context contains relevant information to answer the query, synthesize it into a helpful and concise response.
- If the context does not provide enough information to directly answer the query, clearly state that the provided documents don't have the specific details. You may then offer a more general travel-related answer if appropriate, but clearly distinguish this general advice from information found in the documents.
- Do not make up information or details not present in the provided context.

Context from travel documents:
---------------------
{context}
---------------------

User's Query: {query}

Assistant's Answer:"""

    # Template for memory-enhanced responses
    MEMORY_ENHANCED_PROMPT_TEMPLATE = """
You are an expert and friendly AI travel assistant. You have access to both general travel information and specific information about this user.

User Context:
{user_context}

Travel Information Context:
{travel_context}

Please provide a personalized response to the user's query that takes into account:
1. Their travel preferences and past experiences
2. Any relevant travel information from your knowledge base
3. Their specific interests and requirements

Be conversational and reference their past experiences or preferences when relevant, but don't be overly familiar.

User's Query: {query}

Assistant's Answer:"""

    # Template for when only user context is available
    USER_CONTEXT_ONLY_TEMPLATE = """
You are an expert and friendly AI travel assistant. Based on what I know about this user, here's some context:

User Context:
{user_context}

Please provide a personalized response to their query that takes into account their preferences, past experiences, and interests.

User's Query: {query}

Assistant's Answer:"""

class EmbeddingModelConfig:
    # Google's embedding model for text
    MODEL_NAME = "text-embedding-005" # Changed from text-embedding-gecko to a valid, versioned model
    
    # Project and region
    PROJECT_ID = GCP_PROJECT_ID # From app.config
    LOCATION = GCP_LOCATION     # From app.config
    API_ENDPOINT = f"{GCP_LOCATION}-aiplatform.googleapis.com"
    TIMEOUT_SECONDS = 30

class RAGConfig:
    """Main RAG configuration class that combines all settings"""
    
    def __init__(self):
        # Core settings
        self.project_id = GCP_PROJECT_ID
        self.location = GCP_LOCATION
        
        # Travel documents vector DB
        self.index_id = VectorDBConfig.VECTOR_SEARCH_INDEX_RESOURCE_ID
        self.endpoint_id = VectorDBConfig.VECTOR_SEARCH_ENDPOINT_RESOURCE_ID
        self.deployed_index_id = VectorDBConfig.VECTOR_SEARCH_DEPLOYED_INDEX_ID
        
        # User profile vector DB (using same index for now)
        self.user_profile_index_id = UserProfileVectorDBConfig.USER_PROFILE_INDEX_RESOURCE_ID
        self.user_profile_endpoint_id = UserProfileVectorDBConfig.USER_PROFILE_ENDPOINT_RESOURCE_ID
        self.user_profile_deployed_index_id = UserProfileVectorDBConfig.USER_PROFILE_DEPLOYED_INDEX_ID
        
        # Embedding settings
        self.embedding_model = EmbeddingModelConfig.MODEL_NAME
        self.embedding_dimensions = VectorDBConfig.EMBEDDING_DIMENSIONS
        
        # Processing settings
        self.chunk_size = DocumentProcessingConfig.CHUNK_SIZE
        self.chunk_overlap = DocumentProcessingConfig.CHUNK_OVERLAP
        self.top_k = VectorDBConfig.DEFAULT_TOP_K
        
        # Memory settings
        self.max_user_context_chunks = UserProfileVectorDBConfig.MAX_USER_CONTEXT_CHUNKS
        self.max_conversation_history = UserProfileVectorDBConfig.MAX_CONVERSATION_HISTORY
        self.memory_chunk_size = UserProfileVectorDBConfig.MEMORY_CHUNK_SIZE
        
        # Prompt templates
        self.context_prompt_template = RAGPromptTemplates.CONTEXT_PROMPT_TEMPLATE
        self.memory_enhanced_template = RAGPromptTemplates.MEMORY_ENHANCED_PROMPT_TEMPLATE
        self.user_context_only_template = RAGPromptTemplates.USER_CONTEXT_ONLY_TEMPLATE

# Other configs like DocumentProcessingConfig, RAGPromptTemplates can be added later 