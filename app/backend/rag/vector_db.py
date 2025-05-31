"""
Vector Database Interface - Capable of Creating Infrastructure

This module provides an interface to Vertex AI Vector Search. It can connect to
existing resources or create them if they don't exist, configured for stream updates.
"""

from typing import List, Dict, Any, Optional
import json
import time
import os
import traceback # Added for direct traceback printing

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct
from google.api_core import exceptions as google_exceptions # For more specific exception handling

from app.backend.rag.config import VectorDBConfig, EmbeddingModelConfig
from app.utils.logging_utils import ChatbotLogger
from app.backend.rag.embedding_service import embedding_service

logger = ChatbotLogger("rag_vector_db")

class VectorSearchDB:
    def __init__(self, index_id: Optional[str] = None, dimensions: Optional[int] = None, description: Optional[str] = None):
        self.project_id = EmbeddingModelConfig.PROJECT_ID
        self.location = EmbeddingModelConfig.LOCATION
        self.index_display_name = index_id or VectorDBConfig.VECTOR_SEARCH_INDEX_DISPLAY_NAME
        self.endpoint_display_name = VectorDBConfig.VECTOR_SEARCH_ENDPOINT_DISPLAY_NAME
        self.deployed_index_id = f"{self.index_display_name}_deploy"  # Unique deployment ID for each index
        self.dimensions = dimensions or VectorDBConfig.EMBEDDING_DIMENSIONS
        self.description = description or "Vector search index for RAG system"

        self.index: Optional[aiplatform.MatchingEngineIndex] = None
        self.index_endpoint: Optional[aiplatform.MatchingEngineIndexEndpoint] = None
        self.initialized = False
        
        logger.info(
            "VectorSearchDB configured:",
            project_id=self.project_id,
            location=self.location,
            index_display_name=self.index_display_name,
            endpoint_display_name=self.endpoint_display_name,
            deployed_index_id=self.deployed_index_id,
            dimensions=self.dimensions,
            description=self.description
        )
    
    def initialize(self) -> bool:
        if self.initialized:
            logger.info("VectorSearchDB already initialized.")
            return True
        try:
            logger.info(f"Initializing Vertex AI SDK for project: {self.project_id}, location: {self.location}")
            aiplatform.init(project=self.project_id, location=self.location)
            
            self._get_or_create_index()
            self._get_or_create_endpoint() # This populates self.index_endpoint

            if not self.index_endpoint:
                logger.error("IndexEndpoint object was not created or retrieved by _get_or_create_endpoint. Cannot proceed.")
                raise ValueError("IndexEndpoint could not be established.")
            
            # Log public domain name for info, but don't re-initialize based on it here.
            if hasattr(self.index_endpoint, 'public_endpoint_domain_name') and self.index_endpoint.public_endpoint_domain_name:
                logger.info(f"Endpoint public domain name: {self.index_endpoint.public_endpoint_domain_name}")
            else:
                logger.info("Endpoint does not have public_endpoint_domain_name or is private.")

            self._deploy_index_to_endpoint()
            
            self.initialized = True
            logger.info("VectorSearchDB initialized successfully (infrastructure checked/created/deployed).")
            return True
        except Exception as e:
            logger.error(
                "Failed to initialize VectorSearchDB (infrastructure setup failed)",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            print("--- TRACEBACK START (direct print for initialize failure) ---")
            # import traceback # Already at top of file
            print(traceback.format_exc())
            print("--- TRACEBACK END (direct print for initialize failure) ---")
            raise 

    def _get_or_create_index(self) -> None:
        try:
            indexes = aiplatform.MatchingEngineIndex.list(location=self.location, project=self.project_id)
            found_indexes = [idx for idx in indexes if idx.display_name == self.index_display_name]
            if found_indexes:
                self.index = found_indexes[0]
                logger.info(f"Using existing Vector Search index: {self.index.display_name} ({self.index.resource_name})")
                return
            else:
                logger.info(f"Index with display name '{self.index_display_name}' not found. Creating new index.")
                raise google_exceptions.NotFound("Index not found by display name, proceeding to create.")
        except google_exceptions.NotFound:
            logger.info(f"Creating new Vector Search index '{self.index_display_name}' with Stream Updates.")
            self.index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=self.index_display_name,
                dimensions=self.dimensions,
                approximate_neighbors_count=VectorDBConfig.APPROXIMATE_NEIGHBORS_COUNT,
                distance_measure_type=VectorDBConfig.SIMILARITY_MEASURE,
                leaf_node_embedding_count=VectorDBConfig.LEAF_NODE_EMBEDDING_COUNT,
                leaf_nodes_to_search_percent=VectorDBConfig.LEAF_NODES_TO_SEARCH_PERCENT,
                project=self.project_id,
                location=self.location,
                index_update_method="STREAM_UPDATE",
                description=self.description,
                sync=True
            )
            logger.info(
                f"Created new Vector Search index: {self.index.display_name} ({self.index.resource_name}) with stream updates enabled."
            )
        except Exception as e:
            logger.error(f"Error in _get_or_create_index for '{self.index_display_name}': {e}", exc_info=True)
            raise
    
    def _get_or_create_endpoint(self) -> None:
        try:
            endpoints = aiplatform.MatchingEngineIndexEndpoint.list(location=self.location, project=self.project_id)
            found_endpoints = [ep for ep in endpoints if ep.display_name == self.endpoint_display_name]
            if found_endpoints:
                # Re-instantiate from resource name to ensure it's fully formed with current config
                ep_resource_name = found_endpoints[0].resource_name
                logger.info(f"Found existing Vector Search endpoint: {found_endpoints[0].display_name} ({ep_resource_name}). Re-instantiating with explicit credentials.")
                self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                    index_endpoint_name=ep_resource_name,
                    project=self.project_id,
                    location=self.location,
                    credentials=aiplatform.initializer.global_config.credentials
                )
                logger.info(f"Re-instantiated existing endpoint: {self.index_endpoint.display_name}")
                return
            else:
                logger.info(f"Endpoint with display name '{self.endpoint_display_name}' not found. Creating new public endpoint.")
                self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                    display_name=self.endpoint_display_name,
                    project=self.project_id,
                    location=self.location,
                    public_endpoint_enabled=True,
                    credentials=aiplatform.initializer.global_config.credentials, # Add credentials here
                    sync=True
                )
                logger.info(f"Created new Vector Search endpoint: {self.index_endpoint.display_name} ({self.index_endpoint.resource_name})")
        except Exception as e:
            logger.error(f"Error in _get_or_create_endpoint for '{self.endpoint_display_name}': {e}", exc_info=True)
            raise

    def _deploy_index_to_endpoint(self) -> None:
        logger.info(f"Entering _deploy_index_to_endpoint. Type of self.index: {type(self.index)}, Value: {self.index}")
        logger.info(f"Type of self.index_endpoint: {type(self.index_endpoint)}, Value: {self.index_endpoint}")

        if not self.index or not self.index_endpoint:
            logger.error("Cannot deploy index: Index or Endpoint not available.")
            if not self.index: logger.error("self.index is None or Falsy.")
            if not self.index_endpoint: logger.error("self.index_endpoint is None or Falsy.")
            raise ValueError("Index and Endpoint must be valid objects before deploying.")
        
        if not isinstance(self.index, aiplatform.MatchingEngineIndex):
            logger.error(f"CRITICAL: self.index is not a MatchingEngineIndex object. Type: {type(self.index)}.")
            raise TypeError(f"self.index expected MatchingEngineIndex, got {type(self.index)}")

        try:
            for deployed_index_obj in self.index_endpoint.deployed_indexes:
                if deployed_index_obj.id == self.deployed_index_id and deployed_index_obj.index == self.index.resource_name:
                    logger.info(f"Index '{self.index.display_name}' (as '{self.deployed_index_id}') already deployed.")
                    return
                elif deployed_index_obj.id == self.deployed_index_id: # Conflicting deployment ID
                    logger.warning(f"Deployment ID '{self.deployed_index_id}' on endpoint points to different index ({deployed_index_obj.index}). Undeploying.")
                    self.index_endpoint.undeploy_index(deployed_index_id=self.deployed_index_id)
                    logger.info(f"Undeployed conflicting ID '{self.deployed_index_id}'.")
                    break 
            
            logger.info(f"Deploying index '{self.index.display_name}' ({self.index.resource_name}) to endpoint '{self.index_endpoint.display_name}' as '{self.deployed_index_id}'")
            self.index_endpoint.deploy_index(index=self.index, deployed_index_id=self.deployed_index_id)
            logger.info(f"Deployment initiated for index '{self.index.display_name}' as '{self.deployed_index_id}'. Async operation.")
        except Exception as e:
            logger.error(f"Failed to deploy index '{self.index.display_name}': {e}", exc_info=True)
            raise

    def _get_index_handle(self) -> Optional[aiplatform.MatchingEngineIndex]:
        if self.index: 
            return self.index
        logger.warning("_get_index_handle called when self.index is None. Re-attempting _get_or_create_index.")
        try:
            self._get_or_create_index()
            return self.index
        except: 
            return None

    def upsert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        index_handle = self._get_index_handle()
        if not index_handle:
            logger.error("Cannot upsert documents: Index handle not available.")
            return []
        if not documents:
            logger.warning("No documents provided for upsert.")
            return []

        datapoints_to_upsert = []
        successful_ids = []
        for doc in documents:
            if 'id' not in doc or 'embedding' not in doc or not doc['embedding']:
                logger.warning(f"Document missing id or embedding, skipping: {doc.get('id', 'N/A')}")
                continue
            
            # REVERTED: Pass an empty Struct for restricts to ensure upsert works.
            # Metadata will not be stored in Vector Search 'restricts' field via this method.
            empty_restricts_struct = Struct() 
            
            datapoints_to_upsert.append({
                "datapoint_id": str(doc['id']),
                "feature_vector": doc['embedding'],
                "restricts": empty_restricts_struct # Pass the empty struct
            })
            successful_ids.append(str(doc['id']))

        if not datapoints_to_upsert:
            logger.warning("No valid datapoints to upsert after filtering.")
            return []

        try:
            logger.info(f"Upserting {len(datapoints_to_upsert)} datapoints to index {index_handle.display_name} ({index_handle.resource_name})")
            index_handle.upsert_datapoints(datapoints=datapoints_to_upsert)
            logger.info(f"Successfully upserted {len(successful_ids)} documents.")
            return successful_ids
        except Exception as e:
            logger.error(
                f"Failed to upsert datapoints to index {index_handle.display_name}",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            print("--- TRACEBACK START (direct print for upsert) ---")
            print(traceback.format_exc())
            print("--- TRACEBACK END (direct print for upsert) ---")
            raise 

    def search(self, query_text: Optional[str] = None, query_embedding: Optional[List[float]] = None, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search using either text or a pre-computed embedding"""
        logger.info(f"Performing search with top_k={top_k}")
        
        if not self.initialized:
            if not self.initialize():
                logger.error("Failed to initialize vector database for search")
                return []

        try:
            if query_embedding is None and query_text:
                # Generate embedding from text
                query_embedding = embedding_service.generate_embedding(query_text)
                if not query_embedding:
                    logger.error(f"Failed to generate embedding for query: {query_text[:100]}...")
                    return []
            elif query_embedding is None:
                logger.error("Neither query_text nor query_embedding provided")
                return []

            # Perform nearest neighbor search
            search_response = self.index_endpoint.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=[query_embedding],
                num_neighbors=top_k or VectorDBConfig.DEFAULT_TOP_K
            )
            
            results = []
            if search_response and search_response[0]:
                for neighbor in search_response[0]:
                    restrictions_dict = {}
                    if hasattr(neighbor, 'datapoint') and neighbor.datapoint and \
                       hasattr(neighbor.datapoint, 'restricts') and neighbor.datapoint.restricts:
                        try:
                            restrictions_dict = json_format.MessageToDict(neighbor.datapoint.restricts)
                        except Exception as e:
                            logger.warning(f"Could not parse restricts for datapoint {neighbor.id}: {e}")
                    
                    result = {
                        'id': neighbor.id, 
                        'score': neighbor.distance, 
                        'text': "", 
                        'metadata': restrictions_dict
                    }
                    results.append(result)
            
            logger.info(f"Search completed. Found {len(results)} results.")
            return results
        except Exception as e:
            logger.error(
                "Error during vector search operation",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            print("--- TRACEBACK START (direct print for search failure) ---")
            print(traceback.format_exc())
            print("--- TRACEBACK END (direct print for search failure) ---")
            raise

# vector_db = VectorSearchDB() # Singleton instantiation can be done here or in RAGEngine 