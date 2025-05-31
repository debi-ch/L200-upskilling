# Chatbot Project Status

Current Status:
- Chatbot deployed on Cloud Run
- Dual model support:
  - Gemini 2.0 Flash
  - Gemma 9B-IT (9 billion parameters, instruction-tuned)
- Using Vertex AI for model integration
- Using Cloud Build for deployments
- Using Artifact Registry for container images
- Advanced prompt management system with versioning
- Comprehensive model evaluation framework
- Retrieval Augmented Generation (RAG) for travel information, connected to a Vertex AI Vector Search index.

Features:
- Web-based chat interface with Streamlit
- Dynamic model switching between Gemini and Gemma
- Persistent chat history with Firestore
- Cloud logging and monitoring
- Session management
- Prompt version control and management
- Specialized travel agent personas:
  - Gemini: Local-focused cultural guide
  - Gemma: Mexican pirate character
- Model performance evaluation and comparison tools
- RAG integration for Gemini model:
  - Processes NDJSON travel data (hotel information for Buenos Aires).
  - Generates embeddings using 'text-embedding-005'.
  - Upserts embeddings to a pre-configured Vertex AI Vector Search index (Tree-AH, Cosine Distance, Stream Updates enabled).
  - Retrieves relevant text chunks and associated metadata based on semantic similarity to user queries.
  - Augments prompts for the Gemini LLM ('gemini-2.0-flash') with retrieved context.
  - Provides contextually aware answers based on the travel documents.
  - In-memory map for quick retrieval of original text and metadata for chunks.
  - UI toggle in Streamlit app to enable/disable RAG mode for Gemini.
- RAG Evaluation System:
  - Custom metrics without requiring external LLM APIs
  - Answer Presence checking
  - Context Utilization measurement
  - Response Length optimization
  - Response Time tracking
  - Comprehensive test suite for all RAG components

To get started:
1. Activate virtual environment: 
   ```bash
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize default prompts:
   ```bash
   python init_prompts.py
   ```

4. Run app locally: 
   ```bash
   # Ensure you are in the L200-upskilling directory
   streamlit run app/frontend/chatbot_app.py
   ```
   (Select Gemini model and enable the "Enable RAG" checkbox in the sidebar to test RAG functionality)

5. Deploy changes using Cloud Build and Cloud Run: 
   ```bash
   # Deploy using the deployment script
   ./deploy_cloud_run.sh
   ```

6. Evaluate model performance:
   ```bash
   # Navigate to evaluation directory
   cd evaluation
   
   # Run complete evaluation process
   ./run_evaluation_lab.sh
   
   # Or run individual evaluations
   ./evaluate_models.sh pointwise --model gemini
   ./evaluate_models.sh pairwise
   ```

7. Test RAG module directly (optional):
   ```bash
   # Ensure you are in the L200-upskilling directory (parent of 'app')
   # This script will process data and run test queries against the RAG system
   python -m app.backend.models.gemini_rag
   ```

8. Run RAG evaluation:
   ```bash
   # Run the comprehensive RAG evaluation suite
   python -m test_rag_evaluation
   ```

Local Development:
- All code is in L200-upskilling directory
- Using virtual environment for Python dependencies
- Streamlit for web interface
- Firestore for chat history
- Cloud Logging for monitoring

Dependencies:
- google-cloud-aiplatform (ensure version >= 1.49.0 for stable Vector Search and Gemini 1.5+ features)
- google-genai (consider for future Gemini 1.5+ calls if `google-cloud-aiplatform` has issues with newest models)
- google-cloud-logging>=3.8.0
- google-cloud-firestore
- streamlit
- pandas and matplotlib (for evaluation)
- torch and torchvision (for image processing)
- transformers (for multimodal embeddings)

Prompt Management:
- Support for multiple prompt versions per model
- UI-based prompt version control
- Ability to view prompt history
- Easy switching between different prompts
- Specialized travel agent prompts for each model

Model Evaluation Framework:
- Pointwise evaluation of individual models
- Pairwise comparison between Gemini and Gemma
- Comprehensive metrics:
  - Response time and length
  - Quality scoring based on content and structure
  - Travel-specific domain metrics
  - Response specificity and detail level
- Visualization tools for performance analysis
- Customizable test queries and evaluation criteria
- Lab guide for running evaluations and interpreting results

RAG System Components:
1. Text RAG:
   - Chat history and user preferences integration
   - Context-aware responses
   - Personalized information retrieval

2. PDF RAG:
   - PDF document processing and chunking
   - Metadata extraction and storage
   - Semantic search capabilities
   - Tested with travel guides

3. Multimodal RAG:
   - Image processing and embedding
   - Combined text-image queries
   - Visual information retrieval
   - Travel image understanding

4. RAG Evaluation:
   - Custom metrics suite
   - No external API dependencies
   - Performance monitoring
   - Quality assurance checks

Documentation:
- `README.md`: Main project documentation (this file)
- `evaluation/README.md`: Technical overview of the evaluation pipeline
- `evaluation/LAB_OVERVIEW.md`: Guide to the evaluation lab materials
- `evaluation/MODEL_EVALUATION_LAB.md`: Step-by-step lab guide for running evaluations
- `evaluation/QUERY_CREATION_GUIDELINES.md`: Guidelines for creating effective test queries
- `evaluation/LAB_README.md`: Instructions for lab instructors

Recent Updates:
- Implemented comprehensive RAG evaluation system
- Added custom metrics without external API dependencies
- Updated deployment process to use Cloud Build
- Enhanced RAG system with PDF and image processing
- Added multimodal search capabilities
- Implemented evaluation framework for all RAG components
- Updated deployment scripts and documentation

## Deployment
The chatbot is currently deployed and accessible at:
https://rag-chatbot-708208532564.us-central1.run.app

