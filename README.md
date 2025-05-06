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
   streamlit run chatbot_app.py
   ```

5. Deploy changes using Cloud Build and Cloud Run: 
   ```bash
   # Build and submit to Artifact Registry
   gcloud builds submit --tag gcr.io/learningemini/chatbot-app

   # Deploy to Cloud Run
   gcloud run deploy chatbot-app \
     --image gcr.io/learningemini/chatbot-app \
     --platform managed \
     --region us-west1 \
     --allow-unauthenticated
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

Local Development:
- All code is in L200-upskilling directory
- Using virtual environment for Python dependencies
- Streamlit for web interface
- Firestore for chat history
- Cloud Logging for monitoring

Dependencies:
- google-cloud-aiplatform
- google-cloud-logging>=3.8.0
- google-cloud-firestore
- streamlit
- google-genai
- pandas and matplotlib (for evaluation)

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

Documentation:
- `README.md`: Main project documentation (this file)
- `evaluation/README.md`: Technical overview of the evaluation pipeline
- `evaluation/LAB_OVERVIEW.md`: Guide to the evaluation lab materials
- `evaluation/MODEL_EVALUATION_LAB.md`: Step-by-step lab guide for running evaluations
- `evaluation/QUERY_CREATION_GUIDELINES.md`: Guidelines for creating effective test queries
- `evaluation/LAB_README.md`: Instructions for lab instructors

Recent Updates:
- Migrated to Vertex AI SDK
- Implemented prompt version control system
- Added specialized travel agent personas
- Enhanced error handling and debugging
- Added support for both Gemini and Gemma models
- Updated deployment process for new dependencies
- Implemented comprehensive model evaluation framework
- Created lab materials for model evaluation training

## Deployment
The chatbot is currently deployed and accessible at:
https://chatbot-app-708208532564.us-central1.run.app

## Next Steps (Week 3)

### Model Fine-tuning and Deployment

1. **Fine-tune Gemini Model with OpenAssistant Guanaco Dataset**
   - Set up fine-tuning environment with Vertex AI
   - Acquire and preprocess the OpenAssistant Guanaco dataset from HuggingFace
   - Format the dataset for Gemini fine-tuning requirements
   - Configure training hyperparameters for optimal learning
   - Execute fine-tuning process on Vertex AI
   - Monitor training metrics and early stopping criteria
   - Validate model performance with test set

2. **Model Registry and Deployment**
   - Save the fine-tuned model to Model Registry / Model Garden
   - Alternatively, upload the model to Hugging Face for community use
   - Document model architecture, training dataset, and performance metrics
   - Set up a Vertex AI endpoint for model serving
   - Configure scaling parameters and resource allocation
   - Deploy the model to the endpoint
   - Set up monitoring and logging for the deployed model

3. **Integration with Existing Application**
   - Update the API client code to support the new model endpoint
   - Add functionality to switch between base and fine-tuned models
   - Implement fallback mechanisms for endpoint failures
   - Create a benchmark system to compare model versions
   - Develop A/B testing framework to evaluate user satisfaction
   - Add user feedback collection to further improve the model

4. **Evaluation Framework**
   - Define evaluation metrics for the fine-tuned model:
     - Response relevance and quality
     - Domain-specific knowledge accuracy
     - Personality consistency
     - Response time and latency
   - Compare fine-tuned model performance against baseline
   - Gather user feedback through the application interface
   - Analyze improvements and areas for further refinement
   - Use the new evaluation framework to measure improvements:
     ```bash
     # Compare base model vs fine-tuned model
     cd evaluation
     ./evaluate_models.sh pairwise --model1 gemini-base --model2 gemini-tuned
     ```

### Implementation Timeline
- Week 3.1: Dataset acquisition and preprocessing
- Week 3.2: Fine-tuning execution and model validation
- Week 3.3: Model registry and endpoint deployment
- Week 3.4: Application integration and testing
- Week 3.5: Performance evaluation and documentation

### Success Criteria
- Fine-tuned model should:
  - Demonstrate improved domain expertise in travel and culture
  - Maintain consistent personality traits based on selected persona
  - Achieve higher relevance scores compared to the base model
  - Show reduced latency in real-world application usage
  - Receive positive user feedback on response quality
  - Show measurable improvements in the evaluation framework metrics


