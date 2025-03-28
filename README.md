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

Prompt Management:
- Support for multiple prompt versions per model
- UI-based prompt version control
- Ability to view prompt history
- Easy switching between different prompts
- Specialized travel agent prompts for each model

Recent Updates:
- Migrated to Vertex AI SDK
- Implemented prompt version control system
- Added specialized travel agent personas
- Enhanced error handling and debugging
- Added support for both Gemini and Gemma models
- Updated deployment process for new dependencies

## Deployment
The chatbot is currently deployed and accessible at:
https://chatbot-app-708208532564.us-central1.run.app

## Next Steps (Week 3)
### Current Challenges
The Gemma 9B-IT model, while powerful, currently exhibits some behavioral issues:
- Tends to ask multiple follow-up questions instead of providing direct answers
- Sometimes struggles to maintain the Mexican Pirate persona while giving substantive responses
- May need better guidance on when to ask questions vs. when to provide information

### Model Fine-tuning Focus
1. **Parameter Optimization (First Priority)**
   - Current parameters:
     ```python
     parameters = {
         "maxOutputTokens": 2048,
         "temperature": 0.7,
         "topP": 0.95,
         "topK": 40
     }
     ```
   - Plan to test:
     - Lower temperature (0.3-0.5) for more focused responses
     - Adjusted topP and topK values
     - Different max output token lengths
   - Track impact on response directness and persona consistency

2. **Prompt Engineering (Second Priority)**
   - Current prompt structure may need reinforcement on:
     - When to ask vs. when to answer
     - Balancing character roleplay with information delivery
     - Handling partial information scenarios
   - Will test variations of the pirate persona prompt with:
     - Explicit instruction about question-asking behavior
     - Examples of good response patterns
     - Clearer decision trees for information gathering

3. **Fine-tuning Pipeline (If Needed)**
   - Prepare training data from successful interactions
   - Focus on examples where direct answers were provided
   - Include varied scenarios:
     - Complete information available → direct answer
     - Partial information → answer what's known, then ask specific questions
     - Missing critical information → focused clarifying questions

4. **Evaluation Framework**
   - Metrics to track:
     - Question-to-answer ratio in responses
     - Persona consistency score
     - Response relevance and completeness
     - User satisfaction ratings
   - A/B testing between different parameter sets and prompts
   - Regular review of chat logs for behavioral patterns

5. **Implementation Roadmap**
   - Week 3.1: Parameter optimization and testing
   - Week 3.2: Prompt engineering and refinement
   - Week 3.3: Evaluation framework setup
   - Week 3.4: Fine-tuning if previous steps insufficient
   - Week 3.5: Final optimization and documentation

### Success Criteria
- Gemma model should:
  - Provide direct answers when sufficient information is available
  - Maintain Mexican Pirate persona while being informative
  - Ask questions only when critical information is missing
  - Keep follow-up questions focused and minimal

