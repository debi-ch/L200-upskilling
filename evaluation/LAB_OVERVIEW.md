# Model Evaluation Framework

This directory contains tools and documentation for evaluating and comparing the performance of different large language models (LLMs) in the travel assistant application.

## Directory Contents

- **MODEL_EVALUATION_LAB.md**: The main lab guide for running evaluations
- **QUERY_CREATION_GUIDELINES.md**: Detailed guidelines for creating effective test queries
- **LAB_README.md**: Instructions for lab instructors
- **run_evaluation_lab.sh**: Script to run the complete evaluation process
- **evaluate_models.sh**: Shell script for running individual evaluation tasks
- **run_evaluation.py**: Python script implementing the evaluation logic
- **sample_visualization_code.py**: Example code for generating evaluation visualizations
- **data/**: Contains test queries and reference data
- **metrics/**: Contains evaluation metric implementations
- **results/**: Where evaluation results and charts are stored

## Quick Start

To quickly run a complete model evaluation:

```bash
# Make the script executable
chmod +x run_evaluation_lab.sh

# Run the automated evaluation
./run_evaluation_lab.sh
```

This will:
1. Verify your environment setup
2. Run pointwise evaluations for Gemini and Gemma models
3. Run a pairwise comparison of both models
4. Generate visualization charts
5. Display a summary of results

## Documentation

For more detailed information:

- **For participants**: Start with the [MODEL_EVALUATION_LAB.md](MODEL_EVALUATION_LAB.md) guide
- **For query creation**: See [QUERY_CREATION_GUIDELINES.md](QUERY_CREATION_GUIDELINES.md)
- **For instructors**: Refer to [LAB_README.md](LAB_README.md)

## Customizing the Evaluation

You can customize the evaluation framework by:

1. **Adding new test queries**: Follow the guidelines in QUERY_CREATION_GUIDELINES.md
2. **Creating custom metrics**: Add new functions to metrics/eval_metrics.py
3. **Evaluating other models**: Modify run_evaluation.py to support additional models
4. **Creating custom visualizations**: Use the sample_visualization_code.py as a reference

## Requirements

This framework requires:
- Python 3.6+
- Pandas and Matplotlib for data processing and visualization
- Access to Vertex AI and the Gemini and Gemma models
- Google Cloud authentication

## Troubleshooting

If you encounter issues:

1. **Check your Google Cloud authentication**
   ```bash
   gcloud auth list
   gcloud auth application-default login
   ```

2. **Verify the model access**
   ```bash
   # Test Gemini access
   python -c "from app.backend.models.gemini_chat_refactored import chat_with_gemini; print(chat_with_gemini('Hello'))"
   
   # Test Gemma access
   python -c "from app.backend.models.gemma_chat import chat_with_gemma; print(chat_with_gemma('Hello'))"
   ```

3. **Check Python path**
   ```bash
   # From the evaluation directory
   export PYTHONPATH="$PYTHONPATH:$(dirname $(pwd))"
   ```

## Examples

Example of running individual evaluation tasks:

```bash
# Run a pointwise evaluation with limited queries
./evaluate_models.sh pointwise --model gemini --limit 3

# Generate charts from existing results
./evaluate_models.sh visualize --file results/pairwise_comparison_YYYYMMDD_HHMMSS.json
```

## Results Structure

- **JSON files**: Contain detailed evaluation data for each query and response
- **CSV files**: Summary tables with key metrics
- **Chart directories**: Visual representations of the evaluation results 