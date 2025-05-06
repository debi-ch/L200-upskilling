# Model Evaluation Pipeline

This directory contains tools for evaluating and comparing the performance of different language models in the chatbot application, particularly Gemini and Gemma models.

## Overview

The evaluation pipeline allows you to:

1. **Pointwise Evaluation**: Evaluate a single model's performance on a set of queries
2. **Pairwise Evaluation**: Compare two models (Gemini and Gemma) side-by-side on the same queries
3. **Visualize Results**: Generate summary tables and charts from evaluation results

## Directory Structure

- **data/**: Contains test data for evaluations
  - `test_queries.json`: A set of travel-related queries for testing the models
- **metrics/**: Contains evaluation metrics
  - `eval_metrics.py`: Metrics for evaluating model responses
- **results/**: Where evaluation results are stored
  - Results are saved with timestamps and include JSON data, CSV summaries, and charts
- `run_evaluation.py`: Main Python script for running evaluations
- `evaluate_models.sh`: Shell script for easier command-line usage

## Requirements

- Python 3.6+
- Required packages: pandas, matplotlib, numpy
- Access to the model APIs (Gemini and Gemma)

## Usage

### Using the Shell Script

The simplest way to run evaluations is using the included shell script:

```bash
# Run pairwise evaluation (Gemini vs Gemma)
./evaluate_models.sh pairwise

# Run pairwise evaluation with limited queries
./evaluate_models.sh pairwise --limit 3

# Run pointwise evaluation on a single model
./evaluate_models.sh pointwise --model gemini

# Generate a summary from existing results
./evaluate_models.sh summarize --file results/pairwise_comparison_20230815_123456.json

# Generate visualizations from existing results
./evaluate_models.sh visualize --file results/pairwise_comparison_20230815_123456.json
```

### Using Python Directly

For more control, you can run the Python script directly:

```bash
# Run pairwise evaluation
python run_evaluation.py --mode pairwise --summarize --charts

# Run pointwise evaluation
python run_evaluation.py --mode pointwise --model gemini --summarize

# Generate a summary from existing results
python run_evaluation.py --mode pointwise --summarize --results-file results/pointwise_gemini_20230815_123456.json
```

## Evaluation Metrics

The evaluation includes several categories of metrics:

1. **Basic Metrics**:
   - Response length
   - Word count
   - Response time

2. **Content-Based Metrics**:
   - Relevance score: How well the response addresses the query
   - Factual correctness (when reference data is available)

3. **Structure and Quality Metrics**:
   - Structure score: Presence of intro, body, conclusion
   - Formatting quality: Paragraphs, lists, headings, formatting

4. **Travel-Specific Metrics**:
   - Coverage of travel categories: attractions, food, accommodation, etc.
   - Local knowledge indicators: Specific place mentions, time/price references

5. **Overall Scores**:
   - Response quality: Combined score from structure, formatting, relevance, and travel content
   - Specificity: Indicates how detailed and specific the response is

## Example Workflow

1. Create or modify test queries in `data/test_queries.json`
2. Run a pairwise evaluation to compare Gemini and Gemma:
   ```bash
   ./evaluate_models.sh pairwise
   ```
3. Review the results in the generated CSV file and visualizations
4. Run a pointwise evaluation to get more detailed metrics for a specific model:
   ```bash
   ./evaluate_models.sh pointwise --model gemini
   ```

## Extending the Pipeline

To add new metrics:
1. Add new metric functions in `metrics/eval_metrics.py`
2. Reference them in the `evaluate_response()` function

To add new models:
1. Update the `run_model()` function in `run_evaluation.py`
2. Add a new case for the model in the appropriate evaluation functions

## Troubleshooting

- **ImportError**: Make sure PYTHONPATH includes the project root
- **API Errors**: Check your model configuration and credentials
- **Missing Results**: Check the `results/` directory and the evaluation log

## License

This evaluation pipeline is part of the chatbot project and follows the same licensing terms. 