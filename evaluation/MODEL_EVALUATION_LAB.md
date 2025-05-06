# Model Evaluation Lab Guide

This lab guide will walk you through the process of evaluating and comparing the performance of different large language models (LLMs) using a structured evaluation framework. By the end of this lab, you'll understand how to run both pointwise (single model) and pairwise (comparing models) evaluations, and how to interpret the results.

## Prerequisites

Before starting this lab, ensure you have:

1. Access to a Google Cloud Platform (GCP) project with the Vertex AI API enabled
2. Python 3.6+ installed
3. Required Python packages (installed via requirements.txt)
4. Models configured in your application (Gemini and Gemma)

## Setting Up Your Environment

You'll use the existing L200-upskilling repository for this lab, as it already contains all the necessary components for model evaluation:

```bash
# If you haven't already cloned the repository
git clone https://github.com/your-org/L200-upskilling.git
cd L200-upskilling

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login
```

## Step 1: Understanding the Evaluation Framework

The evaluation framework consists of several components:

- **Test Queries**: Predefined questions that will be sent to the models
- **Evaluation Metrics**: Measurements that assess various aspects of model responses
- **Execution Scripts**: Tools for running evaluations and generating results

Let's look at each component:

### Test Queries

The test queries are stored in `evaluation/data/test_queries.json`. Each query has:
- A unique identifier
- The query text
- A category (e.g., "destination_recommendations", "safety")
- A difficulty level

You can examine the queries by opening this file:

```bash
cat evaluation/data/test_queries.json
```

For detailed guidelines on creating your own test queries, see the `QUERY_CREATION_GUIDELINES.md` file in the evaluation directory.

### Evaluation Metrics

The evaluation metrics are defined in `evaluation/metrics/eval_metrics.py`. These include:

1. **Basic Metrics**: Response length, word count, response time
2. **Content Metrics**: Relevance to the query, factual correctness
3. **Structure Metrics**: Organization of the response (intro, body, conclusion)
4. **Quality Metrics**: Formatting, use of paragraphs, lists, headings
5. **Travel-Specific Metrics**: Categories covered, specificity of recommendations

### Execution Scripts

The main scripts for running evaluations are:
- `evaluate_models.sh`: A shell script that simplifies running common evaluation tasks
- `run_evaluation.py`: The Python script that powers the evaluations

## Step 2: Running a Pointwise Evaluation

A pointwise evaluation assesses a single model's performance on a set of queries. Let's start by evaluating the Gemini model:

```bash
cd evaluation
./evaluate_models.sh pointwise --model gemini
```

This command:
1. Loads the test queries from `data/test_queries.json`
2. Sends each query to the Gemini model
3. Records the model's response and response time
4. Evaluates the response using the metrics in `metrics/eval_metrics.py`
5. Saves the results to a JSON file in the `results/` directory
6. Generates a summary CSV file

You should see output indicating which queries are being processed:

```
Running pointwise evaluation for gemini...
2025-XX-XX XX:XX:XX - evaluation - INFO - Loaded 10 test queries
2025-XX-XX XX:XX:XX - evaluation - INFO - Starting pointwise evaluation of gemini model on 10 queries
2025-XX-XX XX:XX:XX - evaluation - INFO - [1/10] Evaluating gemini on query q1 (destination_recommendations)
...
```

The results are saved in two files:
- A detailed JSON file: `results/pointwise_gemini_YYYYMMDD_HHMMSS.json`
- A summary CSV file: `results/summary_YYYYMMDD_HHMMSS.csv`

## Step 3: Running a Pointwise Evaluation for Another Model

Now let's evaluate the Gemma model using the same queries:

```bash
./evaluate_models.sh pointwise --model gemma
```

This follows the same process as before, but with the Gemma model. You'll again get a JSON file and a summary CSV for this model.

## Step 4: Running a Pairwise Evaluation

A pairwise evaluation directly compares two models on the same set of queries. Run a pairwise evaluation:

```bash
./evaluate_models.sh pairwise
```

This command:
1. Loads the test queries from `data/test_queries.json`
2. For each query:
   - Sends it to both the Gemini and Gemma models
   - Records responses and response times for both models
   - Evaluates both responses using the metrics in `metrics/eval_metrics.py`
   - Determines which model performed better on different metrics
3. Saves the results to a JSON file in the `results/` directory
4. Generates a summary CSV file and comparison charts

You should see output similar to:

```
Running pairwise evaluation...
2025-XX-XX XX:XX:XX - evaluation - INFO - Loaded 10 test queries
2025-XX-XX XX:XX:XX - evaluation - INFO - Starting pairwise evaluation of Gemini vs. Gemma on 10 queries
2025-XX-XX XX:XX:XX - evaluation - INFO - [1/10] Running pairwise evaluation on query q1 (destination_recommendations)
...
```

The results include:
- A detailed JSON file: `results/pairwise_comparison_YYYYMMDD_HHMMSS.json`
- A summary CSV file: `results/summary_YYYYMMDD_HHMMSS.csv`
- Comparison charts in the `results/charts_YYYYMMDD_HHMMSS/` directory

## Step 5: Interpreting the Results

### Understanding the Summary Tables

The summary CSV files contain metrics for each query, including:

- **Response Time**: How long the model took to generate a response (in seconds)
- **Quality Score**: A combined score (0-1) of structure, formatting, relevance, and travel content
- **Specificity Score**: How detailed and specific the response is (0-1)
- **Structure Score**: How well the response is organized with intro, body, conclusion (0-1)
- **Formatting Score**: Quality of formatting with paragraphs, lists, headings (0-1)
- **Travel Content Score**: How well the response covers travel-specific categories (0-1)

For pairwise evaluations, the summary also includes:
- Which model was faster
- Which model had the higher quality score
- Which model gave more specific responses

### Analyzing the Charts

For pairwise evaluations, charts are generated to visualize:

1. **Response Time Comparison**: Bar chart showing response times for both models
2. **Quality Score Comparison**: Bar chart comparing quality scores
3. **Specificity Comparison**: Bar chart comparing specificity scores
4. **Category Breakdown**: Radar chart showing strengths in different categories

## Step 6: Running Limited Evaluations (Optional)

To run evaluations on a subset of queries (useful for testing):

```bash
# Run pairwise evaluation with only 3 queries
./evaluate_models.sh pairwise --limit 3

# Run pointwise evaluation with only 3 queries
./evaluate_models.sh pointwise --model gemini --limit 3
```

## Step 7: Generating Summaries from Existing Results (Optional)

If you have existing results files, you can generate summaries:

```bash
# Generate a summary from an existing pointwise results file
./evaluate_models.sh summarize --file results/pointwise_gemini_YYYYMMDD_HHMMSS.json

# Generate charts from an existing pairwise results file
./evaluate_models.sh visualize --file results/pairwise_comparison_YYYYMMDD_HHMMSS.json
```

## Step 8: Extending the Evaluation (Advanced)

### Adding New Test Queries

To add new test queries:
1. Edit `data/test_queries.json` or create a new file with your custom queries
2. Follow the format guidelines in `QUERY_CREATION_GUIDELINES.md`
3. Ensure each query has a unique `query_id`, a clear question text, a relevant category, and an appropriate difficulty level

### Adding New Metrics

To add new evaluation metrics:
1. Edit `metrics/eval_metrics.py`
2. Add a new metric function
3. Update the `evaluate_response()` function to include your new metric

## Common Issues and Troubleshooting

### API Rate Limits
If you see errors about rate limits, try increasing the pause between queries:
- In `run_evaluation.py`, find the `time.sleep()` calls and increase the duration

### Model API Errors
If you encounter errors with the model APIs:
- Check your authentication with `gcloud auth list`
- Ensure the Vertex AI API is enabled in your project
- Verify your model configuration

### Python Environment Issues
If you see import errors:
- Ensure you've activated your virtual environment
- Check that all required packages are installed with `pip list`

## Conclusion

In this lab, you've learned how to:
1. Run pointwise evaluations on individual models
2. Run pairwise evaluations to compare models
3. Interpret the evaluation results
4. Extend the evaluation framework with new queries and metrics

The evaluation results provide valuable insights into model performance, helping you make data-driven decisions about which model to use for different scenarios.

For more advanced usage, explore the Python code in `run_evaluation.py` and `metrics/eval_metrics.py` and review the comprehensive query creation guidelines in `QUERY_CREATION_GUIDELINES.md`. 