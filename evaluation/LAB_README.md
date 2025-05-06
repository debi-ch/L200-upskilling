# LLM Model Evaluation Lab

## Instructor Guide

This document provides instructions for lab instructors on how to prepare and deliver the Model Evaluation Lab. The lab demonstrates how to evaluate and compare different large language models using a structured framework.

## Lab Overview

**Duration**: 2-3 hours

**Target Audience**: ML engineers, data scientists, and AI practitioners interested in LLM evaluation

**Prerequisites**:
- Basic Python knowledge
- Understanding of LLMs
- Google Cloud Platform account with Vertex AI enabled

## Lab Materials

This lab includes the following components:

1. **LAB_README.md** (this file): Instructions for the instructor
2. **MODEL_EVALUATION_LAB.md**: Step-by-step lab guide for participants
3. **run_evaluation_lab.sh**: Script to run the complete evaluation process
4. **sample_visualization_code.py**: Script to generate example visualizations
5. **evaluate_models.sh**: Core evaluation script
6. **run_evaluation.py**: Python implementation of the evaluation pipeline
7. **metrics/eval_metrics.py**: Implementation of evaluation metrics
8. **data/test_queries.json**: Test queries for the evaluation

## Instructor Preparation

Before delivering the lab, follow these steps:

1. **Set up the environment**:
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Verify model access**:
   - Ensure the Gemini and Gemma models are accessible through the appropriate APIs
   - Verify that your GCP project has the Vertex AI API enabled

3. **Test the evaluation pipeline**:
   ```bash
   cd evaluation
   ./run_evaluation_lab.sh
   ```
   
4. **Review sample visualizations**:
   - Run the sample visualization script to become familiar with the output
   ```bash
   python sample_visualization_code.py
   ```

5. **Prepare for common questions**:
   - How to interpret the different metrics
   - Ways to extend the evaluation for other models
   - How to create custom metrics
   - Troubleshooting API and authentication issues

## Delivering the Lab

### Introduction (15 minutes)

1. Introduce the concept of model evaluation and its importance in AI development
2. Explain the difference between pointwise and pairwise evaluations
3. Show examples of evaluation metrics and what they measure
4. Discuss the structure of the lab and what participants will learn

### Environment Setup (15 minutes)

1. Help participants set up their environment:
   - Clone the repository
   - Create a virtual environment
   - Install requirements
   - Authenticate with Google Cloud

2. Verify that everyone has access to the necessary models

### Understanding the Framework (30 minutes)

1. Walk through the components of the evaluation framework:
   - Test queries
   - Metrics
   - Execution scripts

2. Explain how the metrics are calculated and what they measure

3. Show the structure of the evaluation results

### Running Evaluations (60 minutes)

1. Guide participants through running a pointwise evaluation:
   ```bash
   ./evaluate_models.sh pointwise --model gemini
   ```

2. Have participants run a pointwise evaluation for the Gemma model:
   ```bash
   ./evaluate_models.sh pointwise --model gemma
   ```

3. Lead participants through running a pairwise evaluation:
   ```bash
   ./evaluate_models.sh pairwise
   ```

### Analyzing Results (30 minutes)

1. Help participants interpret the summary tables
2. Walk through the visualization charts
3. Discuss what the results tell us about the models' strengths and weaknesses

### Extension (Optional) (30 minutes)

1. Guide interested participants in extending the evaluation:
   - Adding new test queries
   - Creating custom metrics
   - Evaluating additional models

### Wrap-up and Q&A (15 minutes)

1. Summarize key takeaways
2. Discuss practical applications of model evaluation
3. Address any remaining questions

## Troubleshooting Guide

### Common Issues and Solutions

1. **API Authentication Errors**:
   - Check that participants have run `gcloud auth login` and `gcloud auth application-default login`
   - Verify the project has the Vertex AI API enabled

2. **Model Access Errors**:
   - Ensure the models are available in the selected region
   - Check IAM permissions for the account

3. **Environment Issues**:
   - Verify Python version compatibility (3.6+)
   - Check for missing dependencies

4. **Slow Responses**:
   - Network or API rate limit issues may cause slow responses
   - Suggest using the `--limit` option to reduce the number of queries during testing

## Additional Resources

Provide these resources to participants who want to learn more:

1. Google Vertex AI Documentation: https://cloud.google.com/vertex-ai/docs
2. LLM Evaluation Best Practices: https://www.deeplearning.ai/short-courses/evaluating-debugging-llm-systems/
3. Metrics for LLM Evaluation: https://huggingface.co/blog/evaluating-llm-metrics
4. Gemini and Gemma Model Documentation: https://ai.google.dev/docs

## Lab Customization

To customize this lab for different needs:

1. **Different Models**:
   - Update `run_evaluation.py` to support additional models
   - Modify the model selection logic in `evaluate_models.sh`

2. **Custom Metrics**:
   - Add new metric functions to `metrics/eval_metrics.py`
   - Update the `evaluate_response()` function to include the new metrics

3. **Alternative Datasets**:
   - Replace the queries in `data/test_queries.json` with domain-specific examples
   - Update the categories and difficulty levels as needed 