"""
Model Evaluation Pipeline

This script runs evaluation of the Gemini and Gemma models in the chatbot application,
enabling both pointwise (individual model) and pairwise (model comparison) evaluations.
"""

import json
import time
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import logging

# Add the application path to import the model functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the models
from app.backend.models.gemini_chat_refactored import chat_with_gemini, set_model_preference
from app.backend.models.gemma_chat import chat_with_gemma

# Import the metrics
from metrics.eval_metrics import evaluate_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'results', 'evaluation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('evaluation')

# --- Evaluation Functions ---

def run_model(model_name: str, query: str) -> Tuple[str, float]:
    """
    Run a query through the specified model and time the response.
    
    Args:
        model_name: Name of the model ('gemini' or 'gemma')
        query: The query text to send to the model
        
    Returns:
        Tuple of (response_text, response_time)
    """
    start_time = time.time()
    
    try:
        if model_name.lower() == 'gemini':
            # Ensure we're using the base Gemini model, not fine-tuned
            set_model_preference(False)
            full_prompt = f"You are a helpful travel assistant. User: {query}\nAssistant:"
            response = chat_with_gemini(full_prompt)
        elif model_name.lower() == 'gemma':
            full_prompt = f"You are a helpful travel assistant. User: {query}\nAssistant:"
            response = chat_with_gemma(full_prompt)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        end_time = time.time()
        return response, end_time - start_time
        
    except Exception as e:
        end_time = time.time()
        logger.error(f"Error running {model_name} model: {str(e)}")
        return f"Error: {str(e)}", end_time - start_time

def evaluate_model_pointwise(model_name: str, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Perform pointwise evaluation of a model on multiple queries.
    
    Args:
        model_name: Name of the model to evaluate
        queries: List of query dictionaries
        
    Returns:
        List of evaluation results
    """
    results = []
    
    logger.info(f"Starting pointwise evaluation of {model_name} model on {len(queries)} queries")
    
    for i, query_dict in enumerate(queries):
        query = query_dict["query"]
        query_id = query_dict["query_id"]
        category = query_dict.get("category", "unknown")
        
        logger.info(f"[{i+1}/{len(queries)}] Evaluating {model_name} on query {query_id} ({category})")
        
        # Run the model
        response_text, response_time = run_model(model_name, query)
        
        # Evaluate the response
        eval_results = evaluate_response(
            response=response_text,
            query=query,
            query_category=category,
            start_time=0,  # We already calculated the time
            end_time=response_time
        )
        
        # Compile results
        result = {
            "model": model_name,
            "query_id": query_id,
            "category": category,
            "query": query,
            "response": response_text,
            "response_time": response_time,
            "evaluation": eval_results
        }
        
        results.append(result)
        
        # Short pause to avoid rate limits
        time.sleep(0.5)
    
    logger.info(f"Completed pointwise evaluation of {model_name} model")
    return results

def evaluate_models_pairwise(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Perform pairwise evaluation of Gemini and Gemma models on multiple queries.
    
    Args:
        queries: List of query dictionaries
        
    Returns:
        List of comparative evaluation results
    """
    results = []
    
    logger.info(f"Starting pairwise evaluation of Gemini vs. Gemma on {len(queries)} queries")
    
    for i, query_dict in enumerate(queries):
        query = query_dict["query"]
        query_id = query_dict["query_id"]
        category = query_dict.get("category", "unknown")
        
        logger.info(f"[{i+1}/{len(queries)}] Running pairwise evaluation on query {query_id} ({category})")
        
        # Run both models
        gemini_response, gemini_time = run_model("gemini", query)
        gemma_response, gemma_time = run_model("gemma", query)
        
        # Evaluate responses
        gemini_eval = evaluate_response(
            response=gemini_response,
            query=query,
            query_category=category,
            start_time=0,
            end_time=gemini_time
        )
        
        gemma_eval = evaluate_response(
            response=gemma_response,
            query=query,
            query_category=category,
            start_time=0,
            end_time=gemma_time
        )
        
        # Compile comparative results
        result = {
            "query_id": query_id,
            "category": category,
            "query": query,
            "gemini": {
                "response": gemini_response,
                "response_time": gemini_time,
                "evaluation": gemini_eval
            },
            "gemma": {
                "response": gemma_response,
                "response_time": gemma_time,
                "evaluation": gemma_eval
            },
            "comparison": {
                "faster_model": "gemini" if gemini_time < gemma_time else "gemma",
                "time_difference": abs(gemini_time - gemma_time),
                "quality_winner": "gemini" if gemini_eval["overall"]["response_quality"] > gemma_eval["overall"]["response_quality"] else "gemma",
                "quality_difference": abs(gemini_eval["overall"]["response_quality"] - gemma_eval["overall"]["response_quality"]),
                "specificity_winner": "gemini" if gemini_eval["overall"]["specificity"] > gemma_eval["overall"]["specificity"] else "gemma",
                "specificity_difference": abs(gemini_eval["overall"]["specificity"] - gemma_eval["overall"]["specificity"])
            }
        }
        
        results.append(result)
        
        # Short pause to avoid rate limits
        time.sleep(1)
    
    logger.info(f"Completed pairwise evaluation")
    return results

# --- Results Processing Functions ---

def save_results(results: List[Dict[str, Any]], filename: str):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: List of evaluation results
        filename: Name of the file to save to (without extension)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(os.path.dirname(__file__), 'results', f"{filename}_{timestamp}.json")
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {filepath}")
    return filepath

def generate_summary_table(results_file: str) -> pd.DataFrame:
    """
    Generate a summary table from evaluation results.
    
    Args:
        results_file: Path to the evaluation results JSON file
        
    Returns:
        DataFrame with summary statistics
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Determine if this is pointwise or pairwise results
    is_pairwise = "gemini" in results[0] and "gemma" in results[0]
    
    if is_pairwise:
        # Create summary for pairwise evaluation
        summary_data = []
        for result in results:
            summary_data.append({
                "Query ID": result["query_id"],
                "Category": result["category"],
                "Query": result["query"],
                "Gemini Response Time": result["gemini"]["response_time"],
                "Gemma Response Time": result["gemma"]["response_time"],
                "Gemini Quality": result["gemini"]["evaluation"]["overall"]["response_quality"],
                "Gemma Quality": result["gemma"]["evaluation"]["overall"]["response_quality"],
                "Gemini Specificity": result["gemini"]["evaluation"]["overall"]["specificity"],
                "Gemma Specificity": result["gemma"]["evaluation"]["overall"]["specificity"],
                "Faster Model": result["comparison"]["faster_model"],
                "Quality Winner": result["comparison"]["quality_winner"],
                "Specificity Winner": result["comparison"]["specificity_winner"]
            })
        
        # Add summary row with averages and counts
        gemini_wins_time = sum(1 for r in results if r["comparison"]["faster_model"] == "gemini")
        gemini_wins_quality = sum(1 for r in results if r["comparison"]["quality_winner"] == "gemini")
        gemini_wins_specificity = sum(1 for r in results if r["comparison"]["specificity_winner"] == "gemini")
        
        summary_row = {
            "Query ID": "SUMMARY",
            "Category": "",
            "Query": "",
            "Gemini Response Time": sum(r["gemini"]["response_time"] for r in results) / len(results),
            "Gemma Response Time": sum(r["gemma"]["response_time"] for r in results) / len(results),
            "Gemini Quality": sum(r["gemini"]["evaluation"]["overall"]["response_quality"] for r in results) / len(results),
            "Gemma Quality": sum(r["gemma"]["evaluation"]["overall"]["response_quality"] for r in results) / len(results),
            "Gemini Specificity": sum(r["gemini"]["evaluation"]["overall"]["specificity"] for r in results) / len(results),
            "Gemma Specificity": sum(r["gemma"]["evaluation"]["overall"]["specificity"] for r in results) / len(results),
            "Faster Model": f"Gemini: {gemini_wins_time}, Gemma: {len(results) - gemini_wins_time}",
            "Quality Winner": f"Gemini: {gemini_wins_quality}, Gemma: {len(results) - gemini_wins_quality}",
            "Specificity Winner": f"Gemini: {gemini_wins_specificity}, Gemma: {len(results) - gemini_wins_specificity}"
        }
        
        summary_data.append(summary_row)
        
    else:
        # Create summary for pointwise evaluation
        summary_data = []
        model_name = results[0]["model"]
        
        for result in results:
            summary_data.append({
                "Query ID": result["query_id"],
                "Category": result["category"],
                "Query": result["query"],
                "Response Time": result["response_time"],
                "Quality Score": result["evaluation"]["overall"]["response_quality"],
                "Specificity Score": result["evaluation"]["overall"]["specificity"],
                "Structure Score": result["evaluation"]["structure"]["structure_score"],
                "Formatting Score": result["evaluation"]["formatting"]["formatting_diversity_score"],
                "Travel Content Score": result["evaluation"]["travel_specific"]["overall_travel_score"]
            })
        
        # Add summary row with averages
        summary_row = {
            "Query ID": "SUMMARY",
            "Category": "",
            "Query": "",
            "Response Time": sum(r["response_time"] for r in results) / len(results),
            "Quality Score": sum(r["evaluation"]["overall"]["response_quality"] for r in results) / len(results),
            "Specificity Score": sum(r["evaluation"]["overall"]["specificity"] for r in results) / len(results),
            "Structure Score": sum(r["evaluation"]["structure"]["structure_score"] for r in results) / len(results),
            "Formatting Score": sum(r["evaluation"]["formatting"]["formatting_diversity_score"] for r in results) / len(results),
            "Travel Content Score": sum(r["evaluation"]["travel_specific"]["overall_travel_score"] for r in results) / len(results)
        }
        
        summary_data.append(summary_row)
    
    # Create a DataFrame from the summary data
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(os.path.dirname(__file__), 'results', f"summary_{timestamp}.csv")
    summary_df.to_csv(csv_file, index=False)
    
    logger.info(f"Saved summary table to {csv_file}")
    
    return summary_df

def generate_comparison_charts(results_file: str):
    """
    Generate comparison charts from pairwise evaluation results.
    
    Args:
        results_file: Path to the pairwise evaluation results JSON file
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Check if this is pairwise results
    if not ("gemini" in results[0] and "gemma" in results[0]):
        logger.error("Not a pairwise evaluation results file")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    charts_dir = os.path.join(os.path.dirname(__file__), 'results', f"charts_{timestamp}")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Response Time Comparison
    plt.figure(figsize=(12, 6))
    queries = [r["query_id"] for r in results]
    gemini_times = [r["gemini"]["response_time"] for r in results]
    gemma_times = [r["gemma"]["response_time"] for r in results]
    
    plt.bar(queries, gemini_times, width=0.4, label='Gemini', align='edge', alpha=0.7)
    plt.bar([x for x in range(len(queries))], gemma_times, width=-0.4, label='Gemma', align='edge', alpha=0.7)
    plt.xlabel('Query ID')
    plt.ylabel('Response Time (seconds)')
    plt.title('Response Time Comparison: Gemini vs Gemma')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'response_time_comparison.png'))
    
    # Quality Score Comparison
    plt.figure(figsize=(12, 6))
    gemini_quality = [r["gemini"]["evaluation"]["overall"]["response_quality"] for r in results]
    gemma_quality = [r["gemma"]["evaluation"]["overall"]["response_quality"] for r in results]
    
    plt.bar(queries, gemini_quality, width=0.4, label='Gemini', align='edge', alpha=0.7)
    plt.bar([x for x in range(len(queries))], gemma_quality, width=-0.4, label='Gemma', align='edge', alpha=0.7)
    plt.xlabel('Query ID')
    plt.ylabel('Quality Score')
    plt.title('Response Quality Comparison: Gemini vs Gemma')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'quality_comparison.png'))
    
    # Specificity Score Comparison
    plt.figure(figsize=(12, 6))
    gemini_specificity = [r["gemini"]["evaluation"]["overall"]["specificity"] for r in results]
    gemma_specificity = [r["gemma"]["evaluation"]["overall"]["specificity"] for r in results]
    
    plt.bar(queries, gemini_specificity, width=0.4, label='Gemini', align='edge', alpha=0.7)
    plt.bar([x for x in range(len(queries))], gemma_specificity, width=-0.4, label='Gemma', align='edge', alpha=0.7)
    plt.xlabel('Query ID')
    plt.ylabel('Specificity Score')
    plt.title('Response Specificity Comparison: Gemini vs Gemma')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'specificity_comparison.png'))
    
    # Travel Content Category Comparison
    plt.figure(figsize=(14, 8))
    
    # Extract categories
    categories = list(results[0]["gemini"]["evaluation"]["travel_specific"].keys())
    categories.remove("overall_travel_score")  # We'll consider this separately
    
    # Calculate average scores for each category
    gemini_category_scores = {cat: sum(r["gemini"]["evaluation"]["travel_specific"][cat] for r in results) / len(results) for cat in categories}
    gemma_category_scores = {cat: sum(r["gemma"]["evaluation"]["travel_specific"][cat] for r in results) / len(results) for cat in categories}
    
    x = range(len(categories))
    
    plt.bar([i - 0.2 for i in x], [gemini_category_scores[cat] for cat in categories], width=0.4, label='Gemini', alpha=0.7)
    plt.bar([i + 0.2 for i in x], [gemma_category_scores[cat] for cat in categories], width=0.4, label='Gemma', alpha=0.7)
    
    plt.xlabel('Travel Content Categories')
    plt.ylabel('Average Score')
    plt.title('Travel Content Category Comparison: Gemini vs Gemma')
    plt.xticks(x, [cat.replace("_", " ").title() for cat in categories], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'travel_category_comparison.png'))
    
    logger.info(f"Saved comparison charts to {charts_dir}")

# --- Main Function ---

def main():
    """
    Run the evaluation pipeline based on command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run model evaluation pipeline')
    parser.add_argument('--mode', choices=['pointwise', 'pairwise'], required=True,
                        help='Evaluation mode: pointwise (one model) or pairwise (comparing models)')
    parser.add_argument('--model', choices=['gemini', 'gemma'], 
                        help='Model to evaluate in pointwise mode')
    parser.add_argument('--queries', default='../evaluation/data/test_queries.json',
                        help='Path to the test queries JSON file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of queries to evaluate')
    parser.add_argument('--summarize', action='store_true',
                        help='Generate a summary table from results')
    parser.add_argument('--charts', action='store_true',
                        help='Generate comparison charts (pairwise only)')
    parser.add_argument('--results-file', 
                        help='Path to existing results file for summarize/charts options')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    
    # If just generating summary or charts from existing results
    if args.summarize or args.charts:
        if not args.results_file:
            logger.error("--results-file is required when using --summarize or --charts")
            return
        
        if args.summarize:
            summary_df = generate_summary_table(args.results_file)
            print("\nSummary Table:")
            print(summary_df.to_string())
        
        if args.charts and args.mode == 'pairwise':
            generate_comparison_charts(args.results_file)
        
        return
    
    # Load test queries
    try:
        with open(args.queries, 'r') as f:
            queries = json.load(f)
        
        if args.limit:
            queries = queries[:args.limit]
            
        logger.info(f"Loaded {len(queries)} test queries")
    except Exception as e:
        logger.error(f"Error loading test queries: {str(e)}")
        return
    
    # Run evaluation
    if args.mode == 'pointwise':
        if not args.model:
            logger.error("--model is required for pointwise evaluation")
            return
        
        results = evaluate_model_pointwise(args.model, queries)
        results_file = save_results(results, f"pointwise_{args.model}")
        
        if args.summarize:
            summary_df = generate_summary_table(results_file)
            print("\nSummary Table:")
            print(summary_df.to_string())
            
    elif args.mode == 'pairwise':
        results = evaluate_models_pairwise(queries)
        results_file = save_results(results, "pairwise_comparison")
        
        if args.summarize:
            summary_df = generate_summary_table(results_file)
            print("\nSummary Table:")
            print(summary_df.to_string())
        
        if args.charts:
            generate_comparison_charts(results_file)

if __name__ == "__main__":
    main() 