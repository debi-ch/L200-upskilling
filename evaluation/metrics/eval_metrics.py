"""
Evaluation Metrics for Model Comparison

This module provides functions to evaluate model responses based on various metrics.
"""

import re
import time
import numpy as np
import json
from typing import Dict, List, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('eval_metrics')

# --- Basic Metrics ---

def response_length(response: str) -> int:
    """Calculate the length of the response in characters."""
    return len(response)

def word_count(response: str) -> int:
    """Calculate the number of words in the response."""
    return len(response.split())

def response_time(start_time: float, end_time: float) -> float:
    """Calculate the response time in seconds."""
    return end_time - start_time

# --- Content-Based Metrics ---

def contains_elements(response: str, elements: List[str], case_sensitive: bool = False) -> Tuple[float, List[str]]:
    """
    Check if the response contains specified elements (keywords, phrases).
    Returns the percentage of elements found and a list of found elements.
    """
    found_elements = []
    if not case_sensitive:
        response = response.lower()
        elements = [element.lower() for element in elements]
    
    for element in elements:
        if element in response:
            found_elements.append(element)
    
    percentage = len(found_elements) / len(elements) if elements else 0
    return percentage, found_elements

def factual_correctness(response: str, facts: Dict[str, str]) -> Dict[str, Any]:
    """
    Check if the response contains factually correct information.
    Requires a dictionary of facts to check against.
    """
    correct_facts = 0
    incorrect_facts = 0
    unmentioned_facts = 0
    
    for fact_key, fact_value in facts.items():
        # Check if the fact key is mentioned
        if fact_key.lower() in response.lower():
            # Check if the value is correctly associated
            if fact_value.lower() in response.lower():
                correct_facts += 1
            else:
                incorrect_facts += 1
        else:
            unmentioned_facts += 1
    
    return {
        "correct_facts": correct_facts,
        "incorrect_facts": incorrect_facts,
        "unmentioned_facts": unmentioned_facts,
        "accuracy": correct_facts / (correct_facts + incorrect_facts) if (correct_facts + incorrect_facts) > 0 else 0
    }

def relevance_score(response: str, query: str, keywords: List[str]) -> float:
    """
    Calculate a relevance score based on the presence of query-related keywords in the response.
    """
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    
    # Calculate overlap between query and response
    query_overlap = len(query_words.intersection(response_words)) / len(query_words) if query_words else 0
    
    # Calculate keyword presence
    keyword_score = sum(1 for keyword in keywords if keyword.lower() in response.lower()) / len(keywords) if keywords else 0
    
    # Weight: 40% query overlap, 60% keyword presence
    return 0.4 * query_overlap + 0.6 * keyword_score

# --- Structure and Quality Metrics ---

def has_structure(response: str) -> Dict[str, Any]:
    """
    Evaluate if the response has a clear structure (intro, body, conclusion).
    """
    # Check for intro (first 100 chars should not have markdown headings or lists)
    intro = not bool(re.search(r'^[#*-]', response[:100].strip()))
    
    # Check for body (contains paragraphs, possibly lists or headings)
    has_paragraphs = len(response.split('\n\n')) > 1
    has_lists = bool(re.search(r'[\n][*-]', response))
    has_headings = bool(re.search(r'[\n]#+\s', response))
    
    body = has_paragraphs or has_lists or has_headings
    
    # Check for conclusion (last 100 chars should summarize or conclude)
    conclusion_indicators = ['overall', 'conclusion', 'summary', 'finally', 'in summary', 'to summarize', 'enjoy', 'hope', 'remember']
    conclusion = any(indicator in response[-200:].lower() for indicator in conclusion_indicators)
    
    score = sum([intro, body, conclusion]) / 3
    
    return {
        "has_intro": intro,
        "has_body": body,
        "has_conclusion": conclusion,
        "structure_score": score
    }

def format_quality(response: str) -> Dict[str, Any]:
    """
    Evaluate the formatting quality of the response.
    """
    # Check for paragraphs
    paragraphs = response.split('\n\n')
    avg_paragraph_length = np.mean([len(p) for p in paragraphs]) if paragraphs else 0
    
    # Check for lists
    list_items = re.findall(r'[\n][*-]\s[^\n]+', response)
    
    # Check for headings
    headings = re.findall(r'[\n]#+\s[^\n]+', response)
    
    # Check for formatting elements
    bold_text = len(re.findall(r'\*\*[^*]+\*\*', response))
    italic_text = len(re.findall(r'\*[^*]+\*', response))
    
    # Calculate formatting diversity score
    format_elements = [bool(paragraphs), bool(list_items), bool(headings), bool(bold_text), bool(italic_text)]
    diversity_score = sum(format_elements) / len(format_elements)
    
    return {
        "paragraph_count": len(paragraphs),
        "avg_paragraph_length": avg_paragraph_length,
        "list_item_count": len(list_items),
        "heading_count": len(headings),
        "bold_text_count": bold_text,
        "italic_text_count": italic_text,
        "formatting_diversity_score": diversity_score
    }

# --- Travel-Specific Metrics ---

def travel_specific_content(response: str) -> Dict[str, Any]:
    """
    Evaluate how well the response addresses common travel considerations.
    """
    categories = {
        "attractions": ["visit", "see", "attraction", "sight", "landmark", "museum", "park", "temple", "church", "building"],
        "food": ["eat", "food", "restaurant", "cuisine", "dish", "meal", "drink", "cafe", "bar", "taste"],
        "accommodation": ["stay", "hotel", "hostel", "airbnb", "accommodation", "lodging", "apartment", "resort", "booking"],
        "transportation": ["transport", "get around", "subway", "bus", "train", "taxi", "car", "bike", "walk", "airport"],
        "budget": ["cost", "price", "cheap", "expensive", "budget", "money", "spend", "dollar", "euro", "currency"],
        "culture": ["culture", "local", "tradition", "custom", "history", "people", "language", "etiquette", "festival"],
        "practical_info": ["tip", "advice", "know", "information", "wifi", "safety", "weather", "season", "hour", "open"]
    }
    
    results = {}
    for category, keywords in categories.items():
        matches = 0
        for keyword in keywords:
            if keyword.lower() in response.lower():
                matches += 1
        score = matches / len(keywords)
        results[category] = score
    
    # Calculate an overall travel content score (average of all categories)
    results["overall_travel_score"] = sum(results.values()) / len(results)
    
    return results

def local_knowledge_indicators(response: str) -> Dict[str, Any]:
    """
    Detect indicators of local knowledge and specific recommendations.
    """
    # Patterns for specific recommendations
    specific_place_pattern = r'([A-Z][a-z]+\s*(?:[A-Z][a-z]+\s*)*(?:Street|Avenue|Park|Museum|Garden|Temple|Restaurant|Cafe|Market|Square|District|Plaza|Building))'
    specific_places = re.findall(specific_place_pattern, response)
    
    # Check for exact time/price references
    time_references = re.findall(r'\b(\d+(?::\d+)?(?:\s*[ap]m)?(?:\s*to\s*\d+(?::\d+)?(?:\s*[ap]m)?)?)\b', response.lower())
    price_references = re.findall(r'\$([\d,]+(?:\.\d+)?)', response)
    
    # Check for local terms (non-English words that might be local terms)
    # This is a simple heuristic and will have false positives
    word_pattern = r'\b([A-Za-z]+)\b'
    words = re.findall(word_pattern, response)
    non_english_words = [word for word in words if len(word) > 3 and not word.lower() in ['this', 'that', 'with', 'from', 'your', 'have', 'they', 'will', 'what', 'when', 'where', 'which', 'there', 'their', 'about', 'would', 'could', 'should']]
    
    local_terms_estimate = len(non_english_words) / len(words) if words else 0
    
    return {
        "specific_place_mentions": len(specific_places),
        "unique_specific_places": len(set(specific_places)),
        "time_references": len(time_references),
        "price_references": len(price_references),
        "estimated_local_terms_ratio": local_terms_estimate,
        "specificity_score": (len(specific_places) + len(time_references) + len(price_references)) / 10  # Normalized to 0-1 range
    }

# --- Comprehensive Evaluation Function ---

def evaluate_response(response: str, query: str, query_category: str = None, reference_data: Dict[str, Any] = None, start_time: float = None, end_time: float = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a model response.
    
    Args:
        response: The model's response text
        query: The query that was asked
        query_category: Category of the query (if available)
        reference_data: Optional reference data for factual evaluation
        start_time: Start time of the response generation
        end_time: End time of the response generation
        
    Returns:
        A dictionary with evaluation metrics
    """
    results = {}
    
    # Basic metrics
    results["basic"] = {
        "response_length": response_length(response),
        "word_count": word_count(response)
    }
    
    # Add response time if provided
    if start_time and end_time:
        results["basic"]["response_time"] = response_time(start_time, end_time)
    
    # Structure and formatting quality
    results["structure"] = has_structure(response)
    results["formatting"] = format_quality(response)
    
    # Content evaluation
    # Extract keywords from the query for relevance scoring
    query_keywords = [word.lower() for word in query.split() if len(word) > 3]
    results["content"] = {
        "relevance_score": relevance_score(response, query, query_keywords)
    }
    
    # Travel-specific evaluations
    results["travel_specific"] = travel_specific_content(response)
    results["local_knowledge"] = local_knowledge_indicators(response)
    
    # Factual correctness if reference data is provided
    if reference_data and "facts" in reference_data:
        results["factual"] = factual_correctness(response, reference_data["facts"])
    
    # Overall scores
    results["overall"] = {
        "response_quality": (
            results["structure"]["structure_score"] * 0.3 +
            results["formatting"]["formatting_diversity_score"] * 0.2 +
            results["content"]["relevance_score"] * 0.3 +
            results["travel_specific"]["overall_travel_score"] * 0.2
        ),
        "specificity": results["local_knowledge"]["specificity_score"]
    }
    
    return results 