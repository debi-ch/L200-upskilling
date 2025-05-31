"""
Comprehensive RAG System Evaluation

This script evaluates all components of our RAG system:
1. Text RAG (chat history/preferences)
2. PDF RAG
3. Multimodal RAG

Using simple metrics that don't require external LLM APIs:
- Answer Presence
- Context Utilization
- Response Length
- Response Time
"""

import os
import time
from typing import Dict, Any, List
from app.backend.rag.rag_engine import RAGEngine
from app.backend.rag.pdf_rag import PDFRagEngine
from app.backend.rag.multimodal_rag import MultimodalRAG
from app.evaluation.rag_evaluator import RAGEvaluator
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("rag_evaluation")

def create_text_rag_test_cases() -> List[Dict[str, Any]]:
    """Create test cases for text-based RAG"""
    return [
        {
            "query": "What are some good hotels in Florence?",
            "expected_info": [
                "Hotel Lungarno",
                "luxury",
                "Arno River",
                "Ponte Vecchio"
            ],
            "context": """
            The Hotel Lungarno is a luxury hotel located on the Arno River. 
            It offers stunning views of the Ponte Vecchio and features elegant rooms 
            with antique furniture. The hotel's restaurant, Borgo San Jacopo, 
            has been awarded a Michelin star.
            """
        },
        {
            "query": "What are the must-see museums in Florence?",
            "expected_info": [
                "Uffizi Gallery",
                "Renaissance art",
                "Botticelli",
                "Michelangelo"
            ],
            "context": """
            The Uffizi Gallery is one of the most famous museums in Florence and Italy.
            It houses an incredible collection of Renaissance art, including works by
            Botticelli and Michelangelo. The museum was originally built in 1560.
            """
        }
    ]

def create_pdf_rag_test_cases() -> List[Dict[str, Any]]:
    """Create test cases for PDF-based RAG"""
    return [
        {
            "query": "What are the best times to visit the Duomo?",
            "expected_info": [
                "early morning",
                "avoid crowds",
                "8:30 AM",
                "weekdays"
            ],
            "context": """
            The best time to visit Florence's Duomo is early morning, right when it
            opens at 8:30 AM. This helps you avoid the crowds that form later in the day.
            Weekdays are generally less crowded than weekends.
            """
        }
    ]

def create_multimodal_rag_test_cases() -> List[Dict[str, Any]]:
    """Create test cases for multimodal RAG"""
    return [
        {
            "query": "Describe the Ponte Vecchio bridge",
            "expected_info": [
                "medieval stone bridge",
                "jewelry shops",
                "Arno River",
                "gold"
            ],
            "context": """
            The Ponte Vecchio is a medieval stone bridge spanning the Arno River.
            It is famous for the jewelry shops built along its edges. The bridge
            has been home to gold merchants since the 16th century.
            """
        }
    ]

def evaluate_rag_component(
    rag_engine: Any,
    test_cases: List[Dict[str, Any]],
    evaluator: RAGEvaluator
) -> Dict[str, float]:
    """
    Evaluate a RAG component using the provided test cases
    
    Args:
        rag_engine: The RAG engine to evaluate
        test_cases: List of test cases
        evaluator: RAGEvaluator instance
    
    Returns:
        Dictionary of average scores for each metric
    """
    all_scores = []
    
    for test_case in test_cases:
        # Record start time for latency measurement
        start_time = time.time()
        
        # Get RAG response
        result = rag_engine.query(test_case["query"])
        response = result.get("response", "")
        
        # Evaluate response
        scores = evaluator.evaluate(
            query=test_case["query"],
            response=response,
            context=test_case["context"],
            expected_info=test_case["expected_info"],
            start_time=start_time
        )
        
        all_scores.append(scores)
        
        # Log individual test results
        logger.info(f"\nTest Case: {test_case['query']}")
        logger.info(f"Response: {response}")
        logger.info("Scores:")
        for metric, score in scores.items():
            logger.info(f"  {metric}: {score:.3f}")
    
    # Calculate average scores
    avg_scores = {}
    for metric in all_scores[0].keys():
        avg_scores[metric] = sum(s[metric] for s in all_scores) / len(all_scores)
    
    return avg_scores

def main():
    """Run comprehensive RAG evaluation"""
    
    # Initialize components
    evaluator = RAGEvaluator()
    text_rag = RAGEngine()
    pdf_rag = PDFRagEngine()
    multimodal_rag = MultimodalRAG()
    
    # Evaluate each component
    components = [
        ("Text RAG", text_rag, create_text_rag_test_cases()),
        ("PDF RAG", pdf_rag, create_pdf_rag_test_cases()),
        ("Multimodal RAG", multimodal_rag, create_multimodal_rag_test_cases())
    ]
    
    for name, engine, test_cases in components:
        logger.info(f"\nEvaluating {name}...")
        avg_scores = evaluate_rag_component(engine, test_cases, evaluator)
        
        logger.info(f"\n{name} Average Scores:")
        for metric, score in avg_scores.items():
            logger.info(f"  {metric}: {score:.3f}")

if __name__ == "__main__":
    main() 