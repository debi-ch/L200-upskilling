"""
RAG Evaluation Module

This module evaluates the RAG system's performance using simple metrics that don't require external LLM APIs:
1. Answer Presence (checks if key information is present)
2. Context Utilization (checks if retrieved context was used)
3. Response Length (prefers concise answers)
4. Response Time (measures latency)
5. Custom Metrics (domain-specific checks)
"""

from typing import List, Dict, Any, Optional
import time
import re
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("rag_evaluator")

class RAGEvaluator:
    """Evaluates RAG system performance using simple metrics"""
    
    def __init__(self):
        self.metrics = {
            "answer_presence": self._check_answer_presence,
            "context_utilization": self._check_context_utilization,
            "response_length": self._check_response_length,
            "response_time": self._check_response_time
        }
    
    def _check_answer_presence(self, response: str, expected_phrases: List[str]) -> float:
        """Check if key phrases are present in the response"""
        if not expected_phrases:
            return 1.0
        matches = sum(1 for phrase in expected_phrases if phrase.lower() in response.lower())
        return matches / len(expected_phrases)
    
    def _check_context_utilization(self, response: str, context: str) -> float:
        """Check if the response uses information from the context"""
        if not context:
            return 1.0
        
        # Extract key phrases from context (simple approach using sentences)
        context_phrases = [s.strip() for s in context.split('.') if s.strip()]
        if not context_phrases:
            return 1.0
            
        # Check how many context phrases are reflected in response
        matches = sum(1 for phrase in context_phrases 
                     if any(word in response.lower() 
                           for word in phrase.lower().split()[:3]))
        return min(1.0, matches / len(context_phrases))
    
    def _check_response_length(self, response: str, ideal_length: int = 100) -> float:
        """Prefer responses close to ideal length"""
        current_length = len(response)
        # Score decreases as response deviates from ideal length
        return max(0.0, 1.0 - abs(current_length - ideal_length) / ideal_length)
    
    def _check_response_time(self, start_time: float, max_time: float = 2.0) -> float:
        """Check if response time is within acceptable range"""
        elapsed = time.time() - start_time
        return max(0.0, 1.0 - (elapsed / max_time))
    
    def evaluate(self, 
                query: str,
                response: str,
                context: str,
                expected_info: List[str] = None,
                start_time: float = None) -> Dict[str, float]:
        """
        Evaluate RAG response across all metrics
        
        Args:
            query: The user's question
            response: The RAG system's response
            context: The context provided to the RAG system
            expected_info: List of key phrases expected in response
            start_time: When the query processing started
        
        Returns:
            Dict of metric names to scores (0.0-1.0)
        """
        if start_time is None:
            start_time = time.time()
            
        scores = {
            "answer_presence": self._check_answer_presence(response, expected_info or []),
            "context_utilization": self._check_context_utilization(response, context),
            "response_length": self._check_response_length(response),
            "response_time": self._check_response_time(start_time)
        }
        
        # Overall score is average of individual metrics
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores

def create_test_cases() -> List[Dict[str, Any]]:
    """Create a set of test cases for RAG evaluation"""
    return [
        {
            "query": "Tell me about wine tours in Florence",
            "contexts": [
                "Florence is known for its wine tours in the Chianti region.",
                "Many tour operators offer day trips to Tuscan vineyards.",
                "Wine tours typically include tastings and vineyard visits."
            ],
            "expected_answer": "Florence offers various wine tours to the Chianti region, where visitors can enjoy tastings and vineyard visits through local tour operators.",
            "response": "Florence offers excellent wine tours to the nearby Chianti region. Local tour operators organize day trips that include wine tastings and visits to traditional Tuscan vineyards."
        },
        {
            "query": "What are the must-see attractions in Florence?",
            "contexts": [
                "The Uffizi Gallery houses Renaissance masterpieces.",
                "The Duomo cathedral is Florence's most iconic landmark.",
                "Ponte Vecchio is a famous medieval bridge with jewelry shops."
            ],
            "expected_answer": "Florence's must-see attractions include the Uffizi Gallery with Renaissance art, the iconic Duomo cathedral, and the historic Ponte Vecchio bridge.",
            "response": "When visiting Florence, you must see the Uffizi Gallery with its Renaissance masterpieces, the magnificent Duomo cathedral, and the historic Ponte Vecchio bridge lined with jewelry shops."
        }
    ]

if __name__ == "__main__":
    # Run some example evaluations
    evaluator = RAGEvaluator()
    test_cases = create_test_cases()
    
    print("\nEvaluating RAG System Performance...")
    results = evaluator.evaluate_batch(test_cases)
    
    print("\nIndividual Test Case Scores:")
    for i, scores in enumerate(results["individual_scores"], 1):
        print(f"\nTest Case {i}:")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.3f}")
    
    print("\nAverage Scores Across All Test Cases:")
    for metric, score in results["average_scores"].items():
        print(f"  {metric}: {score:.3f}") 