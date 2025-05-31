"""
Test script for PDF RAG functionality
"""

import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from app.backend.rag.pdf_rag import test_pdf_rag

if __name__ == "__main__":
    test_pdf_rag() 