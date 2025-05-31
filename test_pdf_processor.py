"""
Test script for PDF processing functionality
"""

from app.backend.rag.pdf_processor import PDFProcessor

def main():
    processor = PDFProcessor()
    
    # Process PDF from bucket
    pdf_path = "documents/Must-Have-guide-to-Florence-www.goinspired.com_.pdf"  # The PDF from the bucket
    print(f"\nProcessing PDF: {pdf_path}")
    
    chunks = processor.process_pdf(pdf_path)
    
    if chunks:
        print(f"\n✅ Successfully processed PDF into {len(chunks)} chunks")
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\nChunk {i}:")
            print(f"Text: {chunk['text'][:200]}...")
            print(f"Pages: {chunk['metadata']['pages']}")
            print(f"Sections: {len(chunk['metadata']['sections'])}")
    else:
        print("\n❌ Failed to process PDF")

if __name__ == "__main__":
    main() 