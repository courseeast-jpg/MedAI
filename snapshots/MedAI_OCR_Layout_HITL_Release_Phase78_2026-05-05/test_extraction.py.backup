"""
Test script for Russian PDF extraction.
Validates extractor fix for stderr pollution issue.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from ingestion.pdf_pipeline import PDFPipeline

def main():
    """Run extraction test on Russian dietary PDF."""
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file")
        sys.exit(1)
    
    # Initialize pipeline
    print("Initializing PDF pipeline...")
    pipeline = PDFPipeline(api_key=api_key)
    
    # Test file path
    test_pdf = Path("data/uploads/Диета при заболеваниях жкт.pdf")
    
    if not test_pdf.exists():
        print(f"ERROR: Test file not found: {test_pdf}")
        print("Available files in data/uploads/:")
        uploads_dir = Path("data/uploads")
        if uploads_dir.exists():
            for f in uploads_dir.iterdir():
                print(f"  - {f.name}")
        sys.exit(1)
    
    print(f"\nProcessing: {test_pdf.name}")
    print(f"File size: {test_pdf.stat().st_size / 1024:.1f} KB")
    print("-" * 60)
    
    # Process PDF
    try:
        records = pipeline.process_chunked(str(test_pdf))
        
        print("-" * 60)
        print(f"\n✓ EXTRACTION COMPLETE")
        print(f"  Records extracted: {len(records)}")
        
        if records:
            print("\nSample entities:")
            for i, record in enumerate(records[:5], 1):
                entity_type = record.get('label', 'UNKNOWN')
                text = record.get('text', '')
                translation = record.get('translation', '')
                
                if translation:
                    print(f"  {i}. [{entity_type}] {text} → {translation}")
                else:
                    print(f"  {i}. [{entity_type}] {text}")
        else:
            print("\n⚠ WARNING: No records extracted")
            print("Check logs above for errors")
        
    except Exception as e:
        print(f"\n✗ EXTRACTION FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
