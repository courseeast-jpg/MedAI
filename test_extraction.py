"""
Test script for Russian PDF extraction.
Validates extractor fix for stderr pollution issue.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from extraction.extractor import Extractor
from extraction.pii_stripper import PIIStripper
from ingestion.pdf_pipeline import PDFPipeline

def main():
    """Run extraction test on Russian dietary PDF."""

    load_dotenv()

    print("Initializing PDF pipeline (Ollama backend)...")
    pipeline = PDFPipeline(extractor=Extractor(), pii_stripper=PIIStripper())
    
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
        records = pipeline.process(test_pdf)

        print("-" * 60)
        print(f"\n✓ EXTRACTION COMPLETE")
        print(f"  Records extracted: {len(records)}")

        if records:
            print("\nSample entities:")
            for i, record in enumerate(records[:5], 1):
                fact_type = getattr(record, "fact_type", "UNKNOWN")
                content = getattr(record, "content", "") or getattr(record, "text", "")
                print(f"  {i}. [{fact_type}] {content}")
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
