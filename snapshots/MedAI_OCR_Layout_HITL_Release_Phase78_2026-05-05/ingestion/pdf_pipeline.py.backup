"""
MedAI v1.1 — PDF Ingestion Pipeline (Track B)
Uses docling for unified digital + scanned PDF processing.
Falls back to PyMuPDF + Tesseract if docling unavailable.
"""
from pathlib import Path
from typing import Optional
from loguru import logger
from uuid import uuid4
import shutil

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
    logger.info("docling available for PDF processing")
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("docling not available — falling back to PyMuPDF + Tesseract")

from app.config import PDF_STORAGE_PATH, TRUST_CLINICAL, TIER_ACTIVE
from app.schemas import MKBRecord, ExtractionOutput
from extraction.pii_stripper import PIIStripper
from extraction.extractor import Extractor


class PDFPipeline:
    def __init__(self, extractor: Extractor, pii_stripper: PIIStripper):
        self.extractor = extractor
        self.pii_stripper = pii_stripper
        PDF_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

        if DOCLING_AVAILABLE:
            self.converter = DocumentConverter()
        else:
            self.converter = None

    def process(
        self,
        pdf_path: Path,
        specialty: str = "general",
        session_id: str = "",
    ) -> list[MKBRecord]:
        """
        Full pipeline: PDF → text → PII-stripped → extracted → MKBRecords
        Returns list of MKBRecord ready for Quality Gate.
        """
        session_id = session_id or str(uuid4())
        source_name = pdf_path.name

        # Store original PDF locally
        stored_path = PDF_STORAGE_PATH / source_name
        if not stored_path.exists():
            shutil.copy2(pdf_path, stored_path)

        # Step 1: Extract text
        raw_text = self._extract_text(pdf_path)
        if not raw_text or len(raw_text.strip()) < 50:
            logger.warning(f"PDF {source_name}: insufficient text extracted")
            return []

        logger.info(f"PDF {source_name}: {len(raw_text)} chars extracted")

        # Step 2: Check if document is large (> 10,000 chars) - use chunked extraction
        if len(raw_text) > 10000:
            logger.info(f"PDF {source_name}: Large document detected. Using chunked extraction.")
            return self._process_chunked(raw_text, source_name, specialty, session_id)

        # Standard processing for smaller documents
        # Step 2: Strip PII (BEFORE any external call)
        stripped_text, pii_method = self.pii_stripper.strip(raw_text)
        is_clean, findings = self.pii_stripper.verify_clean(stripped_text)
        if not is_clean:
            logger.warning(f"PII audit: possible remnants in {source_name}: {findings}")

        # Step 3: Extract entities (on stripped text)
        extraction = self.extractor.extract(stripped_text, specialty)

        # Step 4: Convert to MKBRecords
        records = self._to_records(extraction, source_name, specialty, session_id)
        logger.info(f"PDF {source_name}: {len(records)} records extracted")
        return records

    def _process_chunked(self, full_text: str, source_name: str, specialty: str, session_id: str) -> list[MKBRecord]:
        """Process large documents in chunks to avoid Gemini JSON errors."""
        # Split by page markers
        chunks = full_text.split('[Page ')
        all_records = []
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            # Add page marker back
            if i > 0:
                chunk = '[Page ' + chunk
            
            logger.info(f"Processing chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
            
            # Strip PII
            stripped_text, _ = self.pii_stripper.strip(chunk)
            
            # Extract entities
            extraction = self.extractor.extract(stripped_text, specialty)
            
            # Convert to records
            records = self._to_records(extraction, f"{source_name} (page {i+1})", specialty, session_id)
            all_records.extend(records)
            
            logger.info(f"Chunk {i+1}: {len(records)} records extracted")
        
        logger.info(f"Total from chunked processing: {len(all_records)} records")
        return all_records

    def _extract_text(self, pdf_path: Path) -> str:
        if DOCLING_AVAILABLE:
            return self._extract_docling(pdf_path)
        return self._extract_pymupdf(pdf_path)

    def _extract_docling(self, pdf_path: Path) -> str:
        try:
            result = self.converter.convert(str(pdf_path))
            return result.document.export_to_markdown()
        except Exception as e:
            logger.warning(f"docling failed: {e}. Trying PyMuPDF.")
            return self._extract_pymupdf(pdf_path)

    def _extract_pymupdf(self, pdf_path: Path) -> str:
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            pages_text = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if len(text.strip()) < 50:
                    # Scanned page — try OCR
                    text = self._ocr_page(page)
                pages_text.append(f"[Page {page_num+1}]\n{text}")
            doc.close()
            return "\n".join(pages_text)
        except Exception as e:
            logger.error(f"PyMuPDF failed: {e}")
            return ""

    def _ocr_page(self, page) -> str:
        try:
            import pytesseract
            from PIL import Image
            import io
            mat = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(mat.tobytes("png")))
            return pytesseract.image_to_string(img)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    def _to_records(
        self,
        extraction: ExtractionOutput,
        source_name: str,
        specialty: str,
        session_id: str,
    ) -> list[MKBRecord]:
        records = []
        extraction_method = extraction.extraction_method
        confidence = extraction.confidence

        # Trust level: if method is rules_based, cap at trust=2
        # Clinical PDFs are trust=1 only for signed documents
        trust = TRUST_CLINICAL  # documents default to trust=1

        for diag in extraction.diagnoses:
            records.append(MKBRecord(
                fact_type="diagnosis",
                content=f"Diagnosis: {diag.name}" + (f" ({diag.date})" if diag.date else ""),
                structured=diag.model_dump(exclude_none=True),
                specialty=specialty,
                source_type="document",
                source_name=source_name,
                trust_level=trust,
                confidence=confidence,
                tier=TIER_ACTIVE,
                extraction_method=extraction_method,
                session_id=session_id,
            ))

        for med in extraction.medications:
            dose_str = f" {med.dose}" if med.dose else ""
            freq_str = f" {med.frequency}" if med.frequency else ""
            records.append(MKBRecord(
                fact_type="medication",
                content=f"Medication: {med.name}{dose_str}{freq_str}",
                structured=med.model_dump(exclude_none=True),
                specialty=specialty,
                source_type="document",
                source_name=source_name,
                trust_level=trust,
                confidence=confidence,
                tier=TIER_ACTIVE,
                extraction_method=extraction_method,
                ddi_checked=False,
                session_id=session_id,
                tags=["medication"],
            ))

        for test in extraction.test_results:
            val_str = f": {test.value} {test.unit or ''}".rstrip() if test.value else ""
            records.append(MKBRecord(
                fact_type="test_result",
                content=f"Test: {test.test_name}{val_str}",
                structured=test.model_dump(exclude_none=True),
                specialty=specialty,
                source_type="document",
                source_name=source_name,
                trust_level=trust,
                confidence=confidence,
                tier=TIER_ACTIVE,
                extraction_method=extraction_method,
                session_id=session_id,
            ))

        for note in extraction.notes:
            records.append(MKBRecord(
                fact_type="note",
                content=note[:500],
                structured={"text": note},
                specialty=specialty,
                source_type="document",
                source_name=source_name,
                trust_level=trust,
                confidence=confidence * 0.8,
                tier=TIER_ACTIVE,
                extraction_method=extraction_method,
                session_id=session_id,
            ))

        return records
