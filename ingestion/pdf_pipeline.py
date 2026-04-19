"""
MedAI v1.1 — PDF Ingestion Pipeline (Track B)
Uses docling for unified digital + scanned PDF processing.
Falls back to PyMuPDF + Tesseract if docling unavailable.
"""
import re
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
from app.schemas import MKBRecord, ExtractionOutput, FoodGuideOutput
from extraction.pii_stripper import PIIStripper
from extraction.extractor import Extractor

# Russian keywords that signal a food reference guide
_FOOD_RU_KEYWORDS = [
    'продукт', 'питание', 'пища', 'блюдо', 'рацион', 'продукты',
    'еда', 'калорий', 'белки', 'жиры', 'углеводы',
]
# Condition names that appear in food scoring tables
_FOOD_CONDITION_KEYWORDS = ['ibs', 'diverticulitis', 'oxalate', 'crystalluria', 'fodmap']


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
        Full pipeline: PDF → text → document type detection → PII-stripped → extracted → MKBRecords
        """
        session_id = session_id or str(uuid4())
        source_name = pdf_path.name

        stored_path = PDF_STORAGE_PATH / source_name
        if not stored_path.exists():
            shutil.copy2(pdf_path, stored_path)

        raw_text = self._extract_text(pdf_path)
        if not raw_text or len(raw_text.strip()) < 50:
            logger.warning(f"PDF {source_name}: insufficient text extracted")
            return []

        logger.info(f"PDF {source_name}: {len(raw_text)} chars extracted")

        # Detect document type before any extraction
        doc_type = self._detect_document_type(raw_text)
        logger.info(f"PDF {source_name}: document type detected as '{doc_type}'")

        if doc_type == "food_guide":
            return self._process_food_guide(raw_text, source_name, specialty, session_id)

        # Clinical processing: chunk large documents
        if len(raw_text) > 10000:
            logger.info(f"PDF {source_name}: large clinical document, using chunked extraction")
            return self._process_chunked_clinical(raw_text, source_name, specialty, session_id)

        stripped_text, pii_method = self.pii_stripper.strip(raw_text)
        is_clean, findings = self.pii_stripper.verify_clean(stripped_text)
        if not is_clean:
            logger.warning(f"PII audit: possible remnants in {source_name}: {findings}")

        extraction = self.extractor.extract(stripped_text, specialty)
        records = self._to_records(extraction, source_name, specialty, session_id)
        logger.info(f"PDF {source_name}: {len(records)} records extracted")
        return records

    # ── Document type detection ────────────────────────────────────────────────

    def _detect_document_type(self, text: str) -> str:
        """Classify the PDF as 'food_guide' or 'clinical_report' using lightweight heuristics."""
        text_lower = text.lower()

        food_ru_hits = sum(1 for kw in _FOOD_RU_KEYWORDS if kw in text_lower)
        food_cond_hits = sum(1 for kw in _FOOD_CONDITION_KEYWORDS if kw in text_lower)

        # A food scoring table contains many single-digit scores; look for dense numeric patterns
        score_matches = len(re.findall(r'\b[0-9]\b', text))

        if food_ru_hits >= 2 or food_cond_hits >= 2 or (food_ru_hits >= 1 and score_matches >= 20):
            return "food_guide"
        return "clinical_report"

    # ── Text splitting ─────────────────────────────────────────────────────────

    def _split_into_chunks(self, text: str, chunk_size: int = 6000) -> list[str]:
        """
        Split text for extraction.

        Priority order:
        1. PyMuPDF [Page N] markers — one chunk per page (natural boundary).
        2. Line-aligned size-based split — used for docling markdown output which
           has no page markers; splits at newline boundaries so table rows are
           never cut in half.
        """
        if '[Page ' in text:
            raw = text.split('[Page ')
            return ['[Page ' + c if i > 0 else c for i, c in enumerate(raw) if c.strip()]

        # Line-aligned split (safe for table rows)
        lines = text.split('\n')
        chunks: list[str] = []
        current_lines: list[str] = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for the newline
            if current_size + line_size > chunk_size and current_lines:
                chunks.append('\n'.join(current_lines))
                current_lines = [line]
                current_size = line_size
            else:
                current_lines.append(line)
                current_size += line_size

        if current_lines:
            chunks.append('\n'.join(current_lines))

        return [c for c in chunks if c.strip()]

    # ── Food guide pipeline ────────────────────────────────────────────────────

    def _process_food_guide(
        self, raw_text: str, source_name: str, specialty: str, session_id: str
    ) -> list[MKBRecord]:
        """Process all pages of a food reference guide with structured food extraction."""
        chunks = self._split_into_chunks(raw_text, chunk_size=6000)
        logger.info(f"Food guide '{source_name}': {len(chunks)} chunks to process")

        all_records: list[MKBRecord] = []
        seen_foods: set[str] = set()  # deduplicate across chunks by English name

        for i, chunk in enumerate(chunks):
            # doc_type="food_guide" → regex-only PII path; avoids Presidio NER
            # misidentifying food names ("Chicken", "Olive oil") as PERSON/LOCATION
            stripped_text, _ = self.pii_stripper.strip(chunk, doc_type="food_guide")
            food_output = self.extractor.extract_food_guide(stripped_text, specialty)

            records = self._food_guide_to_records(
                food_output, source_name, specialty, session_id, seen_foods
            )
            all_records.extend(records)
            logger.info(
                f"Food guide chunk {i + 1}/{len(chunks)}: "
                f"{len(food_output.foods)} foods, {len(records)} records"
            )

        logger.info(f"Food guide '{source_name}': {len(all_records)} total records")
        return all_records

    def _food_guide_to_records(
        self,
        food_output: FoodGuideOutput,
        source_name: str,
        specialty: str,
        session_id: str,
        seen_foods: set,
    ) -> list[MKBRecord]:
        records: list[MKBRecord] = []
        confidence = food_output.confidence
        method = food_output.extraction_method

        for food in food_output.foods:
            name_key = food.food_name.lower().strip()
            if not name_key:
                continue
            if name_key in seen_foods:
                continue
            seen_foods.add(name_key)

            # Build human-readable content string
            parts = [f"Food: {food.food_name}"]
            if food.food_name_ru:
                parts.append(f"(ru: {food.food_name_ru})")

            scores = []
            if food.ibs_score is not None:
                scores.append(f"IBS={food.ibs_score}")
            if food.diverticulitis_score is not None:
                scores.append(f"Diverticulitis={food.diverticulitis_score}")
            if food.oxalates_score is not None:
                scores.append(f"Oxalates={food.oxalates_score}")
            if food.crystalluria_score is not None:
                scores.append(f"Crystalluria={food.crystalluria_score}")
            if scores:
                parts.append("| Scores: " + ", ".join(scores))
            if food.safety_category:
                parts.append(f"| Safety: {food.safety_category}")

            tags = ["food_guide", "food_entry"]
            if food.safety_category:
                tags.append(food.safety_category.lower())

            records.append(MKBRecord(
                fact_type="note",
                content=" ".join(parts)[:500],
                structured=food.model_dump(exclude_none=True),
                specialty=specialty,
                source_type="document",
                source_name=source_name,
                trust_level=TRUST_CLINICAL,
                confidence=confidence,
                tier=TIER_ACTIVE,
                extraction_method=method,
                session_id=session_id,
                tags=tags,
            ))

        for note in food_output.general_notes:
            records.append(MKBRecord(
                fact_type="note",
                content=note[:500],
                structured={"text": note, "document_type": "food_guide"},
                specialty=specialty,
                source_type="document",
                source_name=source_name,
                trust_level=TRUST_CLINICAL,
                confidence=confidence * 0.8,
                tier=TIER_ACTIVE,
                extraction_method=method,
                session_id=session_id,
                tags=["food_guide", "general_note"],
            ))

        return records

    # ── Clinical chunked pipeline ──────────────────────────────────────────────

    def _process_chunked_clinical(
        self, full_text: str, source_name: str, specialty: str, session_id: str
    ) -> list[MKBRecord]:
        """Process large clinical documents in page-sized chunks."""
        chunks = self._split_into_chunks(full_text, chunk_size=6000)
        logger.info(f"Clinical '{source_name}': {len(chunks)} chunks to process")

        all_records: list[MKBRecord] = []
        for i, chunk in enumerate(chunks):
            stripped_text, _ = self.pii_stripper.strip(chunk)
            extraction = self.extractor.extract(stripped_text, specialty)
            records = self._to_records(
                extraction, f"{source_name} (chunk {i + 1})", specialty, session_id
            )
            all_records.extend(records)
            logger.info(f"Clinical chunk {i + 1}/{len(chunks)}: {len(records)} records")

        logger.info(f"Clinical '{source_name}': {len(all_records)} total records")
        return all_records

    # ── Text extraction ────────────────────────────────────────────────────────

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
                    text = self._ocr_page(page)
                pages_text.append(f"[Page {page_num + 1}]\n{text}")
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

    # ── Clinical record conversion ─────────────────────────────────────────────

    def _to_records(
        self,
        extraction: ExtractionOutput,
        source_name: str,
        specialty: str,
        session_id: str,
    ) -> list[MKBRecord]:
        records = []
        method = extraction.extraction_method
        confidence = extraction.confidence
        trust = TRUST_CLINICAL

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
                extraction_method=method,
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
                extraction_method=method,
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
                extraction_method=method,
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
                extraction_method=method,
                session_id=session_id,
            ))

        return records
